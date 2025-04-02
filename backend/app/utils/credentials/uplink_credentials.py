from datetime import datetime, timedelta
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlparse

import pyotp
import requests
from playwright.sync_api import Playwright, sync_playwright
from redis import Redis

from app.utils.common.logger import get_logger
from app.utils.constants import UPLINK_ACCESS_TOKEN
from app.utils.credentials.credentials import Credentials
from app.utils.fetch_data import get_required_env_var
from app.utils.redis_utils import init_redis_client
from app.utils.urls import REDIRECT_URI, UPLINK_ACCESS_TOKEN_URL, UPLINK_AUTH_URL

logger = get_logger(Path(__file__).name)


def set_key_with_expiry(key: str, value: str) -> None:
    """
    Store a key-value pair in Redis with an expiry time set to 3:30 PM of the current or next day

    Parameters
    ----------
    key: ``str``
        The key to store in Redis
    value: ``str``
        The value to associate with the key
    """
    client = init_redis_client(False)
    now = datetime.now()
    expiry_time = now.replace(
        hour=15, minute=30, second=0, microsecond=0
    )  # Today at 3:30 PM

    # If current time is past 3:30 PM, set expiry for next day's 3:30 PM
    if now >= expiry_time:
        expiry_time += timedelta(days=1)

    expiry_timestamp = int(expiry_time.timestamp())  # Convert to UNIX timestamp

    # Store key with expiration
    client.set(key, value)
    client.expireat(key, expiry_timestamp)


def get_and_validate_key(key: str) -> str | None:
    """
    Retrieve and validate a key from Redis

    Parameters
    ----------
    key: ``str``
        The key to retrieve from Redis

    Returns
    -------
    value: ``str | None``
        The value associated with the key if it exists and is valid, otherwise None
    """
    client = cast(Redis, init_redis_client(False))
    value = cast(str | None, client.get(key))

    if value is None:
        logger.info("Key '%s' does not exist or has expired.", key)
        return None

    ttl = client.ttl(key)
    if ttl == -2:
        logger.info("Key '%s' has expired.", key)
        return None
    if ttl == -1:
        logger.warning("Warning: Key '%s' exists but has no expiration set!", key)

    return value


def _get_auth_code(
    playwright: Playwright,
    auth_url: str,
    totp_key: str,
    mobile_no: str,
    pin: str,
    redirect_uri: str,
) -> str | None:
    """
    Automate the login process to retrieve an authorization code

    Parameters
    ----------
    playwright: ``Playwright``
        The Playwright instance to use for browser automation
    auth_url: ``str``
        The URL to initiate the authentication process
    totp_key: ``str``
        The TOTP key for generating OTP
    mobile_no: ``str``
        The mobile number to use for login
    pin: ``str``
        The PIN to use for login
    redirect_uri: ``str``
        The redirect URI to capture the authorization code

    Returns
    -------
    auth_code: ``str | None``
        The authorization code if successful, otherwise None
    """
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    # Expect a request containing the redirect URI with the authorization code
    with page.expect_request(f"*{redirect_uri}/?code*") as request:
        page.goto(auth_url)

        # Enter mobile number
        page.locator("#mobileNum").fill(mobile_no)
        page.get_by_role("button", name="Get OTP").click()

        # Enter OTP
        otp = pyotp.TOTP(totp_key).now()
        page.locator("#otpNum").fill(otp)
        page.get_by_role("button", name="Continue").click()

        # Enter PIN
        page.get_by_label("Enter 6-digit PIN").fill(pin)
        page.get_by_role("button", name="Continue").click()

        # Wait for navigation to complete
        page.wait_for_load_state()

    # Extract authorization code from redirect URL
    url = request.value.url
    parsed_url = urlparse(url)
    auth_code_list = parse_qs(parsed_url.query).get("code", [])
    auth_code = auth_code_list[0] if auth_code_list else None

    # Close browser
    context.close()
    browser.close()

    return auth_code


def _get_access_token(
    code: str, api_key: str, secret_key: str, redirect_uri: str
) -> str:
    """
    Exchange an authorization code for an access token

    Parameters
    ----------
    code: ``str``
        The authorization code to exchange
    api_key: ``str``
        The API key for the client
    secret_key: ``str``
        The secret key for the client
    redirect_uri: ``str``
        The redirect URI used in the authorization process

    Returns
    -------
    access_token: ``str``
        The access token if successful

    Raises
    ------
    ``Exception``
        If the access token cannot be retrieved
    """
    headers = {
        "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {
        "code": code,
        "client_id": api_key,
        "client_secret": secret_key,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }

    response = requests.post(
        UPLINK_ACCESS_TOKEN_URL, headers=headers, data=data, timeout=10
    )
    json_response = response.json()
    access_token = json_response.get("access_token", None)

    if not access_token:
        raise ValueError(
            f"Failed to get access token: {json_response.get('error_description', 'Unknown error')}"
        )

    return access_token


class UplinkCredentials(Credentials):
    """
    Credentials class to store the credentials required to authenticate the Uplink connection

    Attributes
    ----------
    access_token: ``str``
        The access token that is generated from the Uplink API
    """

    def __init__(self, access_token: str) -> None:
        """
        Initialize the UplinkCredentials with an access token

        Parameters
        ----------
        access_token: ``str``
            The access token to use for authentication
        """
        self.access_token = access_token

    @classmethod
    def get_credentials(cls) -> "UplinkCredentials":
        """
        Create a Credentials object with the API key

        Returns
        -------
        ``UplinkCredentials``
            The credentials object with the access token
        """
        access_token = get_and_validate_key(UPLINK_ACCESS_TOKEN)

        if access_token:
            logger.info("Using cached Uplink access token.")
            return UplinkCredentials(access_token)

        logger.info("Generating new Uplink access token.")
        api_key = get_required_env_var("UPLINK_API_KEY")
        secret_key = get_required_env_var("UPLINK_SECRET_KEY")
        totp_key = get_required_env_var("UPLINK_TOTP_KEY")
        mobile_no = get_required_env_var("UPLINK_MOBILE_NO")
        pin = get_required_env_var("UPLINK_PIN")

        # Automate the login process using Playwright
        with sync_playwright() as playwright:
            auth_code = _get_auth_code(
                playwright,
                UPLINK_AUTH_URL.format(api_key),
                totp_key,
                mobile_no,
                pin,
                REDIRECT_URI,
            )

            if not auth_code:
                raise ValueError("Failed to get authorization code.")

            access_token = _get_access_token(
                auth_code, api_key, secret_key, REDIRECT_URI
            )
            set_key_with_expiry(UPLINK_ACCESS_TOKEN, access_token)

        return UplinkCredentials(access_token)
