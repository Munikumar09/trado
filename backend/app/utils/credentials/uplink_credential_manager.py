import json
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import pyotp
import requests
from playwright.sync_api import Playwright, sync_playwright
from redis import Redis, RedisError

from app.core.config import UplinkSettings, settings
from app.core.mixins import FactoryMixin
from app.data_layer.data_models.credential_model import (
    UplinkCredentialInput,
    UplinkCredentialOutput,
)
from app.utils.common.logger import get_logger
from app.utils.credentials.base_credential_manager import CredentialManager
from app.utils.redis_utils import RedisSyncConnection
from app.utils.urls import REDIRECT_URI, UPLINK_ACCESS_TOKEN_URL, UPLINK_AUTH_URL

logger = get_logger(Path(__file__).name)


@CredentialManager.register("uplink_credential_manager")
class UplinkCredentialManager(
    CredentialManager[UplinkCredentialInput, UplinkCredentialOutput],
    FactoryMixin[UplinkSettings],
):
    """
    Credentials class to store the credentials required to authenticate the Uplink connection

    Attributes
    ----------
    credential_input: ``UplinkCredentialInput``
        The input credentials for the Uplink connection
    credentials: ``UplinkCredentialOutput``
        The output credentials for the Uplink connection
    """

    max_connections = 1

    def __init__(
        self,
        credential_input: UplinkCredentialInput,
        credential: UplinkCredentialOutput,
    ) -> None:
        self.credential_input = credential_input
        self.credentials = credential

    def get_next_expiry_time(self) -> int:
        """
        The credentials expire at 3:30 PM IST every day. It calculates the next expiry time
        and returns the remaining time until expiry in seconds.

        Returns
        -------
        ``int``
            The remaining time until expiry in seconds
        """
        ist = ZoneInfo("Asia/Kolkata")
        now = datetime.now(ist)
        next_expiry = now.replace(hour=15, minute=30, second=0, microsecond=0)

        if now > next_expiry:
            next_expiry += timedelta(days=1)

        return int((next_expiry - now).total_seconds())

    def check_cache(
        self, redis_client: Redis, key: str
    ) -> UplinkCredentialOutput | None:
        """
        Check if the access token is cached in Redis
        Parameters
        ----------
        redis_client: ``Redis``
            The Redis client to use for checking the cache
        key: ``str``
            The key to check in Redis
        Returns
        -------
        ``UplinkCredentialOutput | None``
            The cached credentials if they exist, otherwise None
        """
        try:
            pipe = redis_client.pipeline()
            pipe.type(key)
            pipe.hgetall(key)

            results = pipe.execute()
            key_type, value = results

            if key_type == "hash" and value:
                return UplinkCredentialOutput(**value)

            return None
        except RedisError as e:
            logger.error("Redis error while checking cache for key %s: %s", key, e)
            return None

    def cache_credentials(
        self, redis_client: Redis, key: str, credentials: UplinkCredentialOutput
    ) -> None:
        """
        Cache the Uplink credentials in Redis
        Parameters:
        -----------
        redis_client: ``Redis``
            The Redis client used to interact with redis to store the credentials
        key: ``str``
            The key to store the credentials in the cache
        credentials: ``UplinkCredentialOutput``
            The credentials to store in the cache
        """
        try:
            pipe = redis_client.pipeline()
            pipe.hset(key, mapping=credentials.to_dict())
            ttl = self.get_next_expiry_time()
            pipe.expire(key, ttl)
            pipe.execute()
            logger.info(
                "Credentials cached successfully for key: %s for next %d seconds",
                key,
                ttl,
            )
        except RedisError as e:
            logger.error("Redis error while caching credentials for key %s: %s", key, e)

    def _get_auth_code(
        self,
        playwright: Playwright,
        auth_url: str,
        credential_input: UplinkCredentialInput,
    ) -> str | None:
        """
        Automate the login process to retrieve an authorization code

        Parameters
        ----------
        playwright: ``Playwright``
            The Playwright instance to use for browser automation
        auth_url: ``str``
            The URL to initiate the authentication process
        credential_input: ``UplinkCredentialInput``
            The input data containing API key, secret key, TOTP key, mobile number, and PIN

        Returns
        -------
        auth_code: ``str | None``
            The authorization code if successful, otherwise None
        """
        browser = playwright.chromium.launch(headless=True)
        try:
            context = browser.new_context()
            page = context.new_page()

            # Expect a request containing the redirect URI with the authorization code
            with page.expect_request(f"*{REDIRECT_URI}/?code*") as request:
                page.goto(auth_url)

                # Enter mobile number
                page.locator("#mobileNum").fill(credential_input.mobile_no)
                page.get_by_role("button", name="Get OTP").click()

                # Enter OTP
                otp = pyotp.TOTP(credential_input.totp_key).now()
                page.locator("#otpNum").fill(otp)
                page.get_by_role("button", name="Continue").click()

                # Enter PIN
                page.get_by_label("Enter 6-digit PIN").fill(credential_input.pin)
                page.get_by_role("button", name="Continue").click()

                # Wait for navigation to complete
                page.wait_for_load_state()

            # Extract authorization code from redirect URL
            url = request.value.url
            parsed_url = urlparse(url)
            auth_code_list = parse_qs(parsed_url.query).get("code", [])
            auth_code = auth_code_list[0] if auth_code_list else None

            return auth_code
        except Exception as e:
            logger.error("Error during browser automation: %s", e)
            raise
        finally:
            # Ensure browser resources are always cleaned up
            context.close()
            browser.close()

    def get_access_token(
        self, credential_input: UplinkCredentialInput, code: str
    ) -> str:
        """
        Create an access token using the authorization code by making a POST request to the Uplink API
        Parameters
        ----------
        credential_input: ``UplinkCredentialInput``
            The input data containing API key, secret key, TOTP key, mobile number, and PIN
        code: ``str``
            The authorization code received after successful login
        Returns
        -------
        access_token: ``str``
            The access token received from the Uplink API
        """
        headers = {
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "code": code,
            "client_id": credential_input.api_key,
            "client_secret": credential_input.secret_key,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code",
        }
        try:
            response = requests.post(
                UPLINK_ACCESS_TOKEN_URL, headers=headers, data=data, timeout=10
            )
            response.raise_for_status()
            json_response = response.json()
            access_token = json_response.get("access_token", None)
            if not access_token:
                raise ValueError(
                    f"Failed to get access token: {json_response.get('error_description', 'Unknown error')}"
                )
            return access_token
        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON response: %s", e)
            raise
        except requests.exceptions.RequestException as e:
            logger.error("Failed to request access token: %s", e)
            raise

    def _generate_new_credentials(
        self, credential_input: UplinkCredentialInput
    ) -> UplinkCredentialOutput:
        """
        Generate new credentials from the input data

        Parameters
        ----------
        credential_input: ``UplinkCredentialInput``
            The input data containing API key, secret key, TOTP key, mobile number, and PIN

        Returns
        -------
        ``UplinkCredentialOutput``
            The output data containing the access token
        """
        # Automate the login process using Playwright
        with sync_playwright() as playwright:
            auth_code = self._get_auth_code(
                playwright,
                UPLINK_AUTH_URL.format(credential_input.api_key),
                credential_input,
            )

            if not auth_code:
                raise ValueError("Failed to get authorization code.")

            access_token = self.get_access_token(credential_input, auth_code)

            return UplinkCredentialOutput(access_token=access_token)

    @classmethod
    def generate_credentials(
        cls, credential_input: UplinkCredentialInput
    ) -> UplinkCredentialOutput:
        """
        Generate credentials from the input data

        Parameters
        ----------
        credential_input: ``UplinkCredentialInput``
            The input data containing API key, secret key, TOTP key, mobile number, and PIN

        Returns
        -------
        ``UplinkCredentialOutput``
            The output data containing the access token
        """
        try:
            redis_connection = RedisSyncConnection.build(settings.redis_config)
            redis_client = redis_connection.get_connection()
            key = f"uplink_credentials:{credential_input.api_key}_{credential_input.connection_num}"

            temp_manager = cls(
                credential_input, UplinkCredentialOutput(access_token="")
            )
            credentials = temp_manager.check_cache(redis_client, key)

            if credentials:
                logger.info("Using cached Uplink access token.")
                return credentials

            logger.info("Generating new Uplink access token.")
            credentials = temp_manager._generate_new_credentials(credential_input)
            temp_manager.cache_credentials(redis_client, key, credentials)

            return credentials
        finally:
            redis_connection.close_connection()

    @classmethod
    def build(cls, settings: UplinkSettings) -> "UplinkCredentialManager":
        """
        Build a UplinkCredentialManager instance from the provided settings.

        Parameters
        ----------
        settings: ``BaseSettings``
            The settings object containing the necessary configuration

        Returns
        -------
        ``UplinkCredentialManager``
            An instance of UplinkCredentialManager
        """
        connection_num = settings.connection_num

        if connection_num < 0 or connection_num >= cls.max_connections:
            raise ValueError(
                f"connection_num must be between 0 and {cls.max_connections} but got {connection_num}"
            )

        credential_input = UplinkCredentialInput(
            api_key=settings.api_key,
            secret_key=settings.secret_key,
            totp_key=settings.totp_key,
            mobile_no=settings.mobile_no,
            pin=settings.pin,
            connection_num=connection_num,
        )
        credentials = cls.generate_credentials(credential_input)

        return cls(credential_input, credentials)
