# pylint: disable=protected-access,too-many-lines
"""
Unit tests for UplinkCredentialManager class.

This module contains comprehensive unit tests for the UplinkCredentialManager
class, testing all methods, edge cases, and error scenarios.
"""

from datetime import datetime, timedelta
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
import redis
from omegaconf import DictConfig
from playwright.sync_api import BrowserContext, Page
from pytest import MonkeyPatch
from redis import RedisError
from requests.exceptions import ConnectionError as RequestConnectionError
from requests.exceptions import HTTPError, Timeout

from app.data_layer.data_models.credential_model import (
    UplinkCredentialInput,
    UplinkCredentialOutput,
)
from app.utils.common import init_from_cfg
from app.utils.credentials import CredentialManager, UplinkCredentialManager
from app.utils.urls import REDIRECT_URI, UPLINK_ACCESS_TOKEN_URL

KEY = "test_key"
UPLINK_CRED_INPUT_ENV = {
    "UPLINK_API_KEY": "test_uplink_api_key",
    "UPLINK_SECRET_KEY": "test_uplink_secret_key",
    "UPLINK_TOTP_KEY": "test_uplink_totp_key",
    "UPLINK_MOBILE_NO": "1234567890",
    "UPLINK_PIN": "123456",
}
AUTH_URL = "https://example.com/auth"


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch: MonkeyPatch):
    """
    Auto-use fixture to mock environment variables for all tests.
    """
    for env_name, env_value in UPLINK_CRED_INPUT_ENV.items():
        monkeypatch.setenv(env_name, env_value)


@pytest.fixture
def uplink_credential_manager(
    mock_uplink_credential_input: UplinkCredentialInput,
    mock_uplink_credential_output: UplinkCredentialOutput,
) -> UplinkCredentialManager:
    """
    Fixture to provide a UplinkCredentialManager instance.
    """
    return UplinkCredentialManager(
        mock_uplink_credential_input, mock_uplink_credential_output
    )


@pytest.fixture
def mock_logger(mocker):
    """
    Fixture to mock the logger used in UplinkCredentialManager.
    """
    return mocker.patch("app.utils.credentials.uplink_credential_manager.logger")


@pytest.fixture
def mock_playwright(mocker):
    """
    Fixture to mock Playwright browser automation.
    """
    mock_page = mocker.MagicMock(spec=Page)
    mock_context = mocker.MagicMock(spec=BrowserContext)
    mock_browser = mocker.MagicMock()

    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page

    mock_playwright = mocker.MagicMock()
    mock_playwright.chromium.launch.return_value = mock_browser

    return mock_playwright


@pytest.fixture
def mock_requests_response(mocker):
    """
    Fixture to mock requests response.
    """
    mock_response = mocker.MagicMock()
    mock_response.json.return_value = {"access_token": "test_access_token"}
    return mock_response


class TestUplinkCredentialManagerInheritance:
    """
    Test class for inheritance and registration.
    """

    def test_inheritance_from_credential_manager(self):
        """
        Test that UplinkCredentialManager properly inherits from CredentialManager.
        """
        assert issubclass(UplinkCredentialManager, CredentialManager)
        assert CredentialManager in CredentialManager._registry
        assert (
            "uplink_credential_manager"
            in CredentialManager._registry[CredentialManager]
        )


class TestUplinkCredentialManagerInit:
    """
    Test class for UplinkCredentialManager initialization.
    """

    def validate_instance(
        self,
        manager: UplinkCredentialManager | Any,
        expected_input: UplinkCredentialInput,
        expected_output: UplinkCredentialOutput,
    ):
        """
        Validate the instance attributes of UplinkCredentialManager.

        Parameters
        ----------
        manager: ``UplinkCredentialManager``
            The instance of UplinkCredentialManager to validate.
        expected_input: ``UplinkCredentialInput``
            The expected input credentials.
        expected_output: ``UplinkCredentialOutput``
            The expected output credentials.
        """
        assert isinstance(manager, UplinkCredentialManager)
        assert manager.credential_input == expected_input
        assert manager.credentials == expected_output

    def test_init_with_valid_credentials(
        self,
        mock_uplink_credential_input: UplinkCredentialInput,
        mock_uplink_credential_output: UplinkCredentialOutput,
    ):
        """
        Test successful initialization with valid credentials.
        """
        manager = UplinkCredentialManager(
            mock_uplink_credential_input, mock_uplink_credential_output
        )
        self.validate_instance(
            manager, mock_uplink_credential_input, mock_uplink_credential_output
        )

    def test_init_with_none_credentials(
        self, mock_uplink_credential_input: UplinkCredentialInput
    ):
        """
        Test initialization with None credentials.
        """
        empty_input = UplinkCredentialOutput(access_token="")
        manager = UplinkCredentialManager(mock_uplink_credential_input, empty_input)
        self.validate_instance(manager, mock_uplink_credential_input, empty_input)

    def test_from_cfg(
        self,
        mocker,
        mock_uplink_credential_input: UplinkCredentialInput,
        mock_uplink_credential_output,
    ):
        """
        Test initialization from configuration.
        """
        mock_generate_credentials = mocker.patch.object(
            UplinkCredentialManager,
            "generate_credentials",
            return_value=mock_uplink_credential_output,
        )
        manager = UplinkCredentialManager.from_cfg()
        self.validate_instance(
            manager, mock_uplink_credential_input, mock_uplink_credential_output
        )
        mock_generate_credentials.assert_called_once_with(mock_uplink_credential_input)

    def test_init_from_cfg(
        self,
        mocker,
        mock_uplink_credential_input: UplinkCredentialInput,
        mock_uplink_credential_output,
    ):
        """
        Test initialization from configuration.
        """
        mock_generate_credentials = mocker.patch.object(
            UplinkCredentialManager,
            "generate_credentials",
            return_value=mock_uplink_credential_output,
        )

        manager = init_from_cfg(
            DictConfig({"name": "uplink_credential_manager"}),
            base_class=CredentialManager,
        )
        self.validate_instance(
            manager, mock_uplink_credential_input, mock_uplink_credential_output
        )

        mock_generate_credentials.assert_called_once_with(mock_uplink_credential_input)


class TestGetNextExpiryTime:
    """
    Test class for get_next_expiry_time method.
    """

    @pytest.fixture
    def mock_datetime(self, mocker):
        """
        Fixture to mock datetime for testing.
        """
        return mocker.patch("app.utils.credentials.uplink_credential_manager.datetime")

    def test_get_next_expiry_time_before_3_30pm(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_datetime_before_3_30pm: datetime,
        mock_datetime: MagicMock,
    ):
        """
        Test get_next_expiry_time when current time is before 3:30 PM IST.
        """
        mock_datetime.now.return_value = mock_datetime_before_3_30pm

        result = uplink_credential_manager.get_next_expiry_time()

        # Should return seconds until 3:30 PM same day
        expected_expiry = mock_datetime_before_3_30pm.replace(
            hour=15, minute=30, second=0, microsecond=0
        )
        expected_seconds = int(
            (expected_expiry - mock_datetime_before_3_30pm).total_seconds()
        )

        assert result == expected_seconds

    def test_get_next_expiry_time_after_3_30pm(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_datetime_after_3_30pm: datetime,
        mock_datetime,
    ):
        """
        Test get_next_expiry_time when current time is after 3:30 PM IST.
        """
        mock_datetime.now.return_value = mock_datetime_after_3_30pm

        result = uplink_credential_manager.get_next_expiry_time()

        # Should return seconds until 3:30 PM next day
        expected_expiry = mock_datetime_after_3_30pm.replace(
            hour=15, minute=30, second=0, microsecond=0
        ) + timedelta(days=1)
        expected_seconds = int(
            (expected_expiry - mock_datetime_after_3_30pm).total_seconds()
        )

        assert result == expected_seconds


class TestCheckCache:
    """
    Test class for check_cache method.
    """

    def test_check_cache_with_valid_hash_data(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_redis_client: MagicMock,
        mock_uplink_redis_hash_data: Dict[str, str],
    ):
        """
        Test check_cache with valid hash data in Redis.
        """
        mock_pipeline = mock_redis_client.pipeline.return_value
        mock_pipeline.execute.return_value = [b"hash", mock_uplink_redis_hash_data]

        result = uplink_credential_manager.check_cache(mock_redis_client, KEY)

        assert result is not None
        assert isinstance(result, UplinkCredentialOutput)
        assert result.access_token == "cached_uplink_access_token"

        mock_redis_client.pipeline.assert_called_once()
        mock_pipeline.type.assert_called_once_with(KEY)
        mock_pipeline.hgetall.assert_called_once_with(KEY)
        mock_pipeline.execute.assert_called_once()

    def test_check_cache_with_non_hash_type(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_redis_client: MagicMock,
    ):
        """
        Test check_cache when Redis key type is not hash.
        """
        mock_pipeline = mock_redis_client.pipeline.return_value
        mock_pipeline.execute.return_value = ["string", {}]

        result = uplink_credential_manager.check_cache(mock_redis_client, KEY)

        assert result is None

    def test_check_cache_with_redis_error(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_redis_client: MagicMock,
        mock_logger: MagicMock,
    ):
        """
        Test check_cache when Redis raises an error.
        """
        redis_error = RedisError("Redis connection failed")
        mock_redis_client.pipeline.side_effect = redis_error

        result = uplink_credential_manager.check_cache(mock_redis_client, KEY)

        assert result is None
        mock_logger.error.assert_called_once_with(
            "Redis error while checking cache for key %s: %s", KEY, redis_error
        )


class TestCacheCredentials:
    """
    Test class for cache_credentials method.
    """

    def test_cache_credentials_success(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_redis_client: MagicMock,
        mock_uplink_credential_output: UplinkCredentialOutput,
        mock_logger: MagicMock,
        mocker,
    ):
        """
        Test successful caching of credentials.
        """
        mock_pipeline = mock_redis_client.pipeline.return_value
        mock_ttl = 3600
        mocker.patch.object(
            uplink_credential_manager, "get_next_expiry_time", return_value=mock_ttl
        )

        uplink_credential_manager.cache_credentials(
            mock_redis_client, KEY, mock_uplink_credential_output
        )

        mock_redis_client.pipeline.assert_called_once()
        mock_pipeline.hset.assert_called_once_with(
            KEY, mapping=mock_uplink_credential_output.to_dict()
        )
        mock_pipeline.expire.assert_called_once_with(KEY, mock_ttl)
        mock_pipeline.execute.assert_called_once()

        mock_logger.info.assert_called_once_with(
            "Credentials cached successfully for key: %s for next %d seconds",
            KEY,
            mock_ttl,
        )

    def test_cache_credentials_with_redis_error(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_redis_client: MagicMock,
        mock_uplink_credential_output: UplinkCredentialOutput,
        mock_logger: MagicMock,
    ):
        """
        Test cache_credentials when Redis raises an error.
        """
        redis_error = RedisError("Redis connection failed")
        mock_redis_client.pipeline.side_effect = redis_error

        uplink_credential_manager.cache_credentials(
            mock_redis_client, KEY, mock_uplink_credential_output
        )

        mock_logger.error.assert_called_once_with(
            "Redis error while caching credentials for key %s: %s", KEY, redis_error
        )


class TestGetAuthCode:
    """
    Test class for _get_auth_code method.
    """

    @pytest.fixture(autouse=True)
    def mock_totp(self, mocker, mock_pyotp):
        """
        Fixture to mock pyotp TOTP instance.
        """
        return mocker.patch(
            "app.utils.credentials.uplink_credential_manager.pyotp.TOTP",
            return_value=mock_pyotp,
        )

    def test_get_auth_code_success(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_playwright,
        mock_uplink_credential_input: UplinkCredentialInput,
        mock_totp: MagicMock,
        mocker,
    ):
        """
        Test successful retrieval of authorization code.
        """
        expected_code = "test_auth_code"

        # Mock page interactions
        mock_page = (
            mock_playwright.chromium.launch.return_value.new_context.return_value.new_page.return_value
        )
        mock_request = mocker.MagicMock()
        mock_request.value.url = f"https://127.0.0.1:5055/?code={expected_code}"
        mock_page.expect_request.return_value.__enter__.return_value = mock_request

        result = uplink_credential_manager._get_auth_code(
            mock_playwright, AUTH_URL, mock_uplink_credential_input
        )

        assert result == expected_code

        # Verify browser automation steps
        mock_page.goto.assert_called_once_with(AUTH_URL)
        mock_page.locator.assert_any_call("#mobileNum")
        mock_page.locator.assert_any_call("#otpNum")
        mock_page.get_by_role.assert_any_call("button", name="Get OTP")
        mock_page.get_by_role.assert_any_call("button", name="Continue")
        mock_page.get_by_label.assert_called_with("Enter 6-digit PIN")

        # Verify TOTP generation
        mock_totp.assert_called_once_with(mock_uplink_credential_input.totp_key)

    def test_get_auth_code_no_code_in_url(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_playwright,
        mock_uplink_credential_input: UplinkCredentialInput,
        mocker,
    ):
        """
        Test _get_auth_code when no code is found in redirect URL.
        """
        mock_page = (
            mock_playwright.chromium.launch.return_value.new_context.return_value.new_page.return_value
        )
        mock_request = mocker.MagicMock()
        mock_request.value.url = "https://127.0.0.1:5055/"  # No code parameter
        mock_page.expect_request.return_value.__enter__.return_value = mock_request

        result = uplink_credential_manager._get_auth_code(
            mock_playwright, AUTH_URL, mock_uplink_credential_input
        )

        assert result is None

    def test_get_auth_code_browser_automation_exception(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_playwright,
        mock_uplink_credential_input: UplinkCredentialInput,
        mock_logger: MagicMock,
        mocker,
    ):
        """
        Test _get_auth_code when browser automation fails.
        """
        # Mock browser launch to raise an exception
        mock_browser = mocker.MagicMock()
        mock_context = mocker.MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_playwright.chromium.launch.return_value = mock_browser

        # Mock page creation to raise an exception
        page_error = Exception("Failed to create page")
        mock_context.new_page.side_effect = page_error

        with pytest.raises(Exception, match="Failed to create page"):
            uplink_credential_manager._get_auth_code(
                mock_playwright, AUTH_URL, mock_uplink_credential_input
            )

        # Verify error logging
        mock_logger.error.assert_called_once_with(
            "Error during browser automation: %s", page_error
        )

        # Verify cleanup
        mock_context.close.assert_called_once()
        mock_browser.close.assert_called_once()

    def test_get_auth_code_totp_generation_error(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_playwright,
        mock_uplink_credential_input: UplinkCredentialInput,
        mock_logger: MagicMock,
        mock_totp: MagicMock,
    ):
        """
        Test _get_auth_code when TOTP generation fails.
        """
        # Mock TOTP to raise an exception
        totp_error = Exception("Invalid TOTP key")
        mock_totp.side_effect = totp_error

        with pytest.raises(Exception, match="Invalid TOTP key"):
            uplink_credential_manager._get_auth_code(
                mock_playwright, AUTH_URL, mock_uplink_credential_input
            )

        # Verify error logging
        mock_logger.error.assert_called_once_with(
            "Error during browser automation: %s", totp_error
        )

        # Verify TOTP was called with correct key
        mock_totp.assert_called_once_with(mock_uplink_credential_input.totp_key)

    def test_get_auth_code_page_interaction_error(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_playwright,
        mock_uplink_credential_input: UplinkCredentialInput,
        mock_logger: MagicMock,
    ):
        """
        Test _get_auth_code when page interaction fails.
        """
        # Mock page interactions
        mock_page = (
            mock_playwright.chromium.launch.return_value.new_context.return_value.new_page.return_value
        )

        # Mock element locator to raise an exception
        locator_error = Exception("Element not found")
        mock_page.locator.side_effect = locator_error

        with pytest.raises(Exception, match="Element not found"):
            uplink_credential_manager._get_auth_code(
                mock_playwright, AUTH_URL, mock_uplink_credential_input
            )

        # Verify error logging
        mock_logger.error.assert_called_once_with(
            "Error during browser automation: %s", locator_error
        )

        # Verify page.goto was called before the error
        mock_page.goto.assert_called_once_with(AUTH_URL)

    def test_get_auth_code_page_loading_error(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_playwright,
        mock_uplink_credential_input: UplinkCredentialInput,
        mock_logger: MagicMock,
    ):
        """
        Test _get_auth_code when page fails to load.
        """
        # Mock page interactions
        mock_page = (
            mock_playwright.chromium.launch.return_value.new_context.return_value.new_page.return_value
        )

        # Mock page.goto to raise an exception
        navigation_error = Exception("Page failed to load")
        mock_page.goto.side_effect = navigation_error

        with pytest.raises(Exception, match="Page failed to load"):
            uplink_credential_manager._get_auth_code(
                mock_playwright, AUTH_URL, mock_uplink_credential_input
            )

        # Verify error logging
        mock_logger.error.assert_called_once_with(
            "Error during browser automation: %s", navigation_error
        )

        # Verify page.goto was called with correct URL
        mock_page.goto.assert_called_once_with(AUTH_URL)

    def test_get_auth_code_malformed_redirect_url(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_playwright,
        mock_uplink_credential_input: UplinkCredentialInput,
        mocker,
    ):
        """
        Test _get_auth_code with malformed redirect URL.
        """
        # Mock page interactions
        mock_page = (
            mock_playwright.chromium.launch.return_value.new_context.return_value.new_page.return_value
        )
        mock_request = mocker.MagicMock()
        mock_request.value.url = "malformed-url-without-query"  # No query parameters
        mock_page.expect_request.return_value.__enter__.return_value = mock_request

        result = uplink_credential_manager._get_auth_code(
            mock_playwright, AUTH_URL, mock_uplink_credential_input
        )

        assert result is None

    def test_get_auth_code_empty_code_parameter(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_playwright,
        mock_uplink_credential_input: UplinkCredentialInput,
        mocker,
    ):
        """
        Test _get_auth_code when code parameter exists but is empty.
        """
        mock_page = (
            mock_playwright.chromium.launch.return_value.new_context.return_value.new_page.return_value
        )
        mock_request = mocker.MagicMock()
        mock_request.value.url = "https://127.0.0.1:5055/?code="  # Empty code parameter
        mock_page.expect_request.return_value.__enter__.return_value = mock_request

        result = uplink_credential_manager._get_auth_code(
            mock_playwright, AUTH_URL, mock_uplink_credential_input
        )

        assert result is None  # parse_qs ignores blank values by default

    def test_get_auth_code_multiple_code_parameters(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_playwright,
        mock_uplink_credential_input: UplinkCredentialInput,
        mocker,
    ):
        """
        Test _get_auth_code when URL contains multiple code parameters.
        """
        expected_code = "first_code"

        # Mock page interactions
        mock_page = (
            mock_playwright.chromium.launch.return_value.new_context.return_value.new_page.return_value
        )
        mock_request = mocker.MagicMock()
        mock_request.value.url = (
            f"https://127.0.0.1:5055/?code={expected_code}&code=second_code"
        )
        mock_page.expect_request.return_value.__enter__.return_value = mock_request

        result = uplink_credential_manager._get_auth_code(
            mock_playwright, AUTH_URL, mock_uplink_credential_input
        )

        assert result == expected_code  # Should return first code

    def test_get_auth_code_browser_cleanup_on_success(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_playwright,
        mock_uplink_credential_input: UplinkCredentialInput,
        mocker,
    ):
        """
        Test _get_auth_code ensures proper browser cleanup on success.
        """
        expected_code = "test_auth_code"

        # Mock browser and context
        mock_browser = mock_playwright.chromium.launch.return_value
        mock_context = mock_browser.new_context.return_value
        mock_page = mock_context.new_page.return_value

        mock_request = mocker.MagicMock()
        mock_request.value.url = f"https://127.0.0.1:5055/?code={expected_code}"
        mock_page.expect_request.return_value.__enter__.return_value = mock_request

        result = uplink_credential_manager._get_auth_code(
            mock_playwright, AUTH_URL, mock_uplink_credential_input
        )

        assert result == expected_code

        # Verify browser cleanup
        mock_context.close.assert_called_once()
        mock_browser.close.assert_called_once()

    def test_get_auth_code_browser_cleanup_on_exception(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_playwright,
        mock_uplink_credential_input: UplinkCredentialInput,
        mock_logger: MagicMock,
    ):
        """
        Test _get_auth_code ensures proper browser cleanup even on exception.
        """
        # Mock browser and context
        mock_browser = mock_playwright.chromium.launch.return_value
        mock_context = mock_browser.new_context.return_value
        mock_page = mock_context.new_page.return_value

        # Mock an exception during page interaction
        page_error = Exception("Page interaction failed")
        mock_page.goto.side_effect = page_error

        with pytest.raises(Exception, match="Page interaction failed"):
            uplink_credential_manager._get_auth_code(
                mock_playwright, AUTH_URL, mock_uplink_credential_input
            )

        # Verify browser cleanup happened despite exception
        mock_context.close.assert_called_once()
        mock_browser.close.assert_called_once()

        # Verify error was logged
        mock_logger.error.assert_called_once_with(
            "Error during browser automation: %s", page_error
        )

    def test_get_auth_code_request_expect_timeout(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_playwright,
        mock_uplink_credential_input: UplinkCredentialInput,
        mock_logger: MagicMock,
    ):
        """
        Test _get_auth_code when request expectation times out.
        """
        # Mock page interactions
        mock_page = (
            mock_playwright.chromium.launch.return_value.new_context.return_value.new_page.return_value
        )

        # Mock expect_request to raise a timeout exception
        timeout_error = Exception("Request expectation timed out")
        mock_page.expect_request.side_effect = timeout_error

        with pytest.raises(Exception, match="Request expectation timed out"):
            uplink_credential_manager._get_auth_code(
                mock_playwright, AUTH_URL, mock_uplink_credential_input
            )

        # Verify error logging
        mock_logger.error.assert_called_once_with(
            "Error during browser automation: %s", timeout_error
        )


class TestGetAccessToken:
    """
    Test class for get_access_token method.
    """

    @pytest.fixture
    def headers_payload(self, mock_uplink_credential_input: UplinkCredentialInput):
        """
        Fixture to provide headers and payload for access token request.
        """
        return (
            {
                "accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            {
                "code": "test_auth_code",
                "client_id": mock_uplink_credential_input.api_key,
                "client_secret": mock_uplink_credential_input.secret_key,
                "redirect_uri": REDIRECT_URI,
                "grant_type": "authorization_code",
            },
        )

    def test_get_access_token_success(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_uplink_credential_input: UplinkCredentialInput,
        mock_requests_response: MagicMock,
        headers_payload: tuple[dict[str, str], dict[str, str]],
        mocker,
    ):
        """
        Test successful retrieval of access token.
        """
        code = "test_auth_code"
        expected_token = "test_access_token"
        headers, data = headers_payload

        mock_post = mocker.patch(
            "app.utils.credentials.uplink_credential_manager.requests.post",
            return_value=mock_requests_response,
        )

        result = uplink_credential_manager.get_access_token(
            mock_uplink_credential_input, code
        )

        assert result == expected_token

        # Verify request parameters
        call_args = mock_post.call_args
        assert call_args[1]["data"]["code"] == code
        assert call_args[1]["data"]["client_id"] == mock_uplink_credential_input.api_key
        assert (
            call_args[1]["data"]["client_secret"]
            == mock_uplink_credential_input.secret_key
        )

        mock_post.assert_called_once_with(
            UPLINK_ACCESS_TOKEN_URL,
            headers=headers,
            data=data,
            timeout=10,
        )

    def test_get_access_token_failure(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_uplink_credential_input: UplinkCredentialInput,
        headers_payload: tuple[dict[str, str], dict[str, str]],
        mocker,
    ):
        """
        Test get_access_token when API returns an error.
        """
        code = "test_auth_code"
        error_description = "Invalid authorization code"

        headers, data = headers_payload

        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {
            "error": "invalid_grant",
            "error_description": error_description,
        }

        mock_post = mocker.patch(
            "app.utils.credentials.uplink_credential_manager.requests.post",
            return_value=mock_response,
        )

        with pytest.raises(
            ValueError, match=f"Failed to get access token: {error_description}"
        ):
            uplink_credential_manager.get_access_token(
                mock_uplink_credential_input, code
            )

        mock_post.assert_called_once_with(
            UPLINK_ACCESS_TOKEN_URL,
            headers=headers,
            data=data,
            timeout=10,
        )

    def test_get_access_token_http_error(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_uplink_credential_input: UplinkCredentialInput,
        mock_logger: MagicMock,
        mocker,
    ):
        """
        Test get_access_token when HTTP request returns an error status.
        """
        code = "test_auth_code"

        mock_response = mocker.MagicMock()
        mock_response.raise_for_status.side_effect = HTTPError("401 Unauthorized")

        mock_post = mocker.patch(
            "app.utils.credentials.uplink_credential_manager.requests.post",
            return_value=mock_response,
        )

        with pytest.raises(HTTPError, match="401 Unauthorized"):
            uplink_credential_manager.get_access_token(
                mock_uplink_credential_input, code
            )

        mock_logger.error.assert_called_once_with(
            "Failed to request access token: %s",
            mock_response.raise_for_status.side_effect,
        )
        mock_post.assert_called_once()

    def test_get_access_token_connection_error(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_uplink_credential_input: UplinkCredentialInput,
        mock_logger: MagicMock,
        mocker,
    ):
        """
        Test get_access_token when network connection fails.
        """
        code = "test_auth_code"

        connection_error = RequestConnectionError("Connection failed")
        mock_post = mocker.patch(
            "app.utils.credentials.uplink_credential_manager.requests.post",
            side_effect=connection_error,
        )

        with pytest.raises(RequestConnectionError, match="Connection failed"):
            uplink_credential_manager.get_access_token(
                mock_uplink_credential_input, code
            )

        mock_logger.error.assert_called_once_with(
            "Failed to request access token: %s", connection_error
        )

        mock_post.assert_called_once()

    def test_get_access_token_timeout_error(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_uplink_credential_input: UplinkCredentialInput,
        mock_logger: MagicMock,
        mocker,
    ):
        """
        Test get_access_token when request times out.
        """
        code = "test_auth_code"

        timeout_error = Timeout("Request timed out")
        mock_post = mocker.patch(
            "app.utils.credentials.uplink_credential_manager.requests.post",
            side_effect=timeout_error,
        )

        with pytest.raises(Timeout, match="Request timed out"):
            uplink_credential_manager.get_access_token(
                mock_uplink_credential_input, code
            )

        mock_logger.error.assert_called_once_with(
            "Failed to request access token: %s", timeout_error
        )
        mock_post.assert_called_once()

    def test_get_access_token_invalid_json_response(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_uplink_credential_input: UplinkCredentialInput,
        mocker,
    ):
        """
        Test get_access_token when API returns invalid JSON.
        """
        code = "test_auth_code"

        mock_response = mocker.MagicMock()
        mock_response.json.side_effect = ValueError("Invalid JSON")

        mock_post = mocker.patch(
            "app.utils.credentials.uplink_credential_manager.requests.post",
            return_value=mock_response,
        )

        with pytest.raises(ValueError, match="Invalid JSON"):
            uplink_credential_manager.get_access_token(
                mock_uplink_credential_input, code
            )
        mock_post.assert_called_once()

        # Note: logger.error won't be called in this case because ValueError from json()
        # is not caught by the requests.exceptions.RequestException handler

    def test_get_access_token_empty_access_token(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_uplink_credential_input: UplinkCredentialInput,
        mocker,
    ):
        """
        Test get_access_token when API returns empty access token.
        """
        code = "test_auth_code"

        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {"access_token": ""}

        mock_post = mocker.patch(
            "app.utils.credentials.uplink_credential_manager.requests.post",
            return_value=mock_response,
        )

        with pytest.raises(
            ValueError, match="Failed to get access token: Unknown error"
        ):
            uplink_credential_manager.get_access_token(
                mock_uplink_credential_input, code
            )
        mock_post.assert_called_once()

    def test_get_access_token_no_access_token_field(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_uplink_credential_input: UplinkCredentialInput,
        mocker,
    ):
        """
        Test get_access_token when API response has no access_token field.
        """
        code = "test_auth_code"

        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {"some_other_field": "value"}

        mock_post = mocker.patch(
            "app.utils.credentials.uplink_credential_manager.requests.post",
            return_value=mock_response,
        )

        with pytest.raises(
            ValueError, match="Failed to get access token: Unknown error"
        ):
            uplink_credential_manager.get_access_token(
                mock_uplink_credential_input, code
            )
        mock_post.assert_called_once()


class TestGenerateNewCredentials:
    """
    Test class for _generate_new_credentials method.
    """

    def test_generate_new_credentials_success(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_uplink_credential_input: UplinkCredentialInput,
        mocker,
    ):
        """
        Test successful generation of new credentials.
        """
        expected_code = "test_auth_code"
        expected_token = "test_access_token"

        mock_sync_playwright = mocker.patch(
            "app.utils.credentials.uplink_credential_manager.sync_playwright"
        )
        mock_playwright_context = mocker.MagicMock()
        mock_sync_playwright.return_value.__enter__.return_value = (
            mock_playwright_context
        )

        mocker.patch.object(
            uplink_credential_manager, "_get_auth_code", return_value=expected_code
        )
        mocker.patch.object(
            uplink_credential_manager, "get_access_token", return_value=expected_token
        )

        result = uplink_credential_manager._generate_new_credentials(
            mock_uplink_credential_input
        )

        assert isinstance(result, UplinkCredentialOutput)
        assert result.access_token == expected_token

    def test_generate_new_credentials_auth_code_failure(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_uplink_credential_input: UplinkCredentialInput,
        mocker,
    ):
        """
        Test _generate_new_credentials when auth code retrieval fails.
        """
        mock_sync_playwright = mocker.patch(
            "app.utils.credentials.uplink_credential_manager.sync_playwright"
        )
        mock_playwright_context = mocker.MagicMock()
        mock_sync_playwright.return_value.__enter__.return_value = (
            mock_playwright_context
        )

        mocker.patch.object(
            uplink_credential_manager, "_get_auth_code", return_value=None
        )

        with pytest.raises(ValueError, match="Failed to get authorization code."):
            uplink_credential_manager._generate_new_credentials(
                mock_uplink_credential_input
            )


class TestGenerateCredentials:
    """
    Test class for generate_credentials class method.
    """

    @pytest.fixture
    def mock_redis_connection(self, mocker, mock_redis_client):
        """
        Fixture to mock RedisSyncConnection class.
        """
        mock_redis_connection = mocker.patch(
            "app.utils.credentials.uplink_credential_manager.RedisSyncConnection"
        )
        mock_redis_connection.return_value = mock_redis_connection
        mock_redis_connection.get_connection.return_value = mock_redis_client
        return mock_redis_connection

    def test_generate_credentials_with_cached_credentials(
        self,
        mock_uplink_credential_input: UplinkCredentialInput,
        mock_uplink_credential_output: UplinkCredentialOutput,
        mock_redis_connection,
        mocker,
        mock_logger: MagicMock,
    ):
        """
        Test generate_credentials when credentials are cached in Redis.
        """
        mocker.patch.object(
            UplinkCredentialManager,
            "check_cache",
            return_value=mock_uplink_credential_output,
        )

        result = UplinkCredentialManager.generate_credentials(
            mock_uplink_credential_input
        )

        assert result == mock_uplink_credential_output
        mock_logger.info.assert_any_call("Using cached Uplink access token.")
        mock_redis_connection.close_connection.assert_called_once()

    def test_generate_credentials_without_cached_credentials(
        self,
        mock_uplink_credential_input: UplinkCredentialInput,
        mock_uplink_credential_output: UplinkCredentialOutput,
        mock_redis_connection: MagicMock,
        mocker,
        mock_logger: MagicMock,
    ):
        """
        Test generate_credentials when no cached credentials exist.
        """
        mock_check_cache = mocker.patch.object(
            UplinkCredentialManager, "check_cache", return_value=None
        )
        mock_generate_new_credentials = mocker.patch.object(
            UplinkCredentialManager,
            "_generate_new_credentials",
            return_value=mock_uplink_credential_output,
        )

        result = UplinkCredentialManager.generate_credentials(
            mock_uplink_credential_input
        )

        assert result == mock_uplink_credential_output
        mock_logger.info.assert_any_call("Generating new Uplink access token.")
        mock_redis_connection.close_connection.assert_called_once()

        mock_check_cache.assert_called_once_with(
            mock_redis_connection.get_connection(),
            "uplink_credentials:test_uplink_api_key",
        )
        mock_generate_new_credentials.assert_called_once_with(
            mock_uplink_credential_input
        )


@pytest.mark.parametrize(
    "redis_error,expected_result",
    [
        (RedisError("Connection timeout"), None),
        (RedisError("Network error"), None),
        (redis.ConnectionError("Redis server down"), None),
    ],
)
class TestRedisErrorHandling:
    """
    Test class for Redis error handling scenarios.
    """

    def test_check_cache_various_redis_errors(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_redis_client: MagicMock,
        mock_logger: MagicMock,
        redis_error: Exception,
        expected_result: Any,
    ):
        """
        Test check_cache method with various Redis errors.
        """
        mock_pipeline = mock_redis_client.pipeline.return_value
        mock_pipeline.execute.side_effect = redis_error

        result = uplink_credential_manager.check_cache(mock_redis_client, KEY)

        assert result == expected_result
        mock_logger.error.assert_called_once_with(
            "Redis error while checking cache for key %s: %s", KEY, redis_error
        )

    def test_cache_credentials_various_redis_errors(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_redis_client: MagicMock,
        mock_uplink_credential_output: UplinkCredentialOutput,
        mock_logger: MagicMock,
        redis_error: Exception,
        expected_result: Any,
    ):
        """
        Test cache_credentials method with various Redis errors.
        """
        mock_pipeline = mock_redis_client.pipeline.return_value
        mock_pipeline.execute.side_effect = redis_error

        # This should not raise an exception, just log the error
        uplink_credential_manager.cache_credentials(
            mock_redis_client, KEY, mock_uplink_credential_output
        )
        mock_logger.error.assert_called_once_with(
            "Redis error while caching credentials for key %s: %s", KEY, redis_error
        )
        assert expected_result is None


@pytest.mark.parametrize(
    "access_token_response,should_raise_error",
    [
        ({"access_token": "valid_token"}, False),
        ({"error": "invalid_grant", "error_description": "Invalid code"}, True),
        ({"error": "invalid_client"}, True),
        ({}, True),  # No access_token field
    ],
)
class TestAccessTokenVariations:
    """Test class for access token retrieval with various API responses."""

    def test_get_access_token_response_variations(
        self,
        uplink_credential_manager: UplinkCredentialManager,
        mock_uplink_credential_input: UplinkCredentialInput,
        access_token_response: Dict[str, str],
        should_raise_error: bool,
        mocker,
    ):
        """
        Test get_access_token with various API response formats.
        """
        code = "test_auth_code"

        mock_response = mocker.MagicMock()
        mock_response.json.return_value = access_token_response

        mocker.patch(
            "app.utils.credentials.uplink_credential_manager.requests.post",
            return_value=mock_response,
        )

        if should_raise_error:
            with pytest.raises(ValueError, match="Failed to get access token:"):
                uplink_credential_manager.get_access_token(
                    mock_uplink_credential_input, code
                )
        else:
            result = uplink_credential_manager.get_access_token(
                mock_uplink_credential_input, code
            )
            assert result == access_token_response["access_token"]
