# pylint: disable=protected-access
"""
Unit tests for SmartapiCredentialManager class.

This module contains comprehensive unit tests for the SmartapiCredentialManager
class, testing all methods, edge cases, and error scenarios.
"""

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
import redis
from omegaconf import DictConfig
from pytest_mock import MockerFixture

from app.data_layer.data_models.credential_model import (
    SmartAPICredentialInput,
    SmartAPICredentialOutput,
)
from app.utils.common import init_from_cfg
from app.utils.credentials import CredentialManager, SmartapiCredentialManager

SMARTAPI_CRED_INPUT_ENV = {
    "SMARTAPI_API_KEY": "test_api_key",
    "SMARTAPI_CLIENT_ID": "test_client_id",
    "SMARTAPI_PWD": "test_password",
    "SMARTAPI_TOKEN": "test_token",
}


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """
    Mock environment variables for testing.
    """
    for env_name, env_value in SMARTAPI_CRED_INPUT_ENV.items():
        monkeypatch.setenv(env_name, env_value)


@pytest.fixture
def smartapi_credential_manager(
    mock_smartapi_credential_output, mock_smartapi_credential_input
):
    """
    Fixture for SmartapiCredentialManager.
    """
    return SmartapiCredentialManager(
        mock_smartapi_credential_input, mock_smartapi_credential_output
    )


@pytest.fixture(autouse=True)
def reset_smartapi_credential_manager():
    """
    Reset the SmartapiCredentialManager class attributes before each test.
    """
    SmartapiCredentialManager.total_connections = 3
    SmartapiCredentialManager.current_connection = 0


@pytest.fixture
def mock_logger(mocker: MockerFixture):
    """
    Fixture to mock the logger used in SmartapiCredentialManager.
    """
    return mocker.patch("app.utils.credentials.smartapi_credential_manager.logger")


class TestSmartapiCredentialManagerClassAttributes:
    """
    Test class for class attributes and registration.
    """

    def test_class_attributes(self):
        """
        Test that class attributes have expected values.
        """
        assert SmartapiCredentialManager.total_connections == 3
        assert SmartapiCredentialManager.current_connection == 0

    def test_inheritance_and_registration(self):
        """
        Test that SmartapiCredentialManager properly inherits from CredentialManager.
        """
        assert issubclass(SmartapiCredentialManager, CredentialManager)

        # Test that the decorator properly registered the class
        assert CredentialManager in CredentialManager._registry
        assert (
            "smartapi_credential_manager"
            in CredentialManager._registry[CredentialManager]
        )


class TestSmartapiCredentialManagerInit:
    """
    Test class for SmartapiCredentialManager initialization.
    """

    @pytest.fixture(autouse=True)
    def mock_generate_credentials(
        self, mocker, mock_smartapi_credential_output: SmartAPICredentialOutput
    ):
        """
        Mock the generate_credentials method to return predefined output.
        """
        mocker.patch(
            "app.utils.credentials.smartapi_credential_manager.SmartapiCredentialManager.generate_credentials",
            return_value=mock_smartapi_credential_output,
        )

    def validate_instance(
        self,
        manager: SmartapiCredentialManager | Any,
        expected_input: SmartAPICredentialInput,
        expected_output: SmartAPICredentialOutput,
        current_connection: int = 0,
    ):
        """
        Validate the instance attributes of SmartapiCredentialManager.

        Parameters:
        -----------
        manager: ``SmartapiCredentialManager``
            The instance to validate.
        expected_input: ``SmartAPICredentialInput``
            Expected input credentials.
        expected_output: ``SmartAPICredentialOutput``
            Expected output credentials.
        """
        assert isinstance(manager, SmartapiCredentialManager)
        assert manager.credential_input == expected_input
        assert manager.credentials == expected_output
        assert manager.total_connections == 3
        assert manager.current_connection == current_connection

    def test_init_with_valid_credentials(
        self,
        mock_smartapi_credential_input: SmartAPICredentialInput,
        mock_smartapi_credential_output: SmartAPICredentialOutput,
    ):
        """
        Test SmartapiCredentialManager initialization with valid credentials.
        """
        manager = SmartapiCredentialManager(
            mock_smartapi_credential_input, mock_smartapi_credential_output
        )
        self.validate_instance(
            manager, mock_smartapi_credential_input, mock_smartapi_credential_output
        )

    def test_init_from_cfg_with_default_index(
        self,
        mock_smartapi_credential_output: SmartAPICredentialOutput,
        mock_smartapi_credential_input: SmartAPICredentialInput,
    ):
        """
        Test SmartapiCredentialManager initialization from configuration with default index.
        """
        manager = SmartapiCredentialManager.from_cfg()
        assert isinstance(manager, SmartapiCredentialManager)
        self.validate_instance(
            manager, mock_smartapi_credential_input, mock_smartapi_credential_output, 1
        )

        # Check if the current connection is incremented
        manager = SmartapiCredentialManager.from_cfg()
        assert isinstance(manager, SmartapiCredentialManager)
        mock_smartapi_credential_input.connection_num += 1
        self.validate_instance(
            manager, mock_smartapi_credential_input, mock_smartapi_credential_output, 2
        )

    def test_init_from_cfg_with_zero_index(
        self,
        mock_smartapi_credential_output: SmartAPICredentialOutput,
        mock_smartapi_credential_input: SmartAPICredentialInput,
    ):
        """
        Test SmartapiCredentialManager initialization from configuration with zero index.
        """
        manager = SmartapiCredentialManager.from_cfg(DictConfig({"connection_num": 0}))
        assert isinstance(manager, SmartapiCredentialManager)
        self.validate_instance(
            manager, mock_smartapi_credential_input, mock_smartapi_credential_output, 1
        )

    def test_init_from_cfg_with_higher_index(
        self,
        mock_smartapi_credential_output: SmartAPICredentialOutput,
        mock_smartapi_credential_input: SmartAPICredentialInput,
    ):
        """
        Test SmartapiCredentialManager initialization from configuration with a higher index.
        """
        mock_smartapi_credential_input.connection_num = 2
        manager = SmartapiCredentialManager.from_cfg(DictConfig({"connection_num": 5}))
        assert isinstance(manager, SmartapiCredentialManager)
        self.validate_instance(
            manager, mock_smartapi_credential_input, mock_smartapi_credential_output, 0
        )

    def test_init_from_cfg(
        self,
        mock_smartapi_credential_output: SmartAPICredentialOutput,
        mock_smartapi_credential_input: SmartAPICredentialInput,
    ):
        """
        Test SmartapiCredentialManager initialization from init_from_cfg with no index specified.
        """
        manager = init_from_cfg(
            DictConfig({"name": "smartapi_credential_manager", "connection_num": 0}),
            base_class=CredentialManager,
        )
        assert isinstance(manager, SmartapiCredentialManager)
        self.validate_instance(
            manager, mock_smartapi_credential_input, mock_smartapi_credential_output, 1
        )


class TestSmartapiCredentialManagerCheckCache:
    """
    Test class for check_cache method.
    """

    def test_check_cache_with_valid_hash_data(
        self,
        mock_redis_client: MagicMock,
        smartapi_credential_manager: SmartapiCredentialManager,
        mock_redis_hash_data: dict[str, str],
    ):
        """
        Test check_cache method with valid hash data in Redis.
        """
        # Setup mock pipeline
        mock_pipeline = mock_redis_client.pipeline.return_value
        mock_pipeline.execute.return_value = ["hash", mock_redis_hash_data]

        result = smartapi_credential_manager.check_cache(mock_redis_client, "test_key")

        assert result is not None
        assert isinstance(result, SmartAPICredentialOutput)
        assert all(
            getattr(result, key) == val for key, val in mock_redis_hash_data.items()
        )

        # Verify Redis calls
        mock_redis_client.pipeline.assert_called_once()
        mock_pipeline.type.assert_called_once_with("test_key")
        mock_pipeline.hgetall.assert_called_once_with("test_key")
        mock_pipeline.execute.assert_called_once()

    def test_check_cache_with_non_hash_type(
        self,
        smartapi_credential_manager: SmartapiCredentialManager,
        mock_redis_client: MagicMock,
    ):
        """
        Test check_cache method when Redis key is not of hash type.
        """
        # Setup mock pipeline to return non-hash type
        mock_pipeline = mock_redis_client.pipeline.return_value
        mock_pipeline.execute.return_value = ["string", {}]
        result = smartapi_credential_manager.check_cache(mock_redis_client, "test_key")

        assert result is None

    def test_check_cache_with_redis_error(
        self,
        smartapi_credential_manager: SmartapiCredentialManager,
        mock_redis_client: MagicMock,
        mock_logger: MagicMock,
    ):
        """
        Test check_cache method when Redis raises an error.
        """
        # Setup mock pipeline to raise Redis error
        mock_pipeline = mock_redis_client.pipeline.return_value
        redis_error = redis.RedisError("Connection failed")
        mock_pipeline.execute.side_effect = redis_error

        result = smartapi_credential_manager.check_cache(mock_redis_client, "test_key")
        assert result is None

        # Verify that logger.error was called
        mock_logger.error.assert_called_once_with(
            "Redis error while checking cache for key %s: %s", "test_key", redis_error
        )


class TestSmartapiCredentialManagerGenerateNewCredentials:
    """
    Test class for _generate_new_credentials method.
    """

    def test_generate_new_credentials_success(
        self,
        mocker,
        mock_smartapi_credential_input: SmartAPICredentialInput,
        mock_smartapi_credential_output: SmartAPICredentialOutput,
        smartapi_credential_manager: SmartapiCredentialManager,
        mock_smart_connect: MagicMock,
    ):
        """
        Test _generate_new_credentials method with successful API response.
        """
        mock_smart_connect_class = mocker.patch(
            "app.utils.credentials.smartapi_credential_manager.SmartConnect"
        )
        mock_pyotp = mocker.patch(
            "app.utils.credentials.smartapi_credential_manager.pyotp"
        )

        # Setup mocks
        mock_smart_connect_class.return_value = mock_smart_connect
        mock_totp_instance = MagicMock()
        mock_totp_instance.now.return_value = "123456"
        mock_pyotp.TOTP.return_value = mock_totp_instance

        result = smartapi_credential_manager._generate_new_credentials(
            mock_smartapi_credential_input
        )

        assert isinstance(result, SmartAPICredentialOutput)
        assert all(
            getattr(result, key) == val
            for key, val in mock_smartapi_credential_output.to_dict().items()
        )

        # Verify mock calls
        mock_smart_connect_class.assert_called_once_with("test_api_key")
        mock_pyotp.TOTP.assert_called_once_with("test_token")
        mock_smart_connect.generateSession.assert_called_once_with(
            "test_client_id", "test_password", "123456"
        )


class TestSmartapiCredentialManagerGetNextExpiryTime:
    """
    Test class for get_next_expiry_time method.
    """

    @pytest.fixture
    def mock_datetime_class(self, mocker):
        """
        Fixture to mock datetime class.
        """
        return mocker.patch(
            "app.utils.credentials.smartapi_credential_manager.datetime"
        )

    def test_get_next_expiry_time_before_5am(
        self,
        mock_datetime_class,
        mock_datetime_before_5am: datetime,
        smartapi_credential_manager: SmartapiCredentialManager,
    ):
        """
        Test get_next_expiry_time method when current time is before 5 AM.
        """
        # MagicMock datetime.now to return time before 5 AM
        mock_datetime_class.now.return_value = mock_datetime_before_5am

        result = smartapi_credential_manager.get_next_expiry_time()

        # Should return seconds until 5 AM same day
        expected_5am = mock_datetime_before_5am.replace(
            hour=5, minute=0, second=0, microsecond=0
        )
        expected_seconds = int(
            (expected_5am - mock_datetime_before_5am).total_seconds()
        )

        assert result == expected_seconds
        assert result > 0

    def test_get_next_expiry_time_after_5am(
        self,
        mock_datetime_class,
        mock_datetime_after_5am: datetime,
        smartapi_credential_manager: SmartapiCredentialManager,
    ):
        """
        Test get_next_expiry_time method when current time is after 5 AM.
        """
        # MagicMock datetime.now to return time after 5 AM
        mock_datetime_class.now.return_value = mock_datetime_after_5am

        result = smartapi_credential_manager.get_next_expiry_time()

        # Should return seconds until 5 AM next day
        expected_5am_next_day = mock_datetime_after_5am.replace(
            hour=5, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)
        expected_seconds = int(
            (expected_5am_next_day - mock_datetime_after_5am).total_seconds()
        )

        assert result == expected_seconds
        assert result > 0


class TestSmartapiCredentialManagerCacheCredentials:
    """
    Test class for cache_credentials method.
    """

    def test_cache_credentials_success(
        self,
        mocker,
        mock_smartapi_credential_output: SmartAPICredentialOutput,
        mock_redis_client: MagicMock,
        smartapi_credential_manager: SmartapiCredentialManager,
        mock_logger,
    ):
        """
        Test cache_credentials method with successful caching.
        """
        # Setup mock pipeline
        mock_pipeline = mock_redis_client.pipeline.return_value

        mocker.patch.object(
            smartapi_credential_manager, "get_next_expiry_time", return_value=3600
        )
        smartapi_credential_manager.cache_credentials(
            mock_redis_client, "test_key", mock_smartapi_credential_output
        )

        # Verify Redis calls
        mock_redis_client.pipeline.assert_called_once()
        mock_pipeline.hset.assert_called_once_with(
            "test_key", mapping=mock_smartapi_credential_output.to_dict()
        )
        mock_pipeline.expire.assert_called_once_with("test_key", 3600)
        mock_pipeline.execute.assert_called_once()

        # Verify that logger.info was called
        mock_logger.info.assert_called_once_with(
            "Credentials cached successfully for key: %s for next %d seconds",
            "test_key",
            3600,
        )

    def test_cache_credentials_redis_error(
        self,
        mocker,
        mock_smartapi_credential_output: SmartAPICredentialOutput,
        mock_redis_client: MagicMock,
        mock_logger: MagicMock,
        smartapi_credential_manager: SmartapiCredentialManager,
    ):
        """
        Test cache_credentials method when Redis raises an error.
        """
        # Setup mock pipeline to raise error
        mock_pipeline = mock_redis_client.pipeline.return_value
        redis_error = redis.RedisError("Caching failed")
        mock_pipeline.execute.side_effect = redis_error

        mocker.patch.object(
            smartapi_credential_manager, "get_next_expiry_time", return_value=3600
        )
        smartapi_credential_manager.cache_credentials(
            mock_redis_client, "test_key", mock_smartapi_credential_output
        )

        # Verify that logger.error was called
        mock_logger.error.assert_called_once_with(
            "Error caching credentials for key %s: %s", "test_key", redis_error
        )


class TestSmartapiCredentialManagerGetHeaders:
    """
    Test class for get_headers method.
    """

    def test_get_headers_returns_correct_format(
        self, smartapi_credential_manager: SmartapiCredentialManager
    ):
        """
        Test get_headers method returns headers in correct format.
        """

        headers = smartapi_credential_manager.get_headers()

        expected_headers = {
            "Authorization": "test_jwt_token",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-UserType": "USER",
            "X-SourceID": "WEB",
            "X-ClientLocalIP": "CLIENT_LOCAL_IP",
            "X-ClientPublicIP": "CLIENT_PUBLIC_IP",
            "X-MACAddress": "MAC_ADDRESS",
            "X-PrivateKey": "test_api_key",
        }

        assert headers == expected_headers
        assert isinstance(headers, dict)
        assert all(
            isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
        )


class TestSmartapiCredentialManagerGenerateCredentials:
    """
    Test class for generate_credentials class method.
    """

    @pytest.fixture
    def mock_redis_connection(self, mocker, mock_redis_client):
        """
        Fixture to mock RedisSyncConnection class.
        """
        mock_redis_connection = mocker.patch(
            "app.utils.credentials.smartapi_credential_manager.RedisSyncConnection"
        )
        mock_redis_connection.return_value = mock_redis_connection
        mock_redis_connection.get_connection.return_value = mock_redis_client
        return mock_redis_connection

    def test_generate_credentials_cache_hit(
        self,
        mocker,
        mock_redis_connection,
        mock_smartapi_credential_input: SmartAPICredentialInput,
        mock_smartapi_credential_output: SmartAPICredentialOutput,
        mock_logger: MagicMock,
    ):
        """
        Test generate_credentials method when credentials found in cache.
        """
        # MagicMock the check_cache method and logger using mocker
        mocker.patch.object(
            SmartapiCredentialManager,
            "check_cache",
            return_value=mock_smartapi_credential_output,
        )
        result = SmartapiCredentialManager.generate_credentials(
            mock_smartapi_credential_input
        )
        assert result == mock_smartapi_credential_output

        # Verify that logger.info was called
        mock_logger.info.assert_called_once_with(
            "Credentials found in cache for key: %s",
            "smartapi_credentials:test_client_id:0",
        )
        mock_redis_connection.close_connection.assert_called_once()

    def test_generate_credentials_cache_miss(
        self,
        mocker: MockerFixture,
        mock_redis_connection,
        mock_smartapi_credential_input: SmartAPICredentialInput,
        mock_smartapi_credential_output: SmartAPICredentialOutput,
        mock_logger: MagicMock,
    ):
        """
        Test generate_credentials method when credentials not found in cache.
        """
        # MagicMock methods to simulate cache miss and credential generation
        mock_check_cache = mocker.patch.object(
            SmartapiCredentialManager, "check_cache", return_value=None
        )
        mock_generate_new_credentials = mocker.patch.object(
            SmartapiCredentialManager,
            "_generate_new_credentials",
            return_value=mock_smartapi_credential_output,
        )
        mock_cache_credentials = mocker.patch.object(
            SmartapiCredentialManager, "cache_credentials"
        )

        result = SmartapiCredentialManager.generate_credentials(
            mock_smartapi_credential_input,
        )

        assert result == mock_smartapi_credential_output
        mock_redis_connection.close_connection.assert_called_once()

        # Verify the mock calls
        mock_logger.info.assert_called_with(
            "Generating new credentials for key: %s",
            "smartapi_credentials:test_client_id:0",
        )

        mock_check_cache.assert_called_once_with(
            mock_redis_connection.get_connection.return_value,
            "smartapi_credentials:test_client_id:0",
        )
        mock_generate_new_credentials.assert_called_once_with(
            mock_smartapi_credential_input
        )
        mock_cache_credentials.assert_called_once_with(
            mock_redis_connection.get_connection.return_value,
            "smartapi_credentials:test_client_id:0",
            mock_smartapi_credential_output,
        )

    def test_generate_credentials_ensures_connection_cleanup(
        self,
        mocker,
        mock_redis_connection,
        mock_smartapi_credential_input: SmartAPICredentialInput,
    ):
        """
        Test generate_credentials method ensures Redis connection cleanup even on exception.
        """

        # MagicMock check_cache to raise an exception
        mocker.patch.object(
            SmartapiCredentialManager,
            "check_cache",
            side_effect=Exception("Test error"),
        )
        with pytest.raises(Exception, match="Test error"):
            SmartapiCredentialManager.generate_credentials(
                mock_smartapi_credential_input
            )

        # Ensure connection is still closed
        mock_redis_connection.close_connection.assert_called_once()

    @pytest.mark.parametrize(
        "connection_num, expected_key",
        [
            (0, "smartapi_credentials:test_client:0"),
            (1, "smartapi_credentials:test_client:1"),
            (999, "smartapi_credentials:test_client:999"),
        ],
    )
    def test_generate_credentials_key_format(
        self,
        mocker,
        mock_redis_connection,
        connection_num,
        expected_key: str,
        mock_smartapi_credential_input: SmartAPICredentialInput,
        mock_smartapi_credential_output: SmartAPICredentialOutput,
    ):
        """
        Test that generate_credentials creates Redis keys in the correct format.
        """
        mock_smartapi_credential_input.client_id = "test_client"
        mock_check = mocker.patch.object(
            SmartapiCredentialManager,
            "check_cache",
            return_value=mock_smartapi_credential_output,
        )
        mock_smartapi_credential_input.connection_num = connection_num
        SmartapiCredentialManager.generate_credentials(mock_smartapi_credential_input)

        # Verify that check_cache was called with the correct key
        mock_check.assert_called_once_with(
            mock_redis_connection.get_connection.return_value, expected_key
        )

        mock_redis_connection.get_connection.assert_called_once()
        mock_redis_connection.close_connection.assert_called_once()
