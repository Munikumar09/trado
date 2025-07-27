"""
Shared fixtures for credential manager tests.
"""

from datetime import datetime
from typing import Generator
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from redis import Redis

from app.data_layer.data_models.credential_model import (
    SmartAPICredentialInput,
    SmartAPICredentialOutput,
    UplinkCredentialInput,
    UplinkCredentialOutput,
)


@pytest.fixture
def mock_smartapi_credential_input() -> SmartAPICredentialInput:
    """
    Fixture to provide mock SmartAPI credential input.
    """
    return SmartAPICredentialInput(
        api_key="test_api_key",
        client_id="test_client_id",
        pwd="test_password",
        token="test_token",
        connection_num=0,
    )


@pytest.fixture
def mock_smartapi_credential_output() -> SmartAPICredentialOutput:
    """
    Fixture to provide mock SmartAPI credential output.
    """
    return SmartAPICredentialOutput(
        access_token="test_jwt_token",
        refresh_token="test_refresh_token",
        feed_token="test_feed_token",
        user_id="test_user_id",
    )


@pytest.fixture
def mock_redis_client(mocker) -> Generator[MagicMock, None, None]:
    """
    Fixture to provide a mock Redis client.
    """
    mock_client = mocker.MagicMock(spec=Redis)
    mock_pipeline = mocker.MagicMock()
    mock_client.pipeline.return_value = mock_pipeline
    yield mock_client


@pytest.fixture
def mock_smart_connect() -> MagicMock:
    """
    Fixture to provide a mock SmartConnect instance.
    """
    mock_smart_connect = MagicMock()
    mock_smart_connect.generateSession.return_value = {
        "data": {
            "jwtToken": "test_jwt_token",
            "refreshToken": "test_refresh_token",
            "feedToken": "test_feed_token",
            "clientcode": "test_user_id",
        }
    }
    return mock_smart_connect


@pytest.fixture
def mock_pyotp() -> MagicMock:
    """
    Fixture to provide a mock pyotp TOTP instance.
    """
    mock_totp = MagicMock()
    mock_totp.now.return_value = "123456"
    return mock_totp


@pytest.fixture
def ist_timezone() -> ZoneInfo:
    """
    Fixture to provide IST timezone.
    """
    return ZoneInfo("Asia/Kolkata")


@pytest.fixture
def mock_datetime_before_5am(ist_timezone: ZoneInfo) -> datetime:
    """
    Fixture to provide a datetime before 5 AM IST.
    """
    return datetime(2024, 7, 26, 3, 30, 0, tzinfo=ist_timezone)


@pytest.fixture
def mock_datetime_after_5am(ist_timezone: ZoneInfo) -> datetime:
    """
    Fixture to provide a datetime after 5 AM IST.
    """
    return datetime(2024, 7, 26, 10, 30, 0, tzinfo=ist_timezone)


@pytest.fixture
def mock_redis_hash_data() -> dict[str, str]:
    """
    Fixture to provide mock Redis hash data.
    """
    return {
        "access_token": "cached_access_token",
        "refresh_token": "cached_refresh_token",
        "feed_token": "cached_feed_token",
        "user_id": "cached_user_id",
    }


@pytest.fixture
def mock_uplink_credential_input() -> UplinkCredentialInput:
    """
    Fixture to provide mock Uplink credential input.
    """
    return UplinkCredentialInput(
        api_key="test_uplink_api_key",
        secret_key="test_uplink_secret_key",
        totp_key="test_uplink_totp_key",
        mobile_no="1234567890",
        pin="123456",
    )


@pytest.fixture
def mock_uplink_credential_output() -> UplinkCredentialOutput:
    """
    Fixture to provide mock Uplink credential output.
    """
    return UplinkCredentialOutput(access_token="test_uplink_access_token")


@pytest.fixture
def mock_uplink_redis_hash_data() -> dict[str, str]:
    """
    Fixture to provide mock Redis hash data for Uplink credentials.
    """
    return {"access_token": "cached_uplink_access_token"}


@pytest.fixture
def mock_datetime_before_3_30pm(ist_timezone: ZoneInfo) -> datetime:
    """
    Fixture to provide a datetime before 3:30 PM IST.
    """
    return datetime(2024, 7, 26, 14, 30, 0, tzinfo=ist_timezone)


@pytest.fixture
def mock_datetime_after_3_30pm(ist_timezone: ZoneInfo) -> datetime:
    """
    Fixture to provide a datetime after 3:30 PM IST.
    """
    return datetime(2024, 7, 26, 16, 30, 0, tzinfo=ist_timezone)
