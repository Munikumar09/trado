# pylint: disable=unused-argument
"""
This module contains unit tests for the BrevoEmailProvider class. It includes tests 
for initialization using environment variables and configuration objects, as well as
tests for sending notification emails and handling API exceptions.
"""

from typing import List, Tuple
from unittest.mock import patch

import pytest
from brevo_python.rest import ApiException
from fastapi import HTTPException
from omegaconf import DictConfig
from pytest import MonkeyPatch
from pytest_mock import MockFixture, MockType

from app.notification.email.providers.brevo import BrevoEmailProvider

invalid_cfg_with_error: List[Tuple[DictConfig, str]] = [
    (
        DictConfig({"security": {"api_key_name": "BREVO_API_KEY"}}),
        "Brevo sender details are required",
    ),
    (
        DictConfig({"sender": {"name": "SENDER_NAME", "email": "SENDER_EMAIL"}}),
        "brevo_api_key_name is required",
    ),
    (
        DictConfig(
            {
                "sender": {"name": "SENDER_NAME"},
                "security": {"api_key_name": "BREVO_API_KEY"},
            }
        ),
        "Brevo sender details are required",
    ),
    (
        DictConfig(
            {
                "sender": {"email": "SENDER_EMAIL"},
                "security": {"api_key_name": "BREVO_API_KEY"},
            }
        ),
        "Brevo sender details are required",
    ),
    (
        DictConfig(
            {
                "sender": {"name": "INVALID_NAME", "email": "SENDER_EMAIL"},
                "security": {"api_key_name": "BREVO_API_KEY"},
            }
        ),
        "Missing required environment variable: INVALID_NAME",
    ),
]


#################### Fixtures ####################


@pytest.fixture
def mock_env_vars(monkeypatch: MonkeyPatch):
    """
    Fixture to set environment variables for testing.
    """
    monkeypatch.setenv("SENDER_NAME", "Test Sender")
    monkeypatch.setenv("SENDER_EMAIL", "sender@example.com")
    monkeypatch.setenv("BREVO_API_KEY", "test_api_key")


@pytest.fixture
def mock_logger(mocker: MockFixture) -> MockType:
    """
    Fixture to mock the logger.
    """
    return mocker.patch("app.notification.email.providers.brevo.logger")


#################### Tests ####################


# Test: 1
def test_init_from_env_vars(mock_env_vars: None):
    """
    Test initialization of BrevoEmailProvider using environment variables.
    """
    provider = BrevoEmailProvider("SENDER_NAME", "SENDER_EMAIL", "BREVO_API_KEY")
    assert provider.sender_name == "Test Sender"
    assert provider.sender_email == "sender@example.com"
    assert provider.configuration.api_key["api-key"] == "test_api_key"


# Test: 2
def test_init_from_cfg(mock_env_vars: None):
    """
    Test initialization of BrevoEmailProvider using a configuration object.
    """
    cfg = DictConfig(
        {
            "sender": {"name": "SENDER_NAME", "email": "SENDER_EMAIL"},
            "security": {"api_key_name": "BREVO_API_KEY"},
        }
    )
    provider = BrevoEmailProvider.from_cfg(cfg)
    assert provider.sender_name == "Test Sender"
    assert provider.sender_email == "sender@example.com"
    assert provider.configuration.api_key["api-key"] == "test_api_key"


# Test: 3
@pytest.mark.parametrize("invalid_config", invalid_cfg_with_error)
def test_init_from_cfg_missing_sender(
    invalid_config: Tuple[DictConfig, str], mock_env_vars: None
):
    """
    Test initialization of BrevoEmailProvider with missing sender details in the configuration.
    """
    cfg, error_msg = invalid_config

    with pytest.raises(ValueError) as e:
        BrevoEmailProvider.from_cfg(cfg)

    assert str(e.value) == error_msg


# Test: 4
@patch("brevo_python.TransactionalEmailsApi.send_transac_email")
def test_send_notification_success(mock_send_email: MockType, mock_env_vars: None):
    """
    Test successful sending of a notification email.
    """
    provider = BrevoEmailProvider("SENDER_NAME", "SENDER_EMAIL", "BREVO_API_KEY")
    provider.send_notification("123456", "recipient@example.com", "Recipient Name")
    mock_send_email.assert_called_once()


# Test: 5
@patch("brevo_python.TransactionalEmailsApi.send_transac_email")
def test_send_notification_api_exception(
    mock_send_email: MockType,
    mock_env_vars: None,
):
    """
    Test handling of an API exception when sending a notification email.
    """
    mock_send_email.side_effect = ApiException("API error")
    provider = BrevoEmailProvider("SENDER_NAME", "SENDER_EMAIL", "BREVO_API_KEY")

    with pytest.raises(HTTPException):
        provider.send_notification("123456", "recipient@example.com", "Recipient Name")
