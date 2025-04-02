from time import sleep

import pytest
from fastapi import HTTPException
from pytest_mock import MockType

from app.routers.authentication.jwt_tokens import (
    access_token_from_refresh_token,
    create_token,
    decode_token,
)
from app.utils.constants import JWT_REFRESH_SECRET, JWT_SECRET


# Test: 1
def test_create_token(token_data: dict[str, str]):
    """
    Test create_token function
    """
    token = create_token(token_data, JWT_SECRET, 15)
    decoded_data = decode_token(token, JWT_SECRET)
    assert decoded_data["user_id"] == token_data["user_id"]
    assert decoded_data["email"] == token_data["email"]


# Test: 2
def test_access_token_from_refresh_token(
    token_data: dict[str, str], mock_session: MockType
):
    """
    Test access_token_from_refresh_token function
    """
    refresh_token = create_token(token_data, JWT_REFRESH_SECRET, 15)
    tokens = access_token_from_refresh_token(refresh_token)
    assert "access_token" in tokens
    assert "refresh_token" in tokens

    dummy_token = create_token(
        {"user_id": -124, "email": "nothing"}, JWT_REFRESH_SECRET, 1000
    )
    mock_session.first.return_value = None

    with pytest.raises(HTTPException) as exc:
        access_token_from_refresh_token(dummy_token)

    assert exc.value.detail == "User not found"


# Test: 3
def test_decode_token(token_data: dict[str, str]):
    """
    Test decode_token function
    """
    # Test: 3.1 ( Valid token )
    token = create_token(token_data, JWT_SECRET, 0.05)

    decoded_data = decode_token(token, JWT_SECRET)
    assert decoded_data["user_id"] == token_data["user_id"]
    assert decoded_data["email"] == token_data["email"]

    # Test: 3.2 ( Expired token )
    sleep(5)
    with pytest.raises(HTTPException) as exc:
        decode_token(token, JWT_SECRET)
    assert exc.value.detail == "Token has expired"

    # Test: 3.3 ( Invalid token )
    with pytest.raises(HTTPException) as exc:
        decode_token("invalid_token", JWT_SECRET)
    assert exc.value.detail == "Invalid token"
