# pylint: disable=no-value-for-parameter too-many-lines
from datetime import datetime, timezone
from time import sleep
from typing import cast

import fakeredis
import pytest
import pytest_asyncio
from _pytest._code.code import ExceptionInfo
from fastapi import HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from fastapi_limiter import FastAPILimiter
from httpx import AsyncClient
from httpx._models import Response
from pytest import MonkeyPatch
from pytest_mock import MockFixture
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

from app.data_layer.database.crud.user_crud import (
    get_user_by_attr,
    get_user_verification,
    update_user,
)
from app.data_layer.database.db_connections.sqlite import (
    create_db_and_tables,
    get_session,
)
from app.data_layer.database.models.user_model import UserVerification
from app.routers.authentication.authenticate import (
    create_or_update_user_verification,
    create_token,
    signup_user,
    update_user_verification_status,
)
from app.routers.authentication.authentication import router
from app.routers.authentication.user_validation import verify_password
from app.schemas.user_model import UserSignup
from app.utils.constants import JWT_REFRESH_SECRET, JWT_SECRET
from main import app

client = TestClient(router)


#################### FIXTURES ####################

VALID_PASSWORD = "Password123!"


class MockNotificationProvider:
    """
    This class mocks the notification provider.
    """

    def __init__(self):
        self.code = None
        self.recipient_email = None
        self.recipient_name = None

    def send_notification(self, code: str, recipient_email: str, recipient_name: str):
        """
        Test method to send a notification.
        """
        self.code = code
        self.recipient_email = recipient_email
        self.recipient_name = recipient_name
        return {
            "message": f"Verification code sent to {recipient_email}. Valid for 10 minutes."
        }


@pytest.fixture
def mock_notification_provider(mocker: MockFixture):
    """
    Mock the notification provider.
    """
    return mocker.patch(
        "app.routers.authentication.authentication.email_notification_provider",
        MockNotificationProvider(),
    )


@pytest.fixture
def test_verification_code():
    """
    Verification code for testing.
    """
    return "123456"


@pytest.fixture(autouse=True)
def mock_session(monkeypatch: MonkeyPatch):
    """
    Mock the session object.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_db_and_tables(engine)
    monkeypatch.setattr(
        "app.data_layer.database.db_connections.postgresql.get_session",
        lambda: get_session(engine),
    )


@pytest_asyncio.fixture(scope="function", autouse=True)
async def initialize_rate_limiter():
    """
    Initialize the rate limiter with a FAKE Redis connection.
    """
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    await FastAPILimiter.init(redis)
    yield redis
    await redis.flushall()
    await redis.aclose()


@pytest_asyncio.fixture(scope="function")
async def async_client():
    """
    Use AsyncClient for async requests to FastAPI.
    """
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client  # ✅ Ensure all requests are properly awaited


#################### HELPER FUNCTIONS ####################


def validate_http_exception(
    exc: ExceptionInfo[HTTPException], status_code: int, detail: str
):
    """
    Validate the HTTPException.
    """
    assert exc.value.status_code == status_code
    assert exc.value.detail == detail


def validate_verification_code_sent(
    response: Response, email: str, mock_notification_provider: MockNotificationProvider
):
    """
    Validate the verification code sent with the following assertions:
    - The response status code is 200
    - The code is not None
    - The code is of length 6
    - The recipient email is the same as the email passed
    """
    assert response.status_code == 200
    assert mock_notification_provider.code is not None
    assert len(mock_notification_provider.code) == 6
    assert mock_notification_provider.recipient_email == email


def validate_user_verification(
    user_verification: UserVerification, email: str, code: str
):
    """
    Validate the user verification object with the following assertions:
    - The user verification object is not None
    - The email is the same as the email passed
    - The code is the same as the code passed
    - The expiration time is greater than the current time
    - The verification datetime is the same as the current date
    - The reverified datetime is the same as the current date
    """
    assert user_verification is not None
    assert user_verification.email == email
    assert user_verification.verification_code == code
    assert user_verification.expiration_time > int(
        datetime.now(timezone.utc).timestamp()
    )
    assert (
        cast(datetime, user_verification.verification_datetime).date()
        == datetime.now().date()
    )
    assert (
        cast(datetime, user_verification.reverified_datetime).date()
        == datetime.now().date()
    )


#################### TESTS ####################


def test_signup(sign_up_data: UserSignup) -> None:
    """
    Test the signup functionality:
    - Successful signup
    - Signup with existing email
    - Signup with existing phone number
    """
    # Successful signup
    response = client.post("/authentication/signup", json=sign_up_data.dict())

    assert response.status_code == 201
    assert (
        response.json()["message"]
        == "User created successfully. Please verify your email to activate your account"
    )
    user = get_user_by_attr("email", sign_up_data.email)
    assert user is not None
    assert user.username == sign_up_data.username
    assert user.email == sign_up_data.email
    assert user.phone_number == sign_up_data.phone_number
    assert user.gender == sign_up_data.gender
    assert not user.is_verified

    # Test for the existing user
    with pytest.raises(HTTPException) as exc:
        client.post("/authentication/signup", json=sign_up_data.dict())

    validate_http_exception(exc, status.HTTP_400_BAD_REQUEST, "email already exists")

    # Test for existing phone number
    invalid_data = sign_up_data.copy()
    invalid_data.email = "invalid_email"
    with pytest.raises(HTTPException) as exc:
        client.post("/authentication/signup", json=invalid_data.dict())

    validate_http_exception(
        exc, status.HTTP_400_BAD_REQUEST, "phone_number already exists"
    )


def test_signin_invalid_email(
    sign_up_data: UserSignup,
) -> None:
    """
    Test the signin functionality with an invalid email.
    """
    signup_user(sign_up_data)
    with pytest.raises(HTTPException) as exc:
        client.post(
            "/authentication/signin",
            json={"email": "invalid_email", "password": sign_up_data.password},
        )
    validate_http_exception(
        exc,
        status.HTTP_404_NOT_FOUND,
        "User does not exist with given email: invalid_email",
    )


def test_signin_invalid_password(
    sign_up_data: UserSignup,
) -> None:
    """
    Test the signin functionality with an invalid password.
    """
    signup_user(sign_up_data)
    with pytest.raises(HTTPException) as exc:
        client.post(
            "/authentication/signin",
            json={"email": sign_up_data.email, "password": "invalid_password"},
        )
    validate_http_exception(exc, status.HTTP_401_UNAUTHORIZED, "Passwords do not match")


def test_signin_user_not_verified(
    sign_up_data: UserSignup,
) -> None:
    """
    Test the signin functionality for a user who has not verified their email.
    """
    signup_user(sign_up_data)
    with pytest.raises(HTTPException) as exc:
        client.post(
            "/authentication/signin",
            json={"email": sign_up_data.email, "password": sign_up_data.password},
        )
    validate_http_exception(exc, status.HTTP_401_UNAUTHORIZED, "User is not verified")


def test_signin_success(
    sign_up_data: UserSignup,
) -> None:
    """
    Test the successful signin functionality.
    """
    signup_user(sign_up_data)
    update_user_verification_status(sign_up_data.email)
    response = client.post(
        "/authentication/signin",
        json={"email": sign_up_data.email, "password": sign_up_data.password},
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert "refresh_token" in response.json()
    assert response.json()["message"] == "Login successful"


@pytest.mark.asyncio
async def test_send_verification_code_invalid_email(
    async_client,
    sign_up_data: UserSignup,
) -> None:
    """
    Test sending a verification code to an invalid email.
    """
    signup_user(sign_up_data)
    response = await async_client.post(
        "/authentication/send-verification-code",
        params={"email": "invalid_email"},
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == {
        "detail": "User does not exist with given email: invalid_email"
    }


@pytest.mark.asyncio
async def test_send_verification_code_already_verified(
    async_client,
    sign_up_data: UserSignup,
) -> None:
    """
    Test sending a verification code to an already verified email.
    """
    signup_user(sign_up_data)
    update_user_verification_status(sign_up_data.email)

    response = await async_client.post(
        "/authentication/send-verification-code",
        params={"email": sign_up_data.email},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Email is already verified."}


@pytest.mark.asyncio
async def test_send_verification_code_success(
    async_client,
    sign_up_data: UserSignup,
    mock_notification_provider,
) -> None:
    """
    Test the successful sending of a verification code.
    """
    signup_user(sign_up_data)
    update_user_verification_status(sign_up_data.email, False)
    response = await async_client.post(
        "/authentication/send-verification-code", params={"email": sign_up_data.email}
    )
    validate_verification_code_sent(
        response, sign_up_data.email, mock_notification_provider
    )
    assert response.json() == {
        "message": f"Verification code sent to {sign_up_data.email}. Valid for 10 minutes."
    }
    user_verification = get_user_verification(sign_up_data.email)
    validate_user_verification(
        user_verification, sign_up_data.email, mock_notification_provider.code
    )


def test_verify_code_invalid_email(
    sign_up_data: UserSignup, test_verification_code
) -> None:
    """
    Test verifying a code with an invalid email.
    """
    signup_user(sign_up_data)
    with pytest.raises(HTTPException) as exc:
        client.post(
            "/authentication/verify-code",
            json={
                "verification_code": test_verification_code,
                "email": "invalid_email",
            },
        )
    validate_http_exception(
        exc,
        status.HTTP_404_NOT_FOUND,
        "User does not exist with email: invalid_email",
    )


def test_verify_code_invalid_code(
    sign_up_data: UserSignup, test_verification_code
) -> None:
    """
    Test verifying an invalid code.
    """
    signup_user(sign_up_data)
    user_verification = UserVerification(
        email=sign_up_data.email,
        verification_code=test_verification_code,
        expiration_time=int(datetime.now(timezone.utc).timestamp() - 600),
    )
    create_or_update_user_verification(user_verification)
    with pytest.raises(HTTPException) as exc:
        client.post(
            "/authentication/verify-code",
            json={"verification_code": "000000", "email": sign_up_data.email},
        )
    validate_http_exception(
        exc, status.HTTP_400_BAD_REQUEST, "Invalid verification code"
    )


def test_verify_code_expired(sign_up_data: UserSignup, test_verification_code) -> None:
    """
    Test verifying an expired code.
    """
    signup_user(sign_up_data)
    user_verification = UserVerification(
        email=sign_up_data.email,
        verification_code=test_verification_code,
        expiration_time=int(datetime.now(timezone.utc).timestamp() - 600),
    )
    create_or_update_user_verification(user_verification)
    with pytest.raises(HTTPException) as exc:
        client.post(
            "/authentication/verify-code",
            json={
                "verification_code": test_verification_code,
                "email": sign_up_data.email,
            },
        )
    validate_http_exception(
        exc, status.HTTP_400_BAD_REQUEST, "Verification code has expired. Try again"
    )


def test_verify_code_success(sign_up_data: UserSignup, test_verification_code) -> None:
    """
    Test the successful verification of a code.
    """
    signup_user(sign_up_data)
    user_verification = UserVerification(
        email=sign_up_data.email,
        verification_code=test_verification_code,
        expiration_time=int(datetime.now(timezone.utc).timestamp() + 700),
    )
    create_or_update_user_verification(user_verification)
    response = client.post(
        "/authentication/verify-code",
        json={"verification_code": test_verification_code, "email": sign_up_data.email},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Email verified successfully"
    assert get_user_by_attr("email", sign_up_data.email).is_verified
    user_verification = get_user_verification(sign_up_data.email)
    assert user_verification.expiration_time == 0


@pytest.mark.asyncio
async def test_send_reset_password_code_invalid_email(async_client) -> None:
    """
    Test sending a reset password code to an invalid email.
    """

    response = await async_client.post(
        "/authentication/send-reset-password-code",
        params={"email": "invalid_email"},
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == {
        "detail": "User does not exist with given email: invalid_email"
    }


@pytest.mark.asyncio
async def test_send_reset_password_code_success(
    async_client, sign_up_data: UserSignup, mock_notification_provider
) -> None:
    """
    Test the successful sending of a reset password code.
    """
    signup_user(sign_up_data)

    assert get_user_verification(sign_up_data.email) is None
    response = await async_client.post(
        "/authentication/send-reset-password-code", params={"email": sign_up_data.email}
    )
    validate_verification_code_sent(
        response, sign_up_data.email, mock_notification_provider
    )
    assert response.json() == {
        "message": f"Reset password code sent to {sign_up_data.email}. Valid for 10 minutes."
    }

    user_verification = get_user_verification(sign_up_data.email)
    validate_user_verification(
        user_verification, sign_up_data.email, mock_notification_provider.code
    )


def test_reset_password_invalid_email(
    sign_up_data: UserSignup,
) -> None:
    """
    Test resetting the password with an invalid email.
    """
    signup_user(sign_up_data)
    with pytest.raises(HTTPException) as exc:
        client.post(
            "/authentication/reset-password",
            json={
                "email": "invalid_email",
                "password": "new_password",
                "verification_code": "123456",
            },
        )
    validate_http_exception(
        exc, status.HTTP_404_NOT_FOUND, "User does not exist with email: invalid_email"
    )


def test_reset_password_invalid_code(
    sign_up_data: UserSignup,
    test_verification_code,
) -> None:
    """
    Test resetting the password with an invalid verification code.
    """
    signup_user(sign_up_data)
    user_verification = UserVerification(
        email=sign_up_data.email,
        verification_code=test_verification_code,
        expiration_time=int(datetime.now(timezone.utc).timestamp() - 600),
    )
    create_or_update_user_verification(user_verification)
    with pytest.raises(HTTPException) as exc:
        client.post(
            "/authentication/reset-password",
            json={
                "email": sign_up_data.email,
                "password": "new_password",
                "verification_code": "023456",
            },
        )
    validate_http_exception(
        exc, status.HTTP_400_BAD_REQUEST, "Invalid verification code"
    )


def test_reset_password_expired_code(
    sign_up_data: UserSignup,
    test_verification_code,
) -> None:
    """
    Test resetting the password with an expired verification code.
    """
    signup_user(sign_up_data)
    user_verification = UserVerification(
        email=sign_up_data.email,
        verification_code=test_verification_code,
        expiration_time=int(datetime.now(timezone.utc).timestamp() - 600),
    )
    create_or_update_user_verification(user_verification)
    with pytest.raises(HTTPException) as exc:
        client.post(
            "/authentication/reset-password",
            json={
                "email": sign_up_data.email,
                "password": "new_password",
                "verification_code": test_verification_code,
            },
        )
    validate_http_exception(
        exc, status.HTTP_400_BAD_REQUEST, "Verification code has expired. Try again"
    )


def test_reset_password_invalid_password(
    sign_up_data: UserSignup,
    test_verification_code,
) -> None:
    """
    Test resetting the password with an invalid new password.
    """
    signup_user(sign_up_data)
    user_verification = UserVerification(
        email=sign_up_data.email,
        verification_code=test_verification_code,
        expiration_time=int(datetime.now(timezone.utc).timestamp() + 700),
    )
    create_or_update_user_verification(user_verification)
    with pytest.raises(HTTPException) as exc:
        client.post(
            "/authentication/reset-password",
            json={
                "email": sign_up_data.email,
                "password": "new_password",
                "verification_code": test_verification_code,
            },
        )
    validate_http_exception(
        exc,
        status.HTTP_400_BAD_REQUEST,
        "Password must be at least 8 characters long and include an uppercase "
        "letter, lowercase letter, digit, and special character",
    )


def test_reset_password_same_as_old(
    sign_up_data: UserSignup,
    test_verification_code,
) -> None:
    """
    Test resetting the password with the same password as the old one.
    """
    signup_user(sign_up_data)
    user_verification = UserVerification(
        email=sign_up_data.email,
        verification_code=test_verification_code,
        expiration_time=int(datetime.now(timezone.utc).timestamp() + 700),
    )
    create_or_update_user_verification(user_verification)
    with pytest.raises(HTTPException) as exc:
        client.post(
            "/authentication/reset-password",
            json={
                "email": sign_up_data.email,
                "password": sign_up_data.password,
                "verification_code": test_verification_code,
            },
        )
    validate_http_exception(
        exc,
        status.HTTP_400_BAD_REQUEST,
        "New password cannot be the same as the old password",
    )


def test_reset_password_success(
    sign_up_data: UserSignup,
    test_verification_code,
) -> None:
    """
    Test the successful resetting of the password.
    """
    signup_user(sign_up_data)
    user_verification = UserVerification(
        email=sign_up_data.email,
        verification_code=test_verification_code,
        expiration_time=int(datetime.now(timezone.utc).timestamp() + 700),
    )
    create_or_update_user_verification(user_verification)
    response = client.post(
        "/authentication/reset-password",
        json={
            "email": sign_up_data.email,
            "password": VALID_PASSWORD,
            "verification_code": test_verification_code,
        },
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Password reset successfully"
    user_verification = get_user_verification(sign_up_data.email)
    assert user_verification.expiration_time == 0
    assert (
        cast(datetime, user_verification.reverified_datetime).date()
        == datetime.now().date()
    )
    user = get_user_by_attr("email", sign_up_data.email)
    assert verify_password(VALID_PASSWORD, user.password, raise_exception=False)
    assert not verify_password(
        sign_up_data.password, user.password, raise_exception=False
    )


def test_change_password_invalid_email(
    sign_up_data: UserSignup,
) -> None:
    """
    Test changing the password with an invalid email.
    """
    signup_user(sign_up_data)
    with pytest.raises(HTTPException) as exc:
        client.post(
            "/authentication/change-password",
            json={
                "email": "invalid_email",
                "old_password": "old_password",
                "new_password": "new_password",
            },
        )
    validate_http_exception(
        exc,
        status.HTTP_404_NOT_FOUND,
        "User does not exist with given email: invalid_email",
    )


def test_change_password_invalid_old_password(
    sign_up_data: UserSignup,
) -> None:
    """
    Test changing the password with an invalid old password.
    """
    signup_user(sign_up_data)
    with pytest.raises(HTTPException) as exc:
        client.post(
            "/authentication/change-password",
            json={
                "email": sign_up_data.email,
                "old_password": "old_password",
                "new_password": "new_password",
            },
        )
    validate_http_exception(exc, status.HTTP_401_UNAUTHORIZED, "Passwords do not match")


def test_change_password_not_verified(
    sign_up_data: UserSignup,
) -> None:
    """
    Test changing the password for a user who has not verified their email.
    """
    signup_user(sign_up_data)
    with pytest.raises(HTTPException) as exc:
        client.post(
            "/authentication/change-password",
            json={
                "email": sign_up_data.email,
                "old_password": sign_up_data.password,
                "new_password": "new_password",
            },
        )
    validate_http_exception(
        exc,
        status.HTTP_400_BAD_REQUEST,
        "Email is not verified. Please verify the email first.",
    )


def test_change_password_invalid_new_password(
    sign_up_data: UserSignup,
) -> None:
    """
    Test changing the password with an invalid new password.
    """
    signup_user(sign_up_data)
    user = get_user_by_attr("email", sign_up_data.email)
    update_user(user.user_id, {"is_verified": True})
    with pytest.raises(HTTPException) as exc:
        client.post(
            "/authentication/change-password",
            json={
                "email": sign_up_data.email,
                "old_password": sign_up_data.password,
                "new_password": "new_password",
            },
        )
    validate_http_exception(
        exc,
        status.HTTP_400_BAD_REQUEST,
        "Password must be at least 8 characters long and include an uppercase letter, "
        "lowercase letter, digit, and special character",
    )


def test_change_password_same_as_old(
    sign_up_data: UserSignup,
) -> None:
    """
    Test changing the password with the same password as the old one.
    """
    signup_user(sign_up_data)
    user = get_user_by_attr("email", sign_up_data.email)
    update_user(user.user_id, {"is_verified": True})
    with pytest.raises(HTTPException) as exc:
        client.post(
            "/authentication/change-password",
            json={
                "email": sign_up_data.email,
                "old_password": sign_up_data.password,
                "new_password": sign_up_data.password,
            },
        )
    validate_http_exception(
        exc,
        status.HTTP_400_BAD_REQUEST,
        "New password cannot be the same as the old password",
    )


def test_change_password_success(
    sign_up_data: UserSignup,
) -> None:
    """
    Test the successful changing of the password.
    """
    signup_user(sign_up_data)
    user = get_user_by_attr("email", sign_up_data.email)
    update_user(user.user_id, {"is_verified": True})
    response = client.post(
        "/authentication/change-password",
        json={
            "email": sign_up_data.email,
            "old_password": sign_up_data.password,
            "new_password": VALID_PASSWORD,
        },
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Password changed successfully"
    user = get_user_by_attr("email", sign_up_data.email)
    assert verify_password(VALID_PASSWORD, user.password, raise_exception=False)
    assert not verify_password(
        sign_up_data.password, user.password, raise_exception=False
    )


def test_protected_endpoint_no_token(
    sign_up_data: UserSignup,
) -> None:
    """
    Test accessing a protected endpoint without a token.
    """
    signup_user(sign_up_data)
    with pytest.raises(HTTPException) as exc:
        client.get("/authentication/protected-endpoint")
    validate_http_exception(exc, status.HTTP_401_UNAUTHORIZED, "Not authenticated")


def test_protected_endpoint_missing_token(
    sign_up_data: UserSignup,
) -> None:
    """
    Test accessing a protected endpoint with a missing token.
    """
    signup_user(sign_up_data)
    with pytest.raises(HTTPException) as exc:
        client.get(
            "/authentication/protected-endpoint", headers={"Authorization": "Bearer "}
        )
    validate_http_exception(exc, status.HTTP_401_UNAUTHORIZED, "Token is missing")


def test_protected_endpoint_invalid_token(
    sign_up_data: UserSignup,
) -> None:
    """
    Test accessing a protected endpoint with an invalid token.
    """
    signup_user(sign_up_data)
    with pytest.raises(HTTPException) as exc:
        client.get(
            "/authentication/protected-endpoint",
            headers={"Authorization": "Bearer InvalidToken"},
        )
    validate_http_exception(exc, status.HTTP_401_UNAUTHORIZED, "Invalid token")


def test_protected_endpoint_expired_token(
    sign_up_data: UserSignup,
) -> None:
    """
    Test accessing a protected endpoint with an expired token.
    """
    signup_user(sign_up_data)
    access_token = create_token(
        {
            "email": sign_up_data.email,
            "user_id": get_user_by_attr("email", sign_up_data.email).user_id,
        },
        JWT_SECRET,
        expire_time=0.01,
    )
    sleep(1)
    with pytest.raises(HTTPException) as exc:
        client.get(
            "/authentication/protected-endpoint",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    validate_http_exception(exc, status.HTTP_401_UNAUTHORIZED, "Token has expired")


def test_protected_endpoint_success(
    sign_up_data: UserSignup,
) -> None:
    """
    Test the successful access to a protected endpoint.
    """
    signup_user(sign_up_data)
    update_user_verification_status(sign_up_data.email)
    access_token = create_token(
        {
            "email": sign_up_data.email,
            "user_id": get_user_by_attr("email", sign_up_data.email).user_id,
        },
        JWT_SECRET,
        expire_time=0.02,
    )
    response = client.get(
        "/authentication/protected-endpoint",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert response.status_code == 200
    assert (
        response.json()["user"]
        == get_user_by_attr("email", sign_up_data.email).to_dict()
    )


def test_refresh_token_invalid_token() -> None:
    """
    Test refreshing the token with an invalid refresh token.
    """
    with pytest.raises(HTTPException) as exc:
        client.post("/authentication/refresh-token?refresh_token=invalid_token")
    validate_http_exception(exc, status.HTTP_401_UNAUTHORIZED, "Invalid token")


def test_refresh_token_expired_token(sign_up_data: UserSignup) -> None:
    """
    Test refreshing the token with an expired refresh token.
    """
    signup_user(sign_up_data)
    refresh_token = create_token(
        {
            "email": sign_up_data.email,
            "user_id": get_user_by_attr("email", sign_up_data.email).user_id,
        },
        JWT_REFRESH_SECRET,
        expire_time=0.01,
    )
    sleep(1)
    with pytest.raises(HTTPException) as exc:
        client.post(f"/authentication/refresh-token?refresh_token={refresh_token}")
    validate_http_exception(exc, status.HTTP_401_UNAUTHORIZED, "Token has expired")


def test_empty_refresh_token() -> None:
    """
    Test refreshing the token with an empty refresh token.
    """
    with pytest.raises(HTTPException) as exc:
        client.post("/authentication/refresh-token?refresh_token=")
    validate_http_exception(exc, status.HTTP_401_UNAUTHORIZED, "Invalid token")


def test_no_refresh_token() -> None:
    """
    Test refreshing the token with no refresh token.
    """
    with pytest.raises(RequestValidationError) as exc:
        client.post("/authentication/refresh-token")
    exc.match("refresh_token")


def test_refresh_token_with_invalid_user(sign_up_data: UserSignup) -> None:
    """
    Test refreshing the token with an invalid user.
    """
    signup_user(sign_up_data)
    refresh_token = create_token(
        {
            "email": "invalid_email",
            "user_id": 1234235,
        },
        JWT_REFRESH_SECRET,
        expire_time=0.02,
    )
    with pytest.raises(HTTPException) as exc:
        client.post(f"/authentication/refresh-token?refresh_token={refresh_token}")
    validate_http_exception(exc, status.HTTP_404_NOT_FOUND, "User not found")


def test_refresh_token_success(sign_up_data: UserSignup) -> None:
    """
    Test the successful refreshing of the token.
    """
    signup_user(sign_up_data)
    refresh_token = create_token(
        {
            "email": sign_up_data.email,
            "user_id": get_user_by_attr("email", sign_up_data.email).user_id,
        },
        JWT_REFRESH_SECRET,
        expire_time=0.02,
    )
    response = client.post(
        f"/authentication/refresh-token?refresh_token={refresh_token}"
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert "refresh_token" in response.json()


#################### RATE LIMITER TESTS ####################
@pytest.mark.asyncio
async def test_rate_limiter_send_verification_code(
    async_client: AsyncClient, sign_up_data: UserSignup
) -> None:
    """
    Test the rate limiter for the send verification code endpoint.
    """
    signup_user(sign_up_data)
    failed_attempts = 0
    for _ in range(5):
        response = await async_client.post(
            "/authentication/send-verification-code",
            params={"email": sign_up_data.email},
        )
        if response.status_code == 200:
            assert (
                response.json()["message"]
                == f"Verification code sent to {sign_up_data.email}. Valid for 10 minutes."
            )
        else:
            assert response.status_code == 429
            assert response.json()["detail"] == "Too Many Requests"
            failed_attempts += 1

    assert failed_attempts == 4


@pytest.mark.asyncio
async def test_rate_limiter_send_password_reset_code(
    async_client: AsyncClient, sign_up_data: UserSignup
) -> None:
    """
    Test the rate limiter for the send password reset code endpoint.
    """
    signup_user(sign_up_data)
    failed_attempts = 0
    for _ in range(5):
        response = await async_client.post(
            "/authentication/send-reset-password-code",
            params={"email": sign_up_data.email},
        )
        if response.status_code == 200:
            assert (
                response.json()["message"]
                == f"Reset password code sent to {sign_up_data.email}. Valid for 10 minutes."
            )
        else:
            assert response.status_code == 429
            assert response.json()["detail"] == "Too Many Requests"
            failed_attempts += 1

    assert failed_attempts == 4
