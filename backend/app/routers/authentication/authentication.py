# pylint: disable=no-value-for-parameter
from pathlib import Path
from typing import cast

import hydra
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi_limiter.depends import RateLimiter

from app import ROOT_DIR
from app.data_layer.database.crud.user_crud import (
    create_or_update_user_verification,
    get_user_by_attr,
)
from app.data_layer.database.models.user_model import User
from app.notification.email.email_provider import EmailProvider
from app.notification.provider import NotificationProvider
from app.routers.authentication.jwt_tokens import access_token_from_refresh_token
from app.schemas.user_model import (
    EmailVerificationRequest,
    UserChangePassword,
    UserResetPassword,
    UserSignIn,
    UserSignup,
)
from app.utils.common import init_from_cfg
from app.utils.common.logger import get_logger
from app.utils.constants import EMAIL, SECONDS, TIMES

from .authenticate import (
    authenticate_user,
    email_identifier,
    get_current_user,
    send_and_save_code,
    signin_user,
    signup_user,
    update_password,
    update_user_verification_status,
    validate_verification_code,
)

# Initialize logging
logger = get_logger(Path(__file__).name)

# Load configuration
with hydra.initialize_config_dir(config_dir=f"{ROOT_DIR}/configs", version_base=None):
    notification_provider_cfg = hydra.compose(config_name="user_verification")

email_notification_provider: EmailProvider = cast(
    EmailProvider,
    init_from_cfg(
        notification_provider_cfg.user_verifier, base_class=NotificationProvider
    ),
)
router = APIRouter(prefix="/authentication", tags=["authentication"])


@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(user: UserSignup) -> dict:
    """
    This endpoint is used to register a new user in the system.

    Parameters:
    -----------
    - **user**:
        The user details to be registered in the system
    """
    logger.info("Signup attempt for user: %s", user.email)
    return signup_user(user)


@router.post("/signin", status_code=status.HTTP_200_OK)
async def signin(user: UserSignIn) -> dict:
    """
    This endpoint is used to authenticate a user in the system.

    Parameters:
    -----------
    - **user**:
        The user details to be authenticated in the system
    """
    logger.info("Signin attempt for user: %s", user.email)
    return signin_user(user.email, user.password)


@router.post(
    "/send-verification-code",
    status_code=status.HTTP_200_OK,
    dependencies=[
        Depends(
            RateLimiter(
                times=TIMES,
                seconds=SECONDS,
                identifier=email_identifier,
            )
        )
    ],
)
async def send_verification_code(email: str, response: Response) -> dict:
    """
    Send a verification code to the user's email.

    Parameters:
    -----------
    - **email** (str): The email to send the verification code to.
    - **response** (Response): The response object to set headers.

    Returns:
    --------
    - JSON response indicating whether the code was sent successfully.
    """

    logger.info(
        "Sending verification code to %s using %s", email, email_notification_provider
    )

    # Fetch the user by email address
    user = get_user_by_attr(EMAIL, email)

    # Check if the user is already verified
    if user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is already verified.",
        )
    send_and_save_code(email, user.username, email_notification_provider)
    response.headers["retry-after"] = str(SECONDS)

    return {"message": f"Verification code sent to {email}. Valid for 10 minutes."}


@router.post("/verify-code", status_code=status.HTTP_200_OK)
async def verify_user(request: EmailVerificationRequest) -> dict:
    """
    Verify the email of the user using the provided verification code.

    Parameters:
    -----------
    - **request**: UserVerificationRequest
        The request object containing the email and the verification code.

    Returns:
    --------
    - JSON response indicating success or failure of the verification.
    """
    logger.info("Verification attempt for %s", request.email)
    user_verification = validate_verification_code(
        request.email, request.verification_code
    )

    # Update verification status
    update_user_verification_status(request.email)

    # Set the expiration time to 0 to indicate that the code has been used
    user_verification.expiration_time = 0
    create_or_update_user_verification(user_verification)

    return {"message": "Email verified successfully"}


@router.post(
    "/send-reset-password-code",
    status_code=status.HTTP_200_OK,
    dependencies=[
        Depends(
            RateLimiter(
                times=TIMES,
                seconds=SECONDS,
                identifier=email_identifier,
            )
        )
    ],
)
async def send_reset_password_code(email: str, response: Response) -> dict:
    """
    Send a reset password code to the user's email.

    Parameters:
    -----------
    - **email** (str): The email to send the reset password code to.
    - **response** (Response): The response object to set headers.

    Returns:
    --------
    - JSON response indicating whether the code was sent successfully.
    """

    logger.info(
        "Sending reset password code to %s using %s", email, email_notification_provider
    )

    # Fetch the user by email address
    user = get_user_by_attr(EMAIL, email)

    send_and_save_code(email, user.username, email_notification_provider)
    response.headers["retry-after"] = str(SECONDS)

    return {"message": f"Reset password code sent to {email}. Valid for 10 minutes."}


@router.post("/reset-password")
async def reset_password(request: UserResetPassword) -> dict:
    """
    Reset the password of the user using the provided verification code.

    Parameters:
    -----------
    - **email** (str): The email of the user.
    - **new_password** (str): The new password to set.
    - **verification_code** (str): The verification code to validate.

    Returns:
    --------
    - JSON response indicating success or failure of the password reset.
    """
    logger.info("Password reset attempt for %s", request.email)
    user_verification = validate_verification_code(
        request.email, request.verification_code
    )
    user = get_user_by_attr(EMAIL, request.email)
    update_password(user.user_id, user.password, request.password)

    user_verification.expiration_time = 0
    create_or_update_user_verification(user_verification)

    return {"message": "Password reset successfully"}


@router.post("/change-password")
async def change_password(request: UserChangePassword) -> dict:
    """
    Change the password of the user.

    Parameters:
    -----------
    - **email** (str): The email of the user.
    - **old_password** (str): The old password of the user.
    - **new_password** (str): The new password to set.

    Returns:
    --------
    - JSON response indicating success or failure of the password change.
    """
    logger.info("Password change attempt for %s", request.email)
    user = authenticate_user(request.email, request.old_password)
    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is not verified. Please verify the email first.",
        )
    update_password(user.user_id, user.password, request.new_password)
    return {"message": "Password changed successfully"}


@router.get(
    "/protected-endpoint",
    response_model=dict,
    responses={
        401: {"description": "Not authenticated"},
        403: {"description": "Not authorized"},
    },
)
def protected_route(current_user: User = Depends(get_current_user)) -> dict:
    """
    Dummy protected route to test the authentication.

    Parameters:
    -----------
    - **current_user** (dict): The current authenticated user.

    Returns:
    --------
    - JSON response indicating the protected route access.
    """
    logger.info("Access to protected route by user: %s", current_user.email)

    return {"message": "This is a protected route", "user": current_user.to_dict()}


@router.post("/refresh-token", status_code=status.HTTP_200_OK)
async def refresh_token(refresh_token: str) -> dict:
    """
    Refresh the access and refresh tokens using a valid refresh token.

    Parameters:
    -----------
    - **refresh_token** (str): The refresh token provided by the client.

    Returns:
    --------
    - JSON response containing the new access and refresh tokens.
    """
    logger.info("Creating new access token using refresh token")
    return access_token_from_refresh_token(refresh_token)
