# pylint: disable=no-value-for-parameter

import random
from datetime import datetime, timedelta, timezone

from dateutil.parser import parse
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from snowflake import SnowflakeGenerator

from app.data_layer.database.crud.user_crud import (
    create_or_update_user_verification,
    create_user,
    get_user,
    get_user_by_attr,
    get_user_verification,
    update_user,
)
from app.data_layer.database.models.user_model import Gender, User, UserVerification
from app.notification.email.email_provider import EmailProvider
from app.schemas.user_model import UserSignup
from app.utils.common.exceptions.authentication import UserSignupError
from app.utils.constants import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    EMAIL,
    JWT_REFRESH_SECRET,
    JWT_SECRET,
    MACHINE_ID,
    REFRESH_TOKEN_EXPIRE_MINUTES,
    USER_ID,
)

from .jwt_tokens import create_token, decode_token
from .user_validation import (
    get_hash_password,
    validate_password,
    validate_user_data,
    validate_user_exists,
    verify_password,
)

snowflake_generator = SnowflakeGenerator(MACHINE_ID)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/authentication/signin")


async def email_identifier(request: Request) -> str:
    """
    Identifier for rate limiting based on email and path.

    Parameters:
    -----------
    request: ``Request``
        The request object from the client

    Returns:
    --------
    ``str``
        The identifier for rate limiting
    """
    # Check query params first
    email = request.query_params.get("email")
    if email is None:
        # If not in query params, check request body
        try:
            body = await request.json()
            email = body.get("email")
        except Exception:
            email = None

    if email is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is required to identify the user",
        )

    return email + ":" + request.scope["path"]


def get_snowflake_id() -> int:
    """
    Generate a unique snowflake ID using the snowflake algorithm.

    Returns:
    --------
    ``int``
        The unique snowflake ID
    """
    return next(snowflake_generator)


def signup_user(user: UserSignup) -> dict["str", "str"]:
    """
    Validates the user data and creates a new user in the system if the user does not
    already exist. If the user already exists, it returns an error message.

    Parameters:
    -----------
    user: ``UserSignup``
        The user details to be registered in the system

    Returns:
    --------
    ``dict``
        A message indicating if the user was created successfully or an error message
    """
    # Check whether the user is already exists with the given details
    if reason := validate_user_exists(user):
        raise UserSignupError(reason)

    validate_user_data(user)

    user_model = User(
        **user.dict(exclude={"password", "date_of_birth", "gender"}),
        password=get_hash_password(user.password),
        user_id=get_snowflake_id(),
        date_of_birth=parse(user.date_of_birth, dayfirst=True).date(),
        gender=Gender.get_gender_enum(
            gender=user.gender, raise_exception=UserSignupError
        ),
    )

    create_user(user_model)

    if not get_user_by_attr(EMAIL, user.email):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User not created successfully. Please try again later",
        )

    return {
        "message": "User created successfully. Please verify your email to activate your account"
    }


def authenticate_user(email: str, password: str) -> User:
    """
    Authenticate the user with the given email and password. If the user does not exist,
    or the password is incorrect, it raises an error message. If the user is not verified,
    it raises an error message. Otherwise, it returns the user details.

    Parameters:
    -----------
    email: ``str``
        The email address of the user
    password: ``str``
        The password of the user

    Returns:
    --------
    ``User``
        The user details retrieved from the database
    """
    user = get_user_by_attr(EMAIL, email)
    verify_password(password, user.password)

    return user


def signin_user(email: str, password: str) -> dict[str, str]:
    """
    Sign in the user with the given email and password. If email or password is incorrect,
    or the user is not verified, it raises an error message. Otherwise, it generates an
    access token and a refresh token and returns them to the user.

    Parameters:
    -----------
    email: ``str``
        The email address of the user
    password: ``str``
        The password of the user

    Returns:
    --------
    ``dict[str, str]``
        Dictionary containing the message, access token, and refresh token
    """
    user = authenticate_user(email, password)

    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User is not verified",
        )

    token_data = {USER_ID: user.user_id, EMAIL: user.email}
    access_token = create_token(
        token_data,
        JWT_SECRET,
        ACCESS_TOKEN_EXPIRE_MINUTES,
    )
    refresh_token = create_token(
        token_data,
        JWT_REFRESH_SECRET,
        REFRESH_TOKEN_EXPIRE_MINUTES,
    )

    return {
        "message": "Login successful",
        "access_token": access_token,
        "refresh_token": refresh_token,
    }


def update_user_verification_status(
    user_email: str, is_verified: bool = True
) -> dict[str, str]:
    """
    Update the user verification status in the database. If the user does not exist,
    it raises an error message. Otherwise, it updates the user verification status
    and returns a success message.

    Parameters:
    -----------
    user_email: ``str``
        The email address of the user
    is_verified: ``bool`` ( default = True )
        The verification status of the user

    Returns:
    --------
    ``dict[str, str]``
        A message indicating if the user was verified successfully or an error message
    """
    user = get_user_by_attr(EMAIL, user_email)
    user.is_verified = is_verified
    update_user(user.user_id, user.model_dump())

    return {"message": "User verified successfully"}


def generate_verification_code(length: int = 6) -> str:
    """
    Generate a random verification code consisting of numbers only with the given length.

    Parameters:
    -----------
    length: ``int``
        The length of the verification code to be generated

    Returns:
    --------
    ``str``
        The generated verification code
    """
    return "".join([str(random.randint(0, 9)) for _ in range(length)])


def send_and_save_code(email: str, user_name: str, provider: EmailProvider):
    """
    Send a verification code to the user's email and save the verification code in the
    database. If the user does not exist, it raises an error message.

    Parameters:
    -----------
    email: ``str``
        The email address of the user
    user_name: ``str``
        The name of the user
    provider: ``EmailProvider``
        The email provider to send the verification code
    """
    # Generate and save the verification code
    verification_code = generate_verification_code()
    expiration_time = int(
        (datetime.now(timezone.utc) + timedelta(minutes=10)).timestamp()
    )

    create_or_update_user_verification(
        UserVerification(
            email=email,
            verification_code=verification_code,
            expiration_time=expiration_time,
        )
    )

    # Send the notification
    provider.send_notification(
        code=verification_code,
        recipient_email=email,
        recipient_name=user_name,
    )


def validate_verification_code(email: str, verification_code: str) -> UserVerification:
    """
    Validate the verification code sent by the user. If the verification code does not
    match the one stored in the database, or if the verification code has expired, it
    raises an error message.

    Parameters:
    -----------
    email: ``str``
        The email address of the user
    verification_code: ``str``
        The verification code sent by the user

    Raises:
    -------
    ``HTTPException``
        If the verification code does not match or has expired

    Returns:
    -------
    ``UserVerification``
        The user verification object
    """
    # Fetch user verification details
    user_verification = get_user_verification(email)

    if user_verification is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User does not exist with email: {email}",
        )

    # Check if verification code matches
    if user_verification.verification_code != verification_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid verification code"
        )

    # Check if the verification code has expired
    if user_verification.expiration_time < int(datetime.now(timezone.utc).timestamp()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification code has expired. Try again",
        )

    return user_verification


def update_password(user_id: int, old_password: str, password: str):
    """
    Update the password of the user with the given user ID. If the old password does not
    match the password stored in the database or the new password is the same as the old
    password, it raises an error message. Otherwise, it updates the password in the database.

    Parameters:
    -----------
    user_id: ``int``
        The user ID of the user
    old_password: ``str``
        The old hashed password of the user
    password: ``str``
        The new password of the user
    """
    validate_password(password)

    if verify_password(password, old_password, raise_exception=False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password cannot be the same as the old password",
        )

    update_user(user_id, {"password": get_hash_password(password)})


def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Retrieve the current user from the database using the access token. If the token
    is invalid or expired, it returns an error message. If the token is valid, it
    retrieves the user details from the database and returns the user.

    Parameters:
    -----------
    token: ``str``
        The access token to retrieve the user details

    Returns:
    --------
    ``User``
        The user details retrieved from the database
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token is missing"
        )

    decoded_data = decode_token(token, JWT_SECRET)
    if USER_ID not in decoded_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )

    return get_user(decoded_data[USER_ID])
