# pylint: disable=no-value-for-parameter
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import HTTPException, status

from app.data_layer.database.crud.user_crud import get_user
from app.utils.constants import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    EMAIL,
    JWT_HASHING_ALGO,
    JWT_REFRESH_SECRET,
    JWT_SECRET,
    USER_ID,
)


def create_token(data: dict, secret: str, expire_time: float) -> str:
    """
    Creates a JWT token with the given data and expiration time.

    Parameters:
    -----------
    data: ``dict``
        The data to be encoded in the token
    secret: ``str``
        The secret key used to encode the token
    expire_time: ``int``
        The expiration time of the token in minutes

    Returns:
    --------
    ``str``
        The JWT token
    """
    to_encode = data.copy()

    # Using `exp` as token expiry time, if `exp` is present in the payload data then
    # pyJwt will verify the token expiry automatically
    to_encode.update(
        {"exp": datetime.now(timezone.utc) + timedelta(minutes=expire_time)}
    )

    return jwt.encode(to_encode, secret, algorithm=JWT_HASHING_ALGO)


def decode_token(token: str, secret: str) -> dict[str, str]:
    """
    Decodes the given token using the secret key and returns the decoded data.
    The decoded data contain the payload used for creating the token.

    Parameters:
    -----------
    token: ``str``
        The token to be decoded
    secret: ``str``
        The secret key used to decode the token

    Returns:
    --------
    ``dict[str, str]``
        The decoded data from the token
    """
    try:
        return jwt.decode(token, secret, algorithms=[JWT_HASHING_ALGO])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token") from exc


def access_token_from_refresh_token(refresh_token: str) -> dict[str, str]:
    """
    Create the access token using the refresh token. If the refresh token is invalid
    or expired, it returns an error message. Otherwise, it generates a new access token
    and returns it.

    Parameters:
    -----------
    refresh_token: ``str``
        The refresh token to generate a new access token

    Returns:
    --------
    ``dict[str, str]``
        A dictionary containing the new access token and the refresh token
    """
    decoded_data = decode_token(refresh_token, JWT_REFRESH_SECRET)

    get_user(decoded_data[USER_ID])

    access_token = create_token(
        {USER_ID: decoded_data[USER_ID], EMAIL: decoded_data[EMAIL]},
        JWT_SECRET,
        ACCESS_TOKEN_EXPIRE_MINUTES,
    )

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
    }
