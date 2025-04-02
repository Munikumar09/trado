# pylint: disable=no-value-for-parameter
import datetime
from pathlib import Path
from typing import Dict, Type

from fastapi import HTTPException, status
from sqlmodel import Session, SQLModel, select

from app.data_layer.database.db_connections.postgresql import with_session
from app.data_layer.database.models.user_model import User, UserVerification
from app.utils.common.logger import get_logger

logger = get_logger(Path(__file__).name)


#### User CRUD operations ####
@with_session
def is_attr_data_in_db(
    model: Type[SQLModel], att_values: Dict[str, str], session: Session
) -> str | None:
    """
    Checks if any of the specified fields have values that already exist in the database.

    For example, if the `att_values` dictionary contains the key-value pair
    {"email": "example@gmail.com"}, this function will check if there is any
    user in the database with the email "example@gmail.com" and return a message
    indicating that the email already exists.

    Parameters:
    ----------
    model: ``Type[SQLModel]``
        The model to check for the existence of the field values
    att_values: ``Dict[str, str]``
        A dictionary containing the field names and values to check for existence
    session: ``Session``
        The session to use for the database operations

    Returns:
    -------
    ``str | None``
        A message indicating that the field already exists if found, otherwise None
    """
    existing_attr: str | None = None

    for attr_name, attr_value in att_values.items():

        if not hasattr(model, attr_name):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"Invalid field: {attr_name}"
            )
        statement = select(model).where(getattr(model, attr_name) == attr_value)

        if session.exec(statement).first():
            existing_attr = f"{attr_name} already exists"
            break

    return existing_attr


@with_session
def get_user_by_attr(attr_name: str, attr_value: str, session: Session) -> User:
    """
    Retrieves a user from the database using the specified attribute name and value.
    Example: get_user_by_attr("email", "example@gmail.com")

    Parameters:
    ----------
    attr_name: ``str``
        The name of the attribute used to query the database
    attr_value: ``str``
        Value of the attribute to match against the database
    session: ``Session``
        Session object to interact with the database

    Raises:
    -------
    ``HTTPException``
        If the given attribute name is invalid or the user is not found

    Returns:
    -------
    ``User``
        The user object if found, otherwise None
    """
    if not hasattr(User, attr_name):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid field: {attr_name}")

    statement = select(User).where(getattr(User, attr_name) == attr_value)
    result = session.exec(statement).first()

    if not result:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"User does not exist with given {attr_name}: {attr_value}",
        )

    return result


@with_session
def create_user(user: User, session: Session):
    """
    Adds a new user to the database.

    Parameters:
    ----------
    user: ``User``
        The user object to add to the database
    session: ``Session ``
        Session object to interact with the database
    """
    try:
        session.add(user)
        session.commit()
        session.refresh(user)
    except Exception as e:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"Error while creating user: {e}"
        ) from e


@with_session
def get_user(user_id: int, session: Session) -> User:
    """
    Retrieve a user from the database using the user_id.

    Parameters:
    ----------
    user_id: ``int``
        The `user_id` of the user to retrieve
    session: ``Session``
        Session object to interact with the database

    Raises:
    -------
    ``HTTPException``
        If the user with give user_id not exists

    Returns:
    -------
    ``User``
        The user object if found, otherwise None
    """
    statement = select(User).where(User.user_id == user_id)
    user = session.exec(statement).first()

    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")

    return user


@with_session
def update_user(user_id: int, user_data: dict, session: Session) -> None:
    """
    Update the fields of a user in the database using the `user_id` if the user
    present in the database. The `user_data` dictionary should contain the fields
    to update and their new values.

    Note:
    -----

    Validation of the fields is not done in this function. Please ensure that
    the fields are valid before calling this function.

    Parameters:
    ----------
    user_id: ``int``
        The `user_id` of the user to update
    user_data: ``dict``
        A dictionary containing the fields to update and their new values
    session: ``Session | None``, ( default = None )
        Session object to interact with the database

    >>> Example:
        update_user(1234, {"first_name": "John", "last_name": "Doe"})
    """
    user = get_user(user_id, session=session)

    for key, value in user_data.items():
        if not hasattr(user, key):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid field: {key}")

        setattr(user, key, value)

    session.add(user)
    session.commit()
    session.refresh(user)


@with_session
def delete_user(user_id: int, session: Session) -> None:
    """
    Delete a user from the database using the `user_id` if the user is present.

    Parameters:
    ----------
    user_id: ``int``
        The `user_id` of the user to delete
    session: ``Session``
        Session object to interact with the database
    """
    user = get_user(user_id, session=session)
    session.delete(user)
    session.commit()


#### UserVerification CRUD operations ####
@with_session
def get_user_verification(email: str, session: Session) -> UserVerification | None:
    """
    Retrieve a user verification object from the database using the email.

    Parameters:
    ----------
    email: ``str``
        The email of the user to retrieve the verification object
    session: ``Session``
        Session object to interact with the database

    Returns:
    -------
    ``UserVerification | None``
        The user verification object if found, otherwise None
    """
    statement = select(UserVerification).where(UserVerification.email == email)
    result = session.exec(statement).first()

    return result


@with_session
def create_or_update_user_verification(
    user_verification: UserVerification, session: Session
):
    """
    Create a new user verification object in the database if it does not exist, otherwise
    update the existing object. If the user verification object already exists, the function
    will update the verification code, expiration time, re-verified datetime, and verification
    medium.

    Parameters:
    ----------
    user_verification: ``UserVerification``
        The user verification object to add or update in the database
    session: ``Session``
        Session object to interact with the database
    """

    existing_user_verification = get_user_verification(
        user_verification.email, session=session
    )

    if not existing_user_verification:
        existing_user_verification = user_verification
    else:
        existing_user_verification.verification_code = (
            user_verification.verification_code
        )
        existing_user_verification.expiration_time = user_verification.expiration_time
        existing_user_verification.reverified_datetime = (
            user_verification.reverified_datetime or datetime.datetime.now()
        )

    session.add(existing_user_verification)
    session.commit()
    session.refresh(existing_user_verification)
