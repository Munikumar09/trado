# pylint: disable=no-value-for-parameter
import re
from datetime import date

import bcrypt
from dateutil.parser import parse
from dateutil.parser._parser import ParserError
from fastapi import HTTPException, status

from app.data_layer.database.crud.user_crud import is_attr_data_in_db
from app.data_layer.database.models.user_model import User
from app.schemas.user_model import UserSignup
from app.utils.common.exceptions.authentication import UserSignupError


def validate_email(email: str):
    """
    This function validates the email format. It allows only email addresses with
    the domain 'gmail.com'

    Parameters:
    -----------
    email: ``str``
        The email address to be validated
    """
    email_regex = r"^[\w\.-]+@gmail\.com$"

    if re.match(email_regex, email) is None:
        raise UserSignupError("Invalid email format")


def validate_phone_number(phone_number: str) -> None:
    """
    This function validates the phone number format. It allows only phone numbers
    with 10 digits.

    Parameters:
    -----------
    phone_number: ``str``
        The phone number to be validated
    """
    phone_number_regex = r"\d{10}$"

    if re.fullmatch(phone_number_regex, phone_number) is None:
        raise UserSignupError("Invalid phone number format")


def get_hash_password(password: str) -> str:
    """
    It hashes the password using bcrypt and returns the hashed password.

    Parameters:
    -----------
    password: ``str``
        The password to be hashed

    Returns:
    --------
    ``str``
        The hashed password
    """
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def validate_password(password: str):
    """
    Validates the password format. The password must be at least 8 characters long
    and include an uppercase letter, lowercase letter, digit, and special character.

    Parameters:
    -----------
    password: ``str``
        The password to be validated
    """

    if not (
        len(password) >= 8
        and re.search(r"[A-Z]", password) is not None
        and re.search(r"[a-z]", password) is not None
        and re.search(r"\d+", password) is not None
        and re.search(r"[!@#$%^&*(),.?\":{}|<>]", password) is not None
    ):

        raise UserSignupError(
            "Password must be at least 8 characters long and include an uppercase letter, "
            "lowercase letter, digit, and special character",
        )


def verify_password(password: str, hash_password: str, raise_exception=True) -> bool:
    """
    Verifies the password by comparing the hashed password with the password.

    Parameters:
    -----------
    password: ``str``
        The password to be verified
    hash_password: ``str``
        The hashed password to be compared with the password
    raise_exception: ``bool``
        If True, it raises an exception if the passwords do not match

    Returns:
    --------
    ``bool``
        True if the passwords match, otherwise False
    """
    if not bcrypt.checkpw(password.encode("utf-8"), hash_password.encode("utf-8")):
        if raise_exception:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Passwords do not match")
        return False
    return True


def validate_date_of_birth(date_of_birth: str) -> None:
    """
    Validates the date of birth format. The date of birth cannot be in the future.

    Parameters:
    -----------
    date_of_birth: ``str``
        The date of birth to be validated
    """
    try:
        dob = parse(date_of_birth, dayfirst=True).date()

        if dob >= date.today():
            raise UserSignupError("Date of birth cannot be in the future")

    except ParserError as exc:
        raise UserSignupError(
            f"Invalid date of birth {date_of_birth}. Please provide a valid date."
        ) from exc


def validate_user_exists(user: UserSignup) -> str | None:
    """
    Checks if the user already exists in the database. It checks the email
    and phone number fields to see if the user already exists.

    Parameters:
    -----------
    user: ``UserSignup``
        The user details to be checked

    Returns:
    --------
    ``str | None``
        An error message if the user already exists, otherwise None
    """
    fields_to_check = {
        "email": user.email,
        "phone_number": user.phone_number,
    }
    response = is_attr_data_in_db(User, fields_to_check)

    return response


def validate_user_data(user: UserSignup) -> None:
    """
    Validates the user data. If any of the user data is invalid, it raises an exception.

    Parameters:
    -----------
    user: ``UserSignup``
        The user details to be validated
    """
    if user.password != user.confirm_password:
        raise UserSignupError("Passwords do not match")

    validate_email(user.email)
    validate_phone_number(user.phone_number)
    validate_password(user.password)
    validate_date_of_birth(user.date_of_birth)
