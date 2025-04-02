from datetime import date, timedelta

import bcrypt
import pytest
from fastapi import HTTPException

from app.routers.authentication.user_validation import (
    get_hash_password,
    validate_date_of_birth,
    validate_email,
    validate_password,
    validate_phone_number,
    verify_password,
)


# Test: 1
@pytest.mark.parametrize(
    "email,is_valid",
    [
        ("test@gmail.com", True),
        ("test@yahoo.com", False),
        ("invalid.email", False),
        ("test@.com", False),
        ("test@domain", False),
        ("@domain.com", False),
        ("", False),
        ("test@subdomain.domain.com", False),
    ],
)
def test_validate_email(email: str, is_valid: bool):
    """
    Test validate_email function
    """
    if is_valid:
        assert validate_email(email) is None
    else:
        with pytest.raises(HTTPException):
            validate_email(email)


# Test: 2
def test_validate_phone_number():
    """
    Test validate_phone_number function
    """
    # Test: 2.1 ( Valid phone number )
    assert validate_phone_number("1234567890") is None

    # Test: 2.2 ( Invalid phone number with less than 10 digits )
    with pytest.raises(HTTPException):
        validate_phone_number("12345")

    # Test: 2.3 ( Invalid phone number with more than 10 digits )
    with pytest.raises(HTTPException):
        validate_phone_number("12345678901")

    # Test: 2.4 ( Invalid phone number format )
    with pytest.raises(HTTPException):
        validate_phone_number("123-456-8901")

    with pytest.raises(HTTPException):
        validate_phone_number("-123456789a")


# Test: 3
def test_get_hash_password():
    """
    Test get_hash_password function
    """
    password = "Password1!"
    hashed_password = get_hash_password(password)

    assert bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))


# Test: 4
def test_verify_password():
    """
    Test verify_password function
    """
    password = "Password1!"
    hashed_password = get_hash_password(password)

    # Test: 4.1 ( Valid password )
    assert verify_password(password, hashed_password) is True

    # Test: 4.2 ( Invalid password )
    with pytest.raises(HTTPException):
        verify_password("wrongpassword", hashed_password)

    # Test: 4.3 ( Invalid password with raise_exception=False )
    assert (
        verify_password("wrongpassword", hashed_password, raise_exception=False)
        is False
    )


# Test: 5
@pytest.mark.parametrize(
    "password,is_valid",
    [
        ("Password1!", True),  # Test: 5.1 ( Valid password )
        # Test: 5.2 ( Invalid password with no special character )
        ("Password1", False),
        # Test: 5.3 ( Invalid password with no uppercase character )
        ("password1!", False),
        # Test: 5.4 ( Invalid password with no lowercase character )
        ("PASSWORD1!", False),
        # Test: 5.5 ( Invalid password with no number )
        ("Password!", False),
        # Test: 5.6 ( Invalid password with less than 8 characters )
        ("Pasw1r@", False),
        ("", False),  # Test: 5.7 ( Empty password )
    ],
)
def test_validate_password(password: str, is_valid: bool):
    """
    Test validate_password function
    """
    if is_valid:
        assert validate_password(password) is None
    else:
        with pytest.raises(HTTPException):
            validate_password(password)


# Test: 6
def test_validate_date_of_birth():
    """
    Test validate_date_of_birth function
    """
    # Test: 6.1 ( Valid date of birth )
    assert validate_date_of_birth("01/01/2000") is None
    assert validate_date_of_birth("01-01-2000") is None

    # Test: 6.2 ( Test with future date )
    with pytest.raises(HTTPException) as exc:
        validate_date_of_birth((date.today() + timedelta(days=1)).strftime("%d/%m/%Y"))
    assert exc.value.detail == "Date of birth cannot be in the future"

    # Test: 6.3 ( Test with invalid date format )
    with pytest.raises(HTTPException) as exc:
        validate_date_of_birth("00-01-2000")
    assert (
        exc.value.detail
        == "Invalid date of birth 00-01-2000. Please provide a valid date."
    )
