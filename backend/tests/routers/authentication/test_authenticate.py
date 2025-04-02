""" 
Test cases for the authenticate.py module
"""

import pytest
from fastapi import HTTPException, status
from pytest_mock import MockType

from app.data_layer.database.models.user_model import User
from app.routers.authentication.authenticate import (
    UserSignupError,
    create_token,
    generate_verification_code,
    get_current_user,
    signin_user,
    signup_user,
    update_password,
    update_user_verification_status,
)
from app.schemas.user_model import UserSignup
from app.utils.constants import JWT_SECRET


# Test: 1
def test_signup_user(mock_session: MockType, sign_up_data: UserSignup):
    """
    Test signup_user function
    """
    # Test: 1.1 ( Successful user signup )
    mock_session.first.side_effect = [None, None, sign_up_data]
    response = signup_user(sign_up_data)

    assert (
        response["message"]
        == "User created successfully. Please verify your email to activate your account"
    )

    # Test: 1.2 ( User already exists )
    mock_session.first.side_effect = sign_up_data
    with pytest.raises(UserSignupError) as signup_error:
        signup_user(sign_up_data)

    assert signup_error.value.detail == "email already exists"

    sign_up_data.confirm_password = "wrong_password"
    with pytest.raises(UserSignupError) as signup_error:
        signup_user(sign_up_data)


# Test: 2
def test_signin_user(mock_session: MockType, test_user: User, sign_up_data: UserSignup):
    """
    Test signin_user function
    """
    # Test: 2.1 ( User not found )
    mock_session.first.return_value = None
    with pytest.raises(HTTPException) as http_exe:
        signin_user(sign_up_data.email, sign_up_data.password)
    assert http_exe.value.status_code == status.HTTP_404_NOT_FOUND
    assert (
        http_exe.value.detail == "User does not exist with given email: test@gmail.com"
    )

    # Test: 2.2 ( Incorrect password )
    mock_session.first.return_value = test_user
    with pytest.raises(HTTPException) as http_exe:
        signin_user(sign_up_data.email, "test_password")

    assert http_exe.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert http_exe.value.detail == "Passwords do not match"

    # Test: 2.3 ( User not verified )
    with pytest.raises(HTTPException) as http_exe:
        signin_user(sign_up_data.email, sign_up_data.password)

    assert http_exe.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert http_exe.value.detail == "User is not verified"

    # Test: 2.4 ( Successful user signin )
    test_user.is_verified = True

    response = signin_user(sign_up_data.email, sign_up_data.password)
    assert response.keys() == {"message", "access_token", "refresh_token"}
    assert response["message"] == "Login successful"


# Test: 3
def test_update_user_verification_status(mock_session: MockType, test_user: User):
    """
    Test update_user_verification_status function
    """
    # Test: 3.1 ( User not found )
    mock_session.first.return_value = None

    with pytest.raises(HTTPException) as http_exe:
        update_user_verification_status(test_user.email)

    assert http_exe.value.status_code == status.HTTP_404_NOT_FOUND
    assert (
        http_exe.value.detail == "User does not exist with given email: test@gmail.com"
    )
    assert test_user.is_verified is False

    # Test: 3.2 ( Verify user )
    mock_session.first.return_value = test_user
    response = update_user_verification_status(user_email=test_user.email)
    assert response["message"] == "User verified successfully"

    assert test_user.is_verified is True


# Test: 4
def test_generate_verification_code():
    """
    Test generate_verification_code function
    """
    # Test: 4.1 ( Generate verification code of default length 6 )
    code = generate_verification_code()
    assert len(code) == 6
    assert code.isdigit()

    # Test: 4.2 ( Generate verification code of length 10 )
    code = generate_verification_code(10)
    assert len(code) == 10
    assert code.isdigit()


# Test: 5
def test_get_current_user(
    mock_session: MockType, token_data: dict[str, str], test_user: User
):
    """
    Test get_current_user function
    """
    token = create_token(token_data, JWT_SECRET, 15)
    mock_session.first.return_value = test_user
    user = get_current_user(token)
    assert user.model_dump() == test_user.model_dump()


# Test: 6
def test_update_password(
    mock_session: MockType,
    test_user: User,
):
    """
    Test update_password function
    """
    new_password = "NewPassword1!"

    # Test: 6.1 ( User not found )
    mock_session.first.return_value = None
    with pytest.raises(HTTPException) as http_exe:
        update_password(test_user.user_id, test_user.password, new_password)
    assert http_exe.value.status_code == status.HTTP_404_NOT_FOUND
    assert http_exe.value.detail == "User not found"

    # Test: 6.2 ( Password validation failure )
    invalid_password = "short"
    mock_session.first.return_value = test_user
    with pytest.raises(HTTPException) as http_exe:
        update_password(test_user.user_id, test_user.password, invalid_password)
    assert http_exe.value.status_code == status.HTTP_400_BAD_REQUEST
    assert (
        http_exe.value.detail
        == "Password must be at least 8 characters long and include an uppercase letter, "
        "lowercase letter, digit, and special character"
    )

    # Test: 6.3 ( Password same as previous password )
    mock_session.first.return_value = test_user

    with pytest.raises(HTTPException) as http_exe:
        update_password(test_user.user_id, test_user.password, "Password1!")
    assert http_exe.value.status_code == status.HTTP_400_BAD_REQUEST
    assert (
        http_exe.value.detail == "New password cannot be the same as the old password"
    )
