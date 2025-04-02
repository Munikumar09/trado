from pydantic import BaseModel


class UserSignup(BaseModel):
    """
    UserSignup schema for user registration
    """

    username: str
    email: str
    password: str
    confirm_password: str
    date_of_birth: str
    phone_number: str
    gender: str


class UserSignIn(BaseModel):
    """
    UserSignIn schema for user authentication
    """

    email: str
    password: str


class EmailVerificationRequest(BaseModel):
    """
    UserVerification schema for user verification
    """

    verification_code: str
    email: str


class UserResetPassword(BaseModel):
    """
    UserResetPassword schema for user password reset
    """

    email: str
    password: str
    verification_code: str


class UserChangePassword(BaseModel):
    """
    UserChangePassword schema for user password change
    """

    email: str
    old_password: str
    new_password: str
