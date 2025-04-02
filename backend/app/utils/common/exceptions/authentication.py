from fastapi import HTTPException, status


class UserSignupError(HTTPException):
    """
    Exception class for user signup errors. This class is used to raise exceptions
    when there is an error during user signup. This exception raises when there is
    invalid user data.

    Attributes:
    -----------
    message: ``str``
        The error message to be displayed to the user
    """

    def __init__(self, message: str):
        super().__init__(status.HTTP_400_BAD_REQUEST, message)
