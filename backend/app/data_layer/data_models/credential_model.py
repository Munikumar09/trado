from dataclasses import dataclass


@dataclass
class CredentialInput:
    """
    Base class for input credentials.

    This dataclass serves as a base for all credential input classes,
    encapsulating the common attributes required for authentication.
    """


@dataclass
class CredentialOutput:
    """
    Base class for output credentials.

    This dataclass serves as a base for all credential output classes,
    encapsulating the common attributes returned after authentication.
    """


@dataclass
class SmartAPICredentialInput(CredentialInput):
    """
    Input credentials required for SmartAPI authentication.

    This dataclass encapsulates the credentials needed to authenticate
    with the SmartAPI service.

    Attributes
    ----------
    api_key: ``str``
        The API key provided by SmartAPI for authentication.
    client_id: ``str``
        The unique client identifier assigned by SmartAPI.
    pwd: ``str``
        The user's password for SmartAPI account access.
    token: ``str``
        The authentication token required for API requests.
    connection_num: ``int``
        The current connection number for the SmartAPI service.
    """

    api_key: str
    client_id: str
    pwd: str
    token: str
    connection_num: int


@dataclass
class SmartAPICredentialOutput(CredentialOutput):
    """
    Output credentials received from SmartAPI authentication.

    This dataclass contains the authentication tokens and user information
    returned after successful authentication with the SmartAPI service.

    Attributes
    ----------
    access_token: ``str``
        The access token used for authenticated API requests.
    refresh_token: ``str``
        The refresh token used to obtain new access tokens.
    feed_token: ``str``
        The feed token used for real-time data streaming.
    user_id: ``str``
        The unique identifier for the authenticated user.
    """

    access_token: str
    refresh_token: str
    feed_token: str
    user_id: str

    def to_dict(self) -> dict:
        """
        Convert the SmartAPICredentialOutput to a dictionary.

        Returns
        -------
        ``dict``
            A dictionary representation of the SmartAPICredentialOutput
        """
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "feed_token": self.feed_token,
            "user_id": self.user_id,
        }


@dataclass
class UplinkCredentialInput(CredentialInput):
    """
    Input credentials required for Uplink authentication.

    This dataclass encapsulates the credentials needed to authenticate
    with the Uplink service, including two-factor authentication.

    Attributes
    ----------
    api_key: ``str``
        The API key provided by Uplink for authentication.
    secret_key: ``str``
        The secret key used in conjunction with the API key.
    totp_key: ``str``
        The Time-based One-Time Password key for two-factor authentication.
    mobile_no: ``str``
        The mobile number associated with the Uplink account.
    pin: ``str``
        The personal identification number for account access.
    """

    api_key: str
    secret_key: str
    totp_key: str
    mobile_no: str
    pin: str
    connection_num: int


@dataclass
class UplinkCredentialOutput(CredentialOutput):
    """
    Output credentials received from Uplink authentication.

    This dataclass contains the authentication token returned after
    successful authentication with the Uplink service.

    Attributes
    ----------
    access_token: ``str``
        The access token used for authenticated API requests to Uplink.
    """

    access_token: str

    def to_dict(self) -> dict:
        """
        Convert the UplinkCredentialOutput to a dictionary.

        Returns
        -------
        ``dict``
            A dictionary representation of the UplinkCredentialOutput
        """
        return {
            "access_token": self.access_token,
        }
