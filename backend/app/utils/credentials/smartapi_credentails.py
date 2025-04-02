# pylint: disable=too-many-arguments
""" 
This module contains the SmartapiCredentials class to store the credentials
required to authenticate the SmartAPI connection. 
"""

from app.utils.credentials.credentials import Credentials
from app.utils.fetch_data import get_required_env_var


@Credentials.register("smartapi")
class SmartapiCredentials(Credentials):
    """
    Credentials class to store the credentials required to authenticate the SmartAPI connection.

    Attributes:
    -----------
    api_key: ``str``
        The API key that is generated from the SmartAPI website
    client_id: ``str``
        The client id is the Angel Broking client id
    pwd: ``str``
        The password is the Angel Broking login password or pin
    token: ``str``
        The token is the client secret token generated from the SmartAPI website
    correlation_id: ``str``
        The correlation id is the unique id to identify the request
    """

    def __init__(self, api_key: str, client_id: str, pwd: str, token: str) -> None:

        self.api_key = api_key
        self.client_id = client_id
        self.pwd = pwd
        self.token = token

    @classmethod
    def get_credentials(cls) -> "SmartapiCredentials":
        """
        Create a Credentials object from the credentials file.

        Returns:
        --------
        ``SmartapiCredentials``
            The credentials object with the API key, client id, password, token and correlation id
        """
        api_key = get_required_env_var("SMARTAPI_API_KEY")
        client_id = get_required_env_var("SMARTAPI_CLIENT_ID")
        pwd = get_required_env_var("SMARTAPI_PWD")
        token = get_required_env_var("SMARTAPI_TOKEN")

        return SmartapiCredentials(api_key, client_id, pwd, token)
