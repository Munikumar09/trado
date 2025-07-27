"""
This module provides utility functions for interacting with the SmartAPI platform, 
including establishing HTTP(S) connections to SmartAPI endpoints with appropriate 
authentication and headers. It is designed to centralize and standardize the process
of making API requests to SmartAPI, leveraging credential management and request type 
abstractions.
"""

import http.client
from http.client import HTTPConnection

from omegaconf import DictConfig

from app.utils.common.types.reques_types import RequestType
from app.utils.credentials.smartapi_credential_manager import SmartapiCredentialManager


def get_endpoint_connection(
    payload: str | dict, request_method_type: RequestType, url: str
) -> HTTPConnection:
    """
    Get the HTTP connection object for making API requests to a specific endpoint
    to the SmartAPI with the given payload.

    Parameters:
    -----------
    payload: ``str | dict``
        The payload to be sent in the API request
    method_type: ``RequestType``
        The request method type like GET, POST, PUT, DELETE from the RequestType enum
        eg: RequestType.GET, RequestType.POST
    url: ``str``
        The URL of the endpoint to connect to

    Returns:
    --------
    ``HTTPConnection``
        The HTTP connection object for the given endpoint
    """
    smartapi_credential_manager = SmartapiCredentialManager.from_cfg(
        DictConfig({"connection_num": 1})
    )
    connection = http.client.HTTPSConnection("apiconnect.angelbroking.com")
    headers = smartapi_credential_manager.get_headers()
    connection.request(request_method_type.value, url, body=payload, headers=headers)

    return connection
