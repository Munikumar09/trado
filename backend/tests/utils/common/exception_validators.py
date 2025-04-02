from typing import Any

from fastapi import HTTPException
from fastapi.testclient import TestClient


def validate_exception(
    endpoint_url: str, expected_error: dict[str, Any], client: TestClient
):
    """
    Test function to validate exception.

    This function checks if the expected exception is valid or not
    for the request with given endpoint_url.

    Parameters:
    -----------
    endpoint_url: ``str``
        URL to request data from historical_stock_data endpoint.
    expected_error: ``dict[str, Any]``
        Expected exception for the request with given endpoint_url.
    """
    try:
        # Make a GET request to the endpoint URL
        client.get(endpoint_url)
    except HTTPException as http_exc:
        # Check if the status code of the exception matches the expected error status code
        assert http_exc.status_code == expected_error["status_code"]

        # Check if the detail message of the exception matches the expected error detail
        assert http_exc.detail == expected_error["error"]
