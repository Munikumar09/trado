import json
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException

from app.utils.common.logger import get_logger
from app.utils.headers import REQUEST_HEADERS
from app.utils.urls import NSE_BASE_URL

logger = get_logger(Path(__file__).name)


def fetch_data(url: str, max_tries: int = 10) -> Any:
    """
    Fetch the data from given nse url and decode the response as a key value paris.

    Parameters:
    -----------
    url: ``str``
        Url from nse to fetch the data
    max_tries: ``int`` (defaults = 1000)
        Maximum number of times the request has to send to get response. Requests
        are made until either get the status code `200` or exceed max_tries

    Raises:
    -------
    ``HTTPException``
        If not get response even after max_tries

    Returns:
    --------
    ``Any``
        Json loaded response from the api.
    """
    if max_tries < 1:
        raise ValueError("max_tries should be greater than 0")

    response = httpx.get(NSE_BASE_URL, headers=REQUEST_HEADERS)
    cookies = dict(response.cookies)

    with httpx.Client(headers=REQUEST_HEADERS, cookies=cookies, timeout=5) as client:
        for _ in range(max_tries):
            response = client.get(url)

            if response.status_code == 200:
                decoded_response = response.content.decode("utf-8")
                try:
                    return json.loads(decoded_response)
                except json.JSONDecodeError:
                    logger.error("Error in decoding response: %s", decoded_response)
                    continue

            if response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail={"Error": "Resource not found or invalid Url"},
                )

        raise HTTPException(
            status_code=503,
            detail={"Error": "Service Unavailable"},
        )


def get_required_env_var(name: str) -> str:
    """
    Get the required environment variable from the system.

    Parameters:
    -----------
    name: ``str``
        Name of the environment variable to fetch

    Raises:
    -------
    ``ValueError``
        If the required environment variable is not found

    Returns:
    --------
    ``str``
        Value of the environment variable
    """
    value = os.environ.get(name)

    if not value:
        raise ValueError(f"Missing required environment variable: {name}")

    return value
