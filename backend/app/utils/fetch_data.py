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
    Fetches JSON data from a specified NSE URL, retrying the request up to a given number of times.
    
    Parameters:
        url (str): The target NSE URL to fetch data from.
        max_tries (int, optional): Maximum number of request attempts. Must be greater than 0.
    
    Returns:
        Any: The parsed JSON response from the API.
    
    Raises:
        ValueError: If max_tries is less than 1.
        HTTPException: If the resource is not found (404) or if the service is unavailable after all retries (503).
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


def get_env_var(name: str, default: str | None = None) -> str:
    """
    Retrieve the value of an environment variable, returning a default if specified.
    
    If the environment variable is not set and no default is provided, raises a ValueError.
    """
    value = os.environ.get(name, default)

    if not value:
        raise ValueError(f"Missing required environment variable: {name}")

    return value
