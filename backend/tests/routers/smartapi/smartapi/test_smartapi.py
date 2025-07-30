import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.routers.smartapi.smartapi.smartapi import router
from app.schemas.stock_model import HistoricalStockDataBundle, SmartAPIStockPriceInfo
from tests.utils.common.exception_validators import validate_exception

from .conftest import (
    candlestick_interval_io,
    data_unavailable_dates_io,
    date_range_io,
    different_datetime_formats_io,
    holiday_dates_io,
    invalid_trading_time_io,
    stock_symbol_io,
    stock_symbol_io_historical,
)

client = TestClient(router)


##### VCR Configuration #####
# This fixture is used to configure VCR for recording and replaying HTTP interactions.
@pytest.fixture(scope="module")
def vcr_config():
    """
    VCR config to remove the sensitive information from the api request and response
    """
    return {
        # Remove sensitive headers
        "filter_headers": ["authorization", "x-api-key", "cookie"],
        # Remove sensitive query params
        "filter_query_parameters": ["api_key", "access_token"],
        # You can also redact request body if needed
        "before_record_request": scrub_request,
        "before_record_response": scrub_response,
    }


def scrub_request(request):
    """
    Scrubs sensitive information from the request before recording it with VCR.
    """
    if "authorization" in request.headers:
        request.headers["authorization"] = "[REDACTED]"

    return request


def scrub_response(response):
    """
    Scrubs sensitive information from the response before recording it with VCR.
    This is useful for removing sensitive headers like Set-Cookie.
    """
    if "headers" in response and "Set-Cookie" in response["headers"]:
        response["headers"]["Set-Cookie"] = "[REDACTED]"

    return response


###### Helper Function ####


def validate_endpoint_io(input_stock_data: dict[str, Any]):
    """
    Validates data received from the historical stock data endpoint. The function takes
    input and expected output; it invokes the historical stock data endpoint with the
    input and then compares the endpoint's result to the provided output.
    """
    endpoint_url = (
        f"/smart-api/equity/history/{input_stock_data['input_stock_symbol']}?interval={input_stock_data['input_interval']}"
        f"&start_date={input_stock_data['input_from_date']}&end_date={input_stock_data['input_to_date']}"
    )

    if input_stock_data["status_code"] == 200:
        # Send a GET request to the endpoint URL
        response = client.get(endpoint_url)

        # Assert that the response status code matches the expected status code
        assert response.status_code == input_stock_data["status_code"]

        data = response.json()
        # Assert that the response contains JSON data
        assert data is not None

        # Parse the response JSON into a HistoricalStockDataBundle object
        smart_api_stock_price_info = HistoricalStockDataBundle.parse_obj(data)

        # Assert that the stock_price_info object is an instance of StockPriceInfo
        assert isinstance(smart_api_stock_price_info, HistoricalStockDataBundle)

    else:
        validate_exception(endpoint_url, input_stock_data, client)

    time.sleep(0.5)


###### Test Cases #####


@pytest.mark.vcr()
@stock_symbol_io
def test_latest_price_quotes(stock_symbol_io):
    """
    Tests the latest price quotes endpoint with various possible stock symbols
    """

    endpoint_url = f"/smart-api/equity/price/{stock_symbol_io['input']}"

    if stock_symbol_io["status_code"] == 200:
        # Send a GET request to the endpoint URL
        response = client.get(endpoint_url)

        # Assert that the response status code matches the expected status code
        assert response.status_code == stock_symbol_io["status_code"]

        # Assert that the response contains JSON data
        assert response.json() is not None

        # Parse the response JSON into a SmartAPIStockPriceInfo object
        smart_api_stock_price_info = SmartAPIStockPriceInfo.parse_obj(response.json())

        # Assert that the stock_price_info object is an instance of StockPriceInfo
        assert isinstance(smart_api_stock_price_info, SmartAPIStockPriceInfo)

        # Check if the stock token and symbol in the SmartAPIStockPriceInfo object matches the stock symbol data
        assert (
            smart_api_stock_price_info.symbol_token == stock_symbol_io["symbol_token"]
        )
        assert smart_api_stock_price_info.symbol == stock_symbol_io["symbol"]
    else:
        validate_exception(endpoint_url, stock_symbol_io, client)


@pytest.mark.vcr()
@stock_symbol_io_historical
def test_historical_stock_data_with_different_stock_symbols(
    stock_symbol_io: dict[str, Any],
):
    """Tests the historical stock data endpoint with various possible stock symbols
    that are either valid or invalid.

    Parameters:
    -----------
    stock_symbol_io: ``dict[str, Any]``
        Input stock data with different stock symbols
    """
    validate_endpoint_io(stock_symbol_io)


@pytest.mark.vcr()
@candlestick_interval_io
def test_historical_stock_data_with_different_intervals(
    candlestick_interval_io: dict[str, Any],
):
    """
    Tests the historical stock data endpoint with various possible candlestick intervals
    that are either valid or invalid.

    Parameters:
    -----------
    candlestick_interval_io: ``dict[str, Any]``
        Input stock data with different candlestick intervals
    """

    validate_endpoint_io(candlestick_interval_io)


@pytest.mark.vcr()
@different_datetime_formats_io
def test_historical_stock_data_with_datetime_formats(
    different_datetime_formats_io: dict[str, Any],
):
    """
    Tests the historical stock data endpoint with various possible datetime formats
    that are either valid or invalid.

    Parameters:
    -----------
    different_datetime_formats_io: ``dict[str, Any]``
        Input stock data with different datetime formats
    """
    validate_endpoint_io(different_datetime_formats_io)


@pytest.mark.vcr()
@holiday_dates_io
def test_historical_stock_data_on_holidays(holiday_dates_io: dict[str, Any]):
    """
    Tests the historical stock data endpoint on market holidays.

    Parameters:
    -----------
    holiday_dates_io: ``dict[str, Any]``
        Input stock data on market holidays
    """

    validate_endpoint_io(holiday_dates_io)


@pytest.mark.vcr()
@data_unavailable_dates_io
def test_historical_stock_data_on_data_unavailable_dates(
    data_unavailable_dates_io: dict[str, Any],
):
    """
    Tests the historical stock data endpoint on data unavailable time periods or dates.

    Parameters:
    -----------
    data_unavailable_dates_io: ``dict[str, Any]``
        Input stock data on data unavailable time periods
    """
    validate_endpoint_io(data_unavailable_dates_io)


@pytest.mark.vcr()
@invalid_trading_time_io
def test_historical_stock_data_invalid_trading_time(
    invalid_trading_time_io: dict[str, Any],
):
    """
    Tests the historical stock data endpoint with invalid trading time.

    Parameters:
    -----------
    invalid_trading_time_io: ``dict[str, Any]``
        Input stock data with invalid trading time
    """
    validate_endpoint_io(invalid_trading_time_io)


@pytest.mark.vcr()
@date_range_io
def test_historical_stock_data_invalid_date_range(date_range_io: dict[str, Any]):
    """
    Tests the historical stock data endpoint with invalid date range where date range
    can exceeds maximum or minimum limit per request.

    Parameters:
    -----------
    date_range_io: ``dict[str, Any]``
        Input stock data with invalid date range
    """
    validate_endpoint_io(date_range_io)
