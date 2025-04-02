import time
from typing import Any

from fastapi.testclient import TestClient

from app.routers.smartapi.smartapi.smartapi import router
from app.schemas.stock_model import HistoricalStockDataBundle
from tests.utils.common.exception_validators import validate_exception

client = TestClient(router)


def validate_endpoint_io(input_stock_data: dict[str, Any]):
    """
    Validates data received from the historical stock data endpoint.

    The function takes input and expected output; it invokes the historical stock data endpoint
    with the input and then compares the endpoint's result to the provided output.
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


def test_historical_stock_data_with_different_stock_symbols(
    stock_symbol_io: list[dict[str, Any]]
):
    """Tests the historical stock data endpoint with various possible stock symbols
    that are either valid or invalid.

    Parameters:
    -----------
    stock_symbol_io: ``list[dict[str, Any]]``
        Input stock data with different stock symbols
    """

    for stock_symbol_data in stock_symbol_io:
        validate_endpoint_io(stock_symbol_data)


def test_historical_stock_data_with_different_intervals(
    candlestick_interval_io: list[dict[str, Any]]
):
    """
    Tests the historical stock data endpoint with various possible candlestick intervals
    that are either valid or invalid.

    Parameters:
    -----------
    candlestick_interval_io: ``list[dict[str, Any]]``
        Input stock data with different candlestick intervals
    """
    for interval_data in candlestick_interval_io:
        validate_endpoint_io(interval_data)


def test_historical_stock_data_with_datetime_formats(
    different_datetime_formats_io: list[dict[str, Any]]
):
    """
    Tests the historical stock data endpoint with various possible datetime formats
    that are either valid or invalid.

    Parameters:
    -----------
    different_datetime_formats_io: ``list[dict[str, Any]]``
        Input stock data with different datetime formats
    """
    for datetime_format_data in different_datetime_formats_io:
        validate_endpoint_io(datetime_format_data)


def test_historical_stock_data_on_holidays(holiday_dates_io: list[dict[str, Any]]):
    """
    Tests the historical stock data endpoint on market holidays.

    Parameters:
    -----------
    holiday_dates_io: ``list[dict[str, Any]]``
        Input stock data on market holidays
    """
    for holiday_date_data in holiday_dates_io:
        validate_endpoint_io(holiday_date_data)


def test_historical_stock_data_on_data_unavailable_dates(
    data_unavailable_dates_io: list[dict[str, Any]]
):
    """
    Tests the historical stock data endpoint on data unavailable time periods or dates.

    Parameters:
    -----------
    data_unavailable_dates_io: ``list[dict[str, Any]]``
        Input stock data on data unavailable time periods
    """
    for data_unavailable_date_data in data_unavailable_dates_io:
        validate_endpoint_io(data_unavailable_date_data)


def test_historical_stock_data_invalid_trading_time(
    invalid_trading_time_io: dict[str, Any]
):
    """
    Tests the historical stock data endpoint with invalid trading time.

    Parameters:
    -----------
    invalid_trading_time_io: ``dict[str, Any]``
        Input stock data with invalid trading time
    """
    validate_endpoint_io(invalid_trading_time_io)


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
