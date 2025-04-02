# pylint: disable=line-too-long,
from typing import Any

import pytest


@pytest.fixture
def stock_symbol_io() -> list[dict[str, Any]]:
    """
    Test cases for different valid or invalid stock symbols.

    Return:
    -------
    ``list[dict]``
        list of inputs and respective expected outputs
    """
    return [
        {
            "input_stock_symbol": "TCS",
            "input_interval": "ONE_MINUTE",
            "input_from_date": "2023-10-11 09:15",
            "input_to_date": "2023-10-11 15:29",
            "status_code": 200,
        },
        {
            "input_stock_symbol": "SCT",
            "input_interval": "ONE_MINUTE",
            "input_from_date": "2023-10-11 09:15",
            "input_to_date": "2023-10-11 15:29",
            "status_code": 404,
            "error": "Symbol SCT not found. Please provide a valid symbol. Refer to the NSE symbols list for valid symbols.",
        },
        {
            "input_stock_symbol": "inFy",
            "input_interval": "ONE_MINUTE",
            "input_from_date": "2023-ocTOber-9 9:15",
            "input_to_date": "2023-10-9 10:5",
            "status_code": 200,
        },
        {
            "input_stock_symbol": "",
            "input_interval": "ONE_MINUTE",
            "input_from_date": "2023-10-11 09:15",
            "input_to_date": "2023-10-11 15:29",
            "status_code": 404,
            "error": "Symbol not found. Please provide a valid symbol. Refer to the NSE symbols list for valid symbols.",
        },
    ]


@pytest.fixture
def candlestick_interval_io() -> list[dict[str, Any]]:
    """
    Test cases for different valid or invalid candlestick intervals.

    Return:
    -------
    ``list[dict]``
        list of inputs and respective expected outputs
    """
    return [
        {
            "input_stock_symbol": "ACC",
            "input_interval": "one minute",
            "input_from_date": "2023-10-5 12:0",
            "input_to_date": "2023-10-05 12:20",
            "status_code": 200,
        },
        {
            "input_stock_symbol": "ACC",
            "input_interval": "",
            "input_from_date": "2023-10-11 09:15",
            "input_to_date": "2023-10-11 15:29",
            "status_code": 404,
            "error": "Candlestick interval  not found. Please provide a valid interval.",
        },
        {
            "input_stock_symbol": "INFY",
            "input_interval": "TWO_MINUTE",
            "input_from_date": "2023-10-11 9:15",
            "input_to_date": "2023-10-11 15:29",
            "status_code": 404,
            "error": "Candlestick interval TWO_MINUTE not found. Please provide a valid interval.",
        },
        {
            "input_stock_symbol": "DAbur",
            "input_interval": "one-MINUTE",
            "input_from_date": "2023-october-5 14:0",
            "input_to_date": "2023-10-05 15:20",
            "status_code": 200,
        },
        {
            "input_stock_symbol": "BPCL",
            "input_interval": "3 minutes",
            "input_from_date": "2023-10-11 9:15",
            "input_to_date": "2023-10-11 15:29",
            "status_code": 404,
            "error": "Candlestick interval 3 minutes not found. Please provide a valid interval.",
        },
    ]


@pytest.fixture
def different_datetime_formats_io() -> list[dict[str, Any]]:
    """
    Test cases for different valid or invalid datetime formats and their respective
    expected outputs.

    Return:
    -------
    ``list[dict]``
        list of inputs and respective expected outputs
    """
    return [
        {
            "input_stock_symbol": "TCS",
            "input_interval": "ONE_MINUTE",
            "input_from_date": "",
            "input_to_date": "2023-10-11 15:29",
            "status_code": 400,
            "error": "Given datetime format  is invalid. Please provide a valid datetime that should be in the form 'year-month-day hour:minute'.",
        },
        {
            "input_stock_symbol": "tcs",
            "input_interval": "ONE_MINUTE",
            "input_from_date": "2023/october/9 15:15",
            "input_to_date": "2023-OCT-9 16:29",
            "status_code": 200,
        },
        {
            "input_stock_symbol": "dabur",
            "input_interval": "ONE_MINUTE",
            "input_from_date": "2023-10-11 9:15",
            "input_to_date": "2023-10-23 3-43",
            "status_code": 400,
            "error": "Given datetime format 2023-10-23 3-43 is invalid. Please provide a valid datetime that should be in the form 'year-month-day hour:minute'.",
        },
        {
            "input_stock_symbol": "infy",
            "input_interval": "ONE_MINUTE",
            "input_from_date": "january",
            "input_to_date": "2023-10-11 15:29",
            "status_code": 400,
            "error": "Given datetime format january is invalid. Please provide a valid datetime that should be in the form 'year-month-day hour:minute'.",
        },
        {
            "input_stock_symbol": "SBIN",
            "input_interval": "ONE_MINUTE",
            "input_from_date": "2023/10/9 15:15",
            "input_to_date": "2023-OCT-9 16:29",
            "status_code": 200,
        },
        {
            "input_stock_symbol": "SBIN",
            "input_interval": "ONE_MINUTE",
            "input_from_date": "2023-10-11 40:80",
            "input_to_date": "2023-10-11 15:29",
            "status_code": 400,
            "error": "Given datetime format 2023-10-11 40:80 is invalid. Please provide a valid datetime that should be in the form 'year-month-day hour:minute'.",
        },
    ]


@pytest.fixture
def holiday_dates_io() -> list[dict[str, Any]]:
    """
    Test cases for market holiday dates and their respective expected output error.

    Return:
    -------
    ``list[dict]``
        list of inputs and respective expected outputs
    """
    return [
        {
            "input_stock_symbol": "TVSMOTOR",
            "input_interval": "ONE_MINUTE",
            "input_from_date": "2023-10-14 9:20",
            "input_to_date": "2023-10-15 13:25",
            "status_code": 400,
            "error": "All days from 2023-10-14 to 2023-10-15 are market holidays.",
        },
        {
            "input_stock_symbol": "PAYTM",
            "input_interval": "ONE_MINUTE",
            "input_from_date": "2022-04-14 12:0",
            "input_to_date": "2022-04-15 13:25",
            "status_code": 400,
            "error": "All days from 2022-04-14 to 2022-04-15 are market holidays.",
        },
    ]


@pytest.fixture
def data_unavailable_dates_io() -> list[dict[str, Any]]:
    """
    Test cases for dates on which data is not available and their respective expected output error.

    Return:
    -------
    ``list[dict]``
        list of inputs and respective expected outputs
    """
    return [
        {
            "input_stock_symbol": "TITAN",
            "input_interval": "ONE_MINUTE",
            "input_from_date": "2015-10-14 9:20",
            "input_to_date": "2015-10-15 13:25",
            "status_code": 404,
            "error": "Data for the provided dates is unavailable; please use a date range starting from the 2016-10-03 date onwards.",
        },
        {
            "input_stock_symbol": "GRINDWELL",
            "input_interval": "1d",
            "input_from_date": "2015-10-14 9:20",
            "input_to_date": "2015-10-15 13:25",
            "status_code": 404,
            "error": "No data available for this stock GRINDWELL",
        },
    ]


@pytest.fixture
def invalid_trading_time_io() -> dict[str, Any]:
    """
    Test cases for invalid trading time and their respective expected output error.

    Return:
    -------
    ``dict[str,Any]``
        list of inputs and respective expected outputs
    """
    return {
        "input_stock_symbol": "TITAN",
        "input_interval": "ONE_MINUTE",
        "input_from_date": "2023-10-11 16:00",
        "input_to_date": "2023-10-11 19:00",
        "status_code": 400,
        "error": "Attempted to access trading system outside of trading hours.",
    }


@pytest.fixture
def date_range_io() -> dict[str, Any]:
    """
    Test case for invalid date range and their respective expected output error.

    Return:
    -------
    ``dict[str,Any]``
        list of inputs and respective expected outputs
    """
    return {
        "input_stock_symbol": "TVSMOTOR",
        "input_interval": "ONE_MINUTE",
        "input_from_date": "2023-10-11 10:34",
        "input_to_date": "2023-11-20 15:29",
        "status_code": 416,
        "error": "The date range from 2023-10-11 10:34 to 2023-11-20 15:29 is invalid. Please ensure that the end date is greater than or equal to start date and difference between them does not exceed 30 for given interval ONE_MINUTE.",
    }
