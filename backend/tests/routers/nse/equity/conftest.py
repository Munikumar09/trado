# pylint: disable=missing-function-docstring
import pytest


@pytest.fixture
def get_stock_data_io():
    return [
        {"input": "TCS", "status_code": 200},
        {
            "input": "AAAAB",
            "status_code": 404,
            "detail": {
                "Error": "AAAAB is not a valid stock symbol. "
                + "Please refer nse official website to get stock symbols"
            },
        },
        {
            "input": "",
            "status_code": 404,
            "detail": {
                "Error": " is not a valid stock symbol. "
                + "Please refer nse official website to get stock symbols"
            },
        },
    ]


@pytest.fixture
def get_nifty_index_stocks_io():
    return [
        {"symbol": "NIFTY 50", "status_code": 200, "total_stocks": 51},
        {"symbol": "NIFTY BANK", "status_code": 200, "total_stocks": 13},
        {
            "symbol": "NIFF",
            "status_code": 404,
            "detail": {
                "Error": "NIFF is not a valid index symbol. "
                + "Please refer nse official website to get index symbols"
            },
        },
    ]


@pytest.fixture
def nse_index_data_io():
    return [
        {"symbol": "NIFTY 50", "status_code": 200},
        {"symbol": "NIFTY BANK", "status_code": 200},
        {
            "symbol": "bank",
            "status_code": 404,
            "detail": {
                "Error": "BANK is not a valid index symbol. "
                + "Please refer nse official website to get index symbols"
            },
        },
    ]


@pytest.fixture
def nse_stock_listing_date_io():
    return [
        {"input": "TCS", "status_code": 200, "listing_date": "25-Aug-2004"},
        {
            "input": "AAAAB",
            "status_code": 404,
            "detail": {
                "Error": "AAAAB is not a valid stock symbol. "
                + "Please refer nse official website to get stock symbols"
            },
        },
        {
            "input": "",
            "status_code": 404,
            "detail": {
                "Error": " is not a valid stock symbol. "
                + "Please refer nse official website to get stock symbols"
            },
        },
    ]
