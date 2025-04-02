# pylint: disable=missing-function-docstring
import pytest


@pytest.fixture
def stock_symbol_io():
    return [
        {
            "input": "TCS",
            "status_code": 200,
            "symbol_token": "11536",
            "symbol": "TCS-EQ",
        },
        {
            "input": "SCT",
            "status_code": 404,
            "error": "Symbol SCT not found. Please provide a valid symbol. Refer to the NSE symbols list for valid symbols.",
        },
        {
            "input": "infy",
            "status_code": 200,
            "symbol_token": "1594",
            "symbol": "INFY-EQ",
        },
        {
            "input": "",
            "status_code": 404,
            "error": "Symbol not found. Please provide a valid symbol. Refer to the NSE symbols list for valid symbols.",
        },
    ]
