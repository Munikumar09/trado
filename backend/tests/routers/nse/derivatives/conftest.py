# pylint: disable=missing-function-docstring
import pytest

from app.utils.date_utils import get_date, get_expiry_dates
from app.utils.type_utils import SymbolType

option_chain_data = [
    {
        "strikePrice": 11000,
        "expiryDate": "28-Dec-2023",
        "PE": {
            "strikePrice": 11000,
            "expiryDate": "28-Dec-2023",
            "underlying": "NIFTY",
            "identifier": "OPTIDXNIFTY28-12-2023PE11000.00",
            "openInterest": 37.5,
            "changeinOpenInterest": 0,
            "pchangeinOpenInterest": 0,
            "totalTradedVolume": 0,
            "impliedVolatility": 0,
            "lastPrice": 3.05,
            "change": 0,
            "pChange": 0,
            "totalBuyQuantity": 3400,
            "totalSellQuantity": 0,
            "bidQty": 50,
            "bidprice": 3.05,
            "askQty": 0,
            "askPrice": 0,
            "underlyingValue": 19653.5,
        },
        "CE": {
            "strikePrice": 11000,
            "expiryDate": "28-Dec-2023",
            "underlying": "NIFTY",
            "identifier": "OPTIDXNIFTY28-12-2023CE11000.00",
            "openInterest": 141,
            "changeinOpenInterest": 0,
            "pchangeinOpenInterest": 0,
            "totalTradedVolume": 0,
            "impliedVolatility": 0,
            "lastPrice": 8785.75,
            "change": 0,
            "pChange": 0,
            "totalBuyQuantity": 950,
            "totalSellQuantity": 100,
            "bidQty": 550,
            "bidprice": 8550.7,
            "askQty": 50,
            "askPrice": 8699.95,
            "underlyingValue": 19653.5,
        },
    },
    {
        "strikePrice": 13000,
        "expiryDate": "21-Dec-2023",
        "CE": {
            "strikePrice": 13000,
            "expiryDate": "21-Dec-2023",
            "underlying": "NIFTY",
            "identifier": "OPTIDXNIFTY28-12-2023CE13000.00",
            "openInterest": 9718,
            "changeinOpenInterest": 3,
            "pchangeinOpenInterest": 0.03088008234688626,
            "totalTradedVolume": 7,
            "impliedVolatility": 0,
            "lastPrice": 6705,
            "change": 111.05000000000018,
            "pChange": 1.684119533815091,
            "totalBuyQuantity": 1100,
            "totalSellQuantity": 350,
            "bidQty": 50,
            "bidprice": 6617.05,
            "askQty": 50,
            "askPrice": 7166.2,
            "underlyingValue": 19653.5,
        },
        "PE": {
            "strikePrice": 13000,
            "expiryDate": "21-Dec-2023",
            "underlying": "NIFTY",
            "identifier": "OPTIDXNIFTY28-12-2023PE13000.00",
            "openInterest": 3923,
            "changeinOpenInterest": -10,
            "pchangeinOpenInterest": -0.25425883549453343,
            "totalTradedVolume": 31,
            "impliedVolatility": 36.57,
            "lastPrice": 5.5,
            "change": 0.4500000000000002,
            "pChange": 8.910891089108915,
            "totalBuyQuantity": 25950,
            "totalSellQuantity": 5600,
            "bidQty": 1800,
            "bidprice": 5,
            "askQty": 150,
            "askPrice": 5.4,
            "underlyingValue": 19653.5,
        },
    },
]


@pytest.fixture
def get_option_chain_io():
    return [
        {
            "input": [
                get_expiry_dates("NIFTY", SymbolType.DERIVATIVE)[0],
                "NIFTY",
                "index",
            ],
            "output": None,
        },
        {
            "input": [get_expiry_dates("TCS")[0], "TCS", "stock"],
            "output": None,
        },
        {
            "input": [get_date("tuesday", True), "INFY", "stock"],
            "output": {
                "status_code": 400,
                "detail": {
                    "Error": f"No expiry for {'INFY'} on {get_date('tuesday',True)}"
                },
            },
        },
    ]


@pytest.fixture
def get_option_io():
    return [
        {
            "input": option_chain_data[0]["CE"],
            "output": {
                "ltp": 8785.75,
                "change": 0,
                "percent_change": 0,
                "change_in_oi": 0,
                "percent_change_in_oi": 0,
            },
        },
        {"input": {}, "output": None},
    ]


@pytest.fixture
def get_filter_option_chain_io():
    return [
        {
            "input": [[option_chain_data[0]], "28-Dec-2023"],
            "output": {
                "expiry_date": "28-Dec-2023",
                "strike_prices": [
                    {
                        "strike_price": 11000,
                        "ce": {
                            "ltp": 8785.75,
                            "change": 0,
                            "percent_change": 0,
                            "change_in_oi": 0,
                            "percent_change_in_oi": 0,
                        },
                        "pe": {
                            "ltp": 3.05,
                            "change": 0,
                            "percent_change": 0,
                            "change_in_oi": 0,
                            "percent_change_in_oi": 0,
                        },
                    }
                ],
            },
        },
        {"input": [[], ""], "output": {"strike_prices": [], "expiry_date": ""}},
        {"input": [None, None], "output": None},
    ]
