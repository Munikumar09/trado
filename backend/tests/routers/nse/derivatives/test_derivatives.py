# pylint: disable=missing-function-docstring
# import json
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.routers.nse.derivatives.derivatives import router
from app.utils.date_utils import get_date  # , get_expiry_dates

client = TestClient(router)


def validate_option(option: dict[str, Any]) -> None:
    assert "ltp" in option
    assert "change" in option
    assert "percent_change" in option
    assert "change_in_oi" in option
    assert "percent_change_in_oi" in option


# def test_index_option_chain_valid():
#     next_expiry_date = get_expiry_dates("tcs")[0]
#     expiry_date_lower_case = next_expiry_date.lower()
#     response = client.get(
#         f"/nse/derivatives/NIFTY?expiry_date={expiry_date_lower_case}&derivative_type=index"
#     )
#     assert response.status_code == 200

#     response_data = json.loads(response.content.decode("utf-8"))
#     assert isinstance(response_data, dict)
#     assert "strike_prices" in response_data
#     assert "expiry_date" in response_data
#     assert next_expiry_date == response_data["expiry_date"]

#     strike_prices = response_data["strike_prices"]
#     assert isinstance(strike_prices, list)
#     assert isinstance(strike_prices[0], dict)
#     assert "strike_price" in strike_prices[0]
#     assert "ce" in strike_prices[0]
#     assert "pe" in strike_prices[0]

#     middle_strike_price = strike_prices[len(strike_prices) // 2]

#     validate_option(middle_strike_price["ce"])
#     validate_option(middle_strike_price["pe"])


def validate_error_response(
    derivative_symbol: str,
    derivative_type: str,
    status_code: int,
    expected_error: dict[str, Any],
    expiry_date: str,
):
    with pytest.raises(HTTPException) as http_exception:
        client.get(
            f"/nse/derivatives/{derivative_symbol}?expiry_date={expiry_date}&derivative_type={derivative_type}"
        )
    assert http_exception.value.status_code == status_code
    assert http_exception.value.detail == expected_error


def test_index_option_chain_invalid_date():
    invalid_expiry_dates = ["31-Feb-2023", "21-se-2023", "21/21/2023", "23/10/23"]

    for expiry_date in invalid_expiry_dates:
        expected_error = {
            "Error": f"{expiry_date} is not valid. It should be dd-MM-yyyy. eg, 28-Sep-2023"
        }
        validate_error_response("NIFTY", "index", 400, expected_error, expiry_date)


# def test_index_option_chain_invalid_expiry_dates():
#     derivative_symbol = "BANKNIFTY"
#     invalid_expiry_dates = ["28-Sep-2023", "08-May-2025", "02-Oct-2023"]

#     for expiry_date in invalid_expiry_dates:
#         expected_error = {
#             "Error": f"No expiry for {derivative_symbol} on {expiry_date}"
#         }
#         validate_error_response(
#             derivative_symbol, "index", 400, expected_error, expiry_date
#         )


def test_index_option_chain_invalid_symbol_and_type():
    next_expiry_date = get_date("thursday")
    derivative_symbols = ["NIFTY", "INFY"]
    derivative_types = ["stock", "index"]

    for derivative_symbol, derivative_type in zip(derivative_symbols, derivative_types):
        expected_error = {
            "Error": f"{derivative_symbol} and {derivative_type} not matched. "
            + "If derivative is index like NIFTY then derivative type should be index and vice versa"
        }

        validate_error_response(
            derivative_symbol, derivative_type, 400, expected_error, next_expiry_date
        )
    invalid_derivative_symbol = "INFYY"
    invalid_symbol_expected_error = {
        "Error": f"{invalid_derivative_symbol} is not a valid derivative symbol. "
        + "Please refer nse official website to get derivative symbols"
    }

    validate_error_response(
        invalid_derivative_symbol,
        "stock",
        400,
        invalid_symbol_expected_error,
        next_expiry_date,
    )
