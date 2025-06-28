from datetime import datetime

import pytest
from fastapi import HTTPException

from app.utils.validators import (
    get_date_format,
    validate_and_format_stock_symbol,
    validate_and_reformat_date,
    validate_derivative_symbol_with_type,
    validate_index_symbol,
)


def test_validate_and_format_stock_symbol() -> None:
    """
    Test function to validate and format stock symbols.

    This function tests the following scenarios:
    - Valid stock symbol
    - Valid stock symbol with lowercase letters
    - Invalid stock symbol

    """
    # Test valid stock symbol
    assert validate_and_format_stock_symbol("TCS") == "TCS"

    # Test valid stock symbol with lowercase letters
    assert validate_and_format_stock_symbol("infy") == "INFY"

    # Test invalid stock symbol
    with pytest.raises(HTTPException) as http_exc:
        validate_and_format_stock_symbol("ABC")
        assert http_exc.value.status_code == 404
        assert http_exc.value.detail == {
            "Error": "ABC is not a valid stock symbol. Please refer nse official website to get stock symbols"
        }


def test_validate_index_symbol() -> None:
    """
    Test function to validate index symbol.

    This function tests the validity of an index symbol by checking if it
    exists in the NSE official website. It tests both valid and invalid
    index symbols and raises an HTTPException if the symbol is invalid.
    """
    # Test valid index symbol
    assert validate_index_symbol("NIFTY 50") == "NIFTY%2050"

    # Test valid index symbol with lowercase letters
    assert validate_index_symbol("nifty 100") == "NIFTY%20100"

    # Test invalid index symbol
    with pytest.raises(HTTPException) as http_exc:
        validate_index_symbol("XYZ")

        assert http_exc.value.status_code == 404
        assert http_exc.value.detail == {
            "Error": "XYZ is not a valid index symbol. Please refer nse official website to get index symbols"
        }


def test_validate_derivative_symbol_with_type() -> None:
    """
    Test function to validate derivative symbol and type.

    This function tests the following scenarios:
    - Valid derivative symbol and type
    - Valid stock derivative symbol and type
    - Invalid derivative symbol and type
    - Invalid derivative symbol
    - Invalid derivative symbol with index type
    - Invalid derivative symbol with stock type

    """
    # Test valid derivative symbol and type
    validate_derivative_symbol_with_type("NIFTY", "index")  # Should not raise

    # Test valid stock derivative symbol and type
    validate_derivative_symbol_with_type("TCS", "stock")  # Should not raise

    # Test invalid derivative symbol and type
    with pytest.raises(HTTPException) as http_exc:
        validate_derivative_symbol_with_type("TCS", "index")
        assert http_exc.value.status_code == 400
        assert http_exc.value.detail == {
            "Error": "TCS and index not matched. "
            + "If derivative is index like NIFTY then derivative type should be index and vice versa"
        }

    # Test invalid derivative symbol
    with pytest.raises(HTTPException) as http_exc:
        validate_derivative_symbol_with_type("XYZ", "stock")
        assert http_exc.value.status_code == 400
        assert http_exc.value.detail == {
            "Error": "XYZ is not a valid derivative symbol. "
            + "Please refer nse official website to get derivative symbols"
        }

    # Test invalid derivative symbol with index type
    with pytest.raises(HTTPException) as http_exc:
        validate_derivative_symbol_with_type("TCS", "index")
        assert http_exc.value.status_code == 400
        assert http_exc.value.detail == {
            "Error": "TCS and index not matched. "
            + "If derivative is index like NIFTY then derivative type should be index and vice versa"
        }

    # Test invalid derivative symbol with stock type
    with pytest.raises(HTTPException) as http_exc:
        validate_derivative_symbol_with_type("NIFTY", "stock")

        assert http_exc.value.status_code == 400
        assert http_exc.value.detail == {
            "Error": "NIFTY is not a valid derivative symbol. "
            + "Please refer nse official website to get derivative symbols"
        }


def test_get_date_format() -> None:
    """
    Test function to check the get_date_format function.

    This function tests the get_date_format function with different date formats.
    It checks if the function returns the correct date format string for each
    input date format.
    """
    # Test date format with "/"
    assert get_date_format("09/09/2023") == "%d/%m/%Y"

    # Test date format with "-"
    assert get_date_format("09-09-2023") == "%d-%m-%Y"

    # Test date format with month name
    assert get_date_format("09-Sep-2023") == "%d-%b-%Y"

    # Test date format with month number
    assert get_date_format("09-09-23") == "%d-%m-%Y"

    # Test date format with mixed format of '/' and '-'
    assert get_date_format("09/09-2023") == "%d/%m/%Y"


def test_validate_and_reformat_expiry_date() -> None:
    """
    Test function to validate and reformat expiry date.

    This function tests the validity of the date format and reformats
    it to the required format.
    """
    # Test valid date format
    assert validate_and_reformat_date("09-Sep-2023") == (
        datetime(2023, 9, 9).strftime("%d-%b-%Y"),
        True,
    )

    # Test invalid date format
    assert validate_and_reformat_date("2023-09-09") == ("2023-09-09", False)
