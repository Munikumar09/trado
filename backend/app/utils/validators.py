import re
from datetime import datetime
from typing import Annotated

from fastapi import HTTPException, Path

from app.utils.file_utils import get_symbols
from app.utils.urls import NSE_F_AND_O_SYMBOLS, NSE_INDEX_SYMBOLS, NSE_STOCK_SYMBOLS


def validate_and_format_stock_symbol(stock_symbol: str) -> str:
    """
    validate stock symbol with the available stock symbols in the Nse official website and change the symbol
    case to upper.

    Parameters:
    -----------
    stock_symbol: ``str``
        stock symbol to be validated
            eg: "TCS", "INFY", etc.

    Raises:
    -------
    ``HTTPException``
        Raises when the stock symbol is not available in the Nse official website

    Return:
    -------
    ``str``
        Given stock symbol in upper case
    """
    symbols: set[str] = set(get_symbols(NSE_STOCK_SYMBOLS)["symbols"])

    if stock_symbol.upper() not in symbols:
        raise HTTPException(
            status_code=404,
            detail={
                "Error": f"{stock_symbol} is not a valid stock symbol. "
                + "Please refer nse official website to get stock symbols"
            },
        )

    return stock_symbol.upper()


def validate_index_symbol(index_symbol: Annotated[str, Path()]) -> str:
    """
    Validate index symbol with the available index symbols in the Nse official website.

    Parameters:
    -----------
    index_symbol: ``Annotated[str, Path()]``
        index symbol to be validated
            eg: "NIFTY 50","NIFTY 100", etc.

    Raises:
    -------
    ``HTTPException``
        Raises when the index symbol is not available in the Nse official website.

    Return:
    -------
    ``str``
        Url path to the index symbol endpoint.
    """
    symbols: dict[str, str] = get_symbols(NSE_INDEX_SYMBOLS)
    index_symbol = index_symbol.upper()

    if index_symbol not in symbols:
        raise HTTPException(
            status_code=404,
            detail={
                "Error": f"{index_symbol} is not a valid index symbol. "
                + "Please refer nse official website to get index symbols"
            },
        )

    return symbols[index_symbol]


def validate_derivative_symbol_with_type(
    derivative_symbol: str, derivative_type: str
) -> None:
    """
    Validate derivative symbol with the available derivative symbols in the Nse official website.

    Parameters:
    -----------
    derivative_symbol: ``str``
        Derivative symbol to be validated.
            eg: "NIFTY", "ABB", etc.
    derivative_type: ``str``
        Derivative type should be either index of stock.

    Raises:
    -------
    ``HTTPException``
        If derivative type and derivate symbol mismatch.
        If derivative symbol is not present the symbols file.
    """
    all_derivative_symbols = get_symbols(NSE_F_AND_O_SYMBOLS)
    index_derivatives = ["NIFTY", "BANKNIFTY", "MIDCPNIFTY", "FINNIFTY"]

    if (derivative_type == "index" and derivative_symbol not in index_derivatives) or (
        derivative_type == "stock" and derivative_symbol in index_derivatives
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "Error": f"{derivative_symbol} and {derivative_type} not matched. "
                + "If derivative is index like NIFTY then derivative type should be index and vice versa"
            },
        )

    if derivative_symbol not in all_derivative_symbols and derivative_type != "index":
        raise HTTPException(
            status_code=400,
            detail={
                "Error": f"{derivative_symbol} is not a valid derivative symbol. "
                + "Please refer nse official website to get derivative symbols"
            },
        )


def get_date_format(date: str) -> str:
    """
    Gives the format of given date eg, if date is `09/09/2023` then format is `%d/%m/%Y`.

    Parameters:
    -----------
    date: ``str``
        Date to get the format.

    Return:
    -------
    ``str``
        Format of the given date.
    """
    date_separator = "/" if len(date.split("/")) > 1 else "-"
    month_format = "%b" if re.search("[a-zA-Z]+", date) else "%m"

    return f"%d{date_separator}{month_format}{date_separator}%Y"


def validate_and_reformat_date(data: str) -> tuple[str, bool]:
    """
    Validate the given expiry date to ensure that the given date is in "dd-MM-yyyy" format.

    Parameters:
    -----------
    date: ``str``
        expiry date to be validated.

    Return:
    -------
    ``tuple[str,bool]``
        Reformated expiry date is date is valid else input date and validation status of expiry date.
    """
    required_date_format = "%d-%b-%Y"
    date_format = get_date_format(data)

    try:
        date_obj = datetime.strptime(data, date_format)
        return date_obj.strftime(required_date_format), True
    except ValueError:
        return data, False
