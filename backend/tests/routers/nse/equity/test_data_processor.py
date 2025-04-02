# pylint: disable=too-many-arguments
import pytest

from app.routers.nse.equity.data_processor import (
    filter_nifty_stocks,
    filter_single_index,
    filter_single_stock,
)
from app.schemas.stock_model import StockPriceInfo


def get_stock_io_data(
    symbol, last_traded_price, day_open, day_low, day_high, change, percent_change
):
    """
    Returns a tuple containing input and expected output data for a stock.

    Parameters:
    -----------
    symbol: ``str``
        The stock symbol
    last_traded_price:``float``
        The last traded price of the stock
    day_open:``float``
        The opening price of the stock for the day
    day_low:``float``
        The lowest price of the stock for the day
    day_high:``float``
        The highest price of the stock for the day
    change:``float``
        The change in the stock price
    percent_change:``float``
        The percentage change in the stock price

    Returns:
    --------
    ``tuple``
        A tuple containing the input data and expected output data for the stock
    """
    stock_input_data = {
        "symbol": symbol,
        "lastPrice": last_traded_price,
        "open": day_open,
        "dayLow": day_low,
        "dayHigh": day_high,
        "change": change,
        "pChange": percent_change,
    }
    expected_output_data = StockPriceInfo(
        symbol=symbol,
        last_traded_price=last_traded_price,
        day_open=day_open,
        day_low=day_low,
        day_high=day_high,
        change=change,
        percent_change=percent_change,
    )

    return stock_input_data, expected_output_data


def test_filter_nifty_stocks():
    """
    Test function to test the filter_nifty_stocks function.

    Test cases:
    1. Empty input list
    2. Single stock data
    3. Multiple stock data
    """
    # Test case 1: Empty input list
    with pytest.raises(ValueError) as value_error:
        filter_nifty_stocks([])
        assert value_error == "Stocks data is empty"

    # Test case 2: Single stock data
    single_stock_data, single_stock_output = get_stock_io_data(
        "RELIANCE", 2000.0, 1990.0, 1980.0, 2010.0, 10.0, 0.5
    )
    assert filter_nifty_stocks([single_stock_data]) == [single_stock_output]

    # Test case 3: Multiple stock data
    stocks_data, stocks_output = [], []
    stock_price_info = [
        ["Reliance", 2000.0, 1990.0, 1980.0, 2010.0, 10.0, 0.5],
        ["TATASTEEL", 1500.0, 1510.0, 1490.0, 1520.0, -10.0, -0.5],
    ]

    for stock_price in stock_price_info:
        stock_data, stock_output = get_stock_io_data(*stock_price)
        stocks_data.append(stock_data)
        stocks_output.append(stock_output)

    assert filter_nifty_stocks(stocks_data) == stocks_output


def test_filter_single_stock():
    """
    Test function for filtering single stock data.

    Test case 1: Single stock data
    Test case 2: Empty input dict
    """
    stock_data = {
        "symbol": "RELIANCE",
        "lastPrice": 2000.0,
        "open": 1990.0,
        "intraDayHighLow": {"max": 2010.0, "min": 1980.0},
        "change": 10.0,
        "pChange": 0.5,
    }
    expected_output = StockPriceInfo(
        symbol="RELIANCE",
        last_traded_price=2000.0,
        day_open=1990.0,
        day_low=1980.0,
        day_high=2010.0,
        change=10.0,
        percent_change=0.5,
    )
    assert filter_single_stock("RELIANCE", stock_data) == expected_output

    with pytest.raises(ValueError) as value_error:
        filter_nifty_stocks({})
        assert value_error == "Stocks data is empty"


def test_filter_single_index():
    """
    Test function to test the filter_single_index function.

    Test case 1: Empty input dict
    Test case 2: Non-empty input dict
    """
    with pytest.raises(ValueError) as value_error:
        filter_nifty_stocks({})
        assert value_error == "index data is empty"

    index_data = {
        "index": "NIFTY 50",
        "last": 15000.0,
        "high": 15100.0,
        "low": 14900.0,
        "open": 14950.0,
        "previousClose": 14800.0,
        "percentChange": 1.01,
    }
    expected_output = StockPriceInfo(
        symbol="NIFTY 50",
        last_traded_price=15000.0,
        day_open=14950.0,
        day_low=14900.0,
        day_high=15100.0,
        change=200.0,
        percent_change=1.01,
    )
    assert filter_single_index(index_data) == expected_output
