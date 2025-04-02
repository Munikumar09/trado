from typing import Any

from app.schemas.stock_model import StockPriceInfo


def filter_nifty_stocks(stocks_data: list[dict[str, Any]]) -> list[StockPriceInfo]:
    """
    Filter the required data and create a list of pydantic models from
    the given api response for the list of index stocks.

    Parameters:
    -----------
    stocks_data: ``list[dict[str,Any]]``
        Json parsed response of stock from the Nse api

    Return:
    -------
    ``list[StockData]``
        List of StockData models that contain the price information about the stocks
    """
    if len(stocks_data) == 0:
        raise ValueError("Stocks data is empty")

    filtered_stocks_data = []

    for stock_data in stocks_data:
        filtered_stocks_data.append(
            StockPriceInfo(
                symbol=stock_data["symbol"],
                last_traded_price=stock_data["lastPrice"],
                day_open=stock_data["open"],
                day_low=stock_data["dayLow"],
                day_high=stock_data["dayHigh"],
                change=stock_data["change"],
                percent_change=stock_data["pChange"],
            )
        )

    return filtered_stocks_data


def filter_single_stock(symbol: str, stock_data: dict[str, Any]) -> StockPriceInfo:
    """
    Filter the required data and create a pydantic model from the
    given api response for single stock.

    Parameters:
    -----------
    symbol: ``str``
        Nse stock symbol, can be obtained from the nse official website
            eg: "SBIN","TCS" etc.
    raw_stock_data: ``dict[str,Any]``
        Json parsed response of s stock from the Nse api

    Return:
    -------
    ``StockData``
        StockData model contain the information about the stock
    """
    if len(stock_data) == 0:
        raise ValueError("Stock data is empty")

    return StockPriceInfo(
        symbol=symbol,
        last_traded_price=stock_data["lastPrice"],
        day_open=stock_data["open"],
        day_high=stock_data["intraDayHighLow"]["max"],
        day_low=stock_data["intraDayHighLow"]["min"],
        change=stock_data["change"],
        percent_change=stock_data["pChange"],
    )


def filter_single_index(index_data: dict[str, Any]) -> StockPriceInfo:
    """
    Filter the required data and create a pydantic model from the
    given api response for single index.

    Parameters:
    -----------
    index_data: ``dict[str,Any]``
        Json parsed response of s stock from the Nse api

    Return:
    -------
    ``StockData``
        StockData model contains the price information about the index
    """
    if len(index_data) == 0:
        raise ValueError("Index data is empty")

    return StockPriceInfo(
        symbol=index_data["index"],
        last_traded_price=index_data["last"],
        day_high=index_data["high"],
        day_low=index_data["low"],
        day_open=index_data["open"],
        change=index_data["last"] - index_data["previousClose"],
        percent_change=index_data["percentChange"],
    )
