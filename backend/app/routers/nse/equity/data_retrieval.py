from typing import Any

from fastapi import HTTPException

from app.routers.nse.equity.data_processor import (
    filter_nifty_stocks,
    filter_single_index,
    filter_single_stock,
)
from app.schemas.stock_model import StockPriceInfo
from app.utils.fetch_data import fetch_data
from app.utils.urls import ALL_INDICES, STOCK_URL


def get_nifty_index_stocks(url: str, max_tries: int = 1000) -> list[StockPriceInfo]:
    """
    Fetch the price information about the list stocks that are in the provided NSE index.

    Parameters:
    -----------
    url: ``str``
        Url for fetching the nse index stocks data
    max_tries: ``int``, (defaults = 1000)
        Maximum number of times the request has to send to get response. Requests
        are made until either get the status code `200` or exceed max_tries

    Return:
    -------
    ``list[StockData]``
        List of StockData models that contain the price information about the stocks
    """
    if url == "":
        raise ValueError("Url can't be empty")

    nifty_index_stocks = fetch_data(url, max_tries=max_tries)

    if "data" not in nifty_index_stocks:
        raise HTTPException(
            status_code=404,
            detail={"Error": "Resource not found or invalid Url"},
        )

    return filter_nifty_stocks(nifty_index_stocks["data"])


def get_stock_url(stock_symbol: str) -> str:
    """
    Returns the stock url based on the stock symbol.

    Parameters:
    -----------
    stock_symbol: ``str``
        Nse stock symbol, can be obtained from the nse official website
        eg: "SBIN","TCS" etc.

    Raises:
    ------
    ``ValueError``
        If the given stock symbol was not registered symbol in NSE

    Returns:
    --------
    stock_url: ``str``
        stock url to the stock information
    """

    if stock_symbol == "" or stock_symbol is None:
        raise ValueError("Stock symbol can't be empty")

    stock_url = f"{STOCK_URL}{stock_symbol}"

    return stock_url


def get_stock_trade_info(symbol: str) -> StockPriceInfo:
    """
    Provide the price information about given stock symbol

    Parameters:
    -----------
    symbol: ``str``
        Nse stock symbol, can be obtained from the nse official website
        eg: "SBIN","TCS" etc.

    Return:
    -------
    ``StockData``
        StockData model contain the information about the stock
    """
    stock_url = get_stock_url(symbol)
    stock_data = fetch_data(stock_url)

    if "priceInfo" not in stock_data:
        raise HTTPException(
            status_code=503,
            detail={"Error": "no data found for the given symbol at the moment"},
        )

    price_info = stock_data["priceInfo"]

    return filter_single_stock(symbol, price_info)


def get_stock_listing_date(stock_symbol: str) -> str:
    """
    Returns the listing date of the stock in NSE.

    Parameters:
    -----------
    stock_symbol: ``str``
        Nse index symbol, can be obtained from the nse official website
        eg: "NIFTY 50","NIFTY NEXT 50" etc.

    Returns:
    --------
    listing_date: ``str``
        Listing date of the given stock symbol

    """
    stock_url = get_stock_url(stock_symbol)
    stock_data = fetch_data(stock_url)

    if "metadata" not in stock_data:
        raise HTTPException(
            status_code=503,
            detail={"Error": "no metadata found for the given symbol at the moment"},
        )

    listing_date = stock_data.get("metadata").get("listingDate")

    return listing_date


def get_index_data(symbol: str) -> StockPriceInfo:
    """
    Provide the price information about the Nse indices like NIFTY 50, NIFTY BANK etc.

    Parameters:
    -----------
    symbol: ``str``
        Nse index symbol, can be obtained from the nse official website
        eg: "NIFTY 50","NIFTY NEXT 50" etc.

    Return:
    -------
    ``StockPriceInfo``
        StockData model contain the information about the index
    """
    if symbol == "" or symbol is None:
        raise ValueError("Symbol can't be empty")

    indices_data: list[dict[str, Any]] = fetch_data(ALL_INDICES)["data"]
    stock_price_info: StockPriceInfo

    for index in indices_data:
        if index["index"] == symbol.upper():
            stock_price_info = filter_single_index(index)

    return stock_price_info
