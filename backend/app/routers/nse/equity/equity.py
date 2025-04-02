from typing import Annotated

from fastapi import APIRouter, Depends, Path

from app.routers.nse.equity.data_retrieval import (
    get_index_data,
    get_nifty_index_stocks,
    get_stock_listing_date,
    get_stock_trade_info,
)
from app.schemas.stock_model import StockPriceInfo
from app.utils.urls import NIFTY_INDEX_BASE
from app.utils.validators import validate_and_format_stock_symbol, validate_index_symbol

router = APIRouter(prefix="/nse/equity", tags=["equity"])


@router.get(
    "/stock/{stock_symbol}",
    response_model=StockPriceInfo,
)
async def get_stock_data(stock_symbol: Annotated[str, Path()]):
    """
    Get the stock data for a given symbol. This endpoint provides
    the latest trade information for the stock symbol from an external API.

    Parameters:
    -----------
    - **stock_symbol**:
        It must be a valid stock symbol that is registered in the NSE website.
        eg: `TCS`, `RELIANCE`

    """
    formatted_stock_symbol = validate_and_format_stock_symbol(stock_symbol)

    return get_stock_trade_info(formatted_stock_symbol)


@router.get("/index_stocks/{index_symbol}", response_model=list[StockPriceInfo])
async def nifty_index_stocks(
    index_symbol: Annotated[dict[str, str], Depends(validate_index_symbol)]
):
    """
    Get the trade information about list of stocks that are belong to the given index.

    Parameters:
    -----------
    - **index_symbol**:
        It must be a valid index symbol that is registered in the NSE website
        eg: `NIFTY 50`, `NIFTY BANK`
    """
    index_url = f"{NIFTY_INDEX_BASE}{index_symbol}"

    return get_nifty_index_stocks(index_url)


@router.get("/index/{index_symbol}", response_model=StockPriceInfo)
async def nse_index_data(index_symbol: Annotated[str, Path()]):
    """
    Get the trade information about a Nse index.

    Parameters:
    -----------
     - **index_symbol**:
        It must be a valid index symbol that is registered in the NSE website
        eg: `NIFTY 50`
    """
    validate_index_symbol(index_symbol)

    return get_index_data(index_symbol)


@router.get("/listing/{stock_symbol}", response_model=str)
async def stock_listing_date_nse(stock_symbol: Annotated[str, Path()]):
    """
    Get the listing date of the given stock in the NSE.

    Parameters:
    -----------
    - **stock_symbol**:
        It must be a valid stock symbol that is registered in the NSE website
        eg: `tcs` or `TCS`
    """
    formatted_stock_symbol = validate_and_format_stock_symbol(stock_symbol)

    return get_stock_listing_date(formatted_stock_symbol)
