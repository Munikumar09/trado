import json
from typing import Annotated

from fastapi import APIRouter, Path, Query

from app.schemas.stock_model import HistoricalStockDataBundle, SmartAPIStockPriceInfo
from app.utils.common.exceptions.historical_data import UnkownException
from app.utils.common.types.financial_types import ExchangeType
from app.utils.common.types.reques_types import CandlestickInterval, RequestType
from app.utils.smartapi.connection import get_endpoint_connection
from app.utils.smartapi.data_processor import (
    process_smart_api_historical_stock_data,
    process_smart_api_stock_data,
)
from app.utils.smartapi.urls import CANDLE_DATA_URL, LAST_TRADED_PRICE_URL
from app.utils.smartapi.validator import validate_dates, validate_symbol_and_get_token

router = APIRouter(prefix="/smart-api/equity", tags=["equity"])


@router.get("/price/{stock_symbol}", response_model=SmartAPIStockPriceInfo)
async def latest_price_quote(stock_symbol: Annotated[str, Path()]):
    """
    Get the stock data for a given symbol. This endpoint provides the latest
    trade information for the stock symbol from an External API in realtime.

    Parameters:
    -----------
    - **stock_symbol**:
        It must be a valid stock symbol that is registered in the NSE website
        eg: `TCS`, `RELIANCE`

    """

    stock_token, stock_symbol = validate_symbol_and_get_token(
        stock_exchange=ExchangeType.NSE, stock_symbol=stock_symbol
    )
    payload = {
        "exchange": ExchangeType.NSE.name,
        "tradingsymbol": stock_symbol,
        "symboltoken": stock_token,
    }
    json_payload = json.dumps(payload)

    connection = get_endpoint_connection(
        payload=json_payload,
        request_method_type=RequestType.POST,
        url=LAST_TRADED_PRICE_URL,
    )
    res = connection.getresponse()
    data = json.loads(res.read().decode("utf-8"))

    if not data["data"]:
        raise UnkownException(error_message=data["message"], status_code=res.status)

    return process_smart_api_stock_data(data["data"])


@router.get("/history/{stock_symbol}", response_model=HistoricalStockDataBundle)
async def historical_stock_data(
    stock_symbol: Annotated[str, Path()],
    interval: Annotated[
        str,
        Query(
            examples={
                "ONE_MINUTE": {
                    "value": "ONE_MINUTE",
                    "description": "Maximum 30 days one minute candlestick data is provided for one request.",
                },
                "THREE_MINUTE": {
                    "value": "THREE_MINUTE",
                    "description": "Maximum 60 days three minute candlestick data is provided for one request.",
                },
                "FIVE_MINUTE": {
                    "value": "FIVE_MINUTE",
                    "description": "Maximum 100 days five minute candlestick data is provided for one request.",
                },
                "TEN_MINUTE": {
                    "value": "TEN_MINUTE",
                    "description": "Maximum 100 days ten minute candlestick data is provided for one request.",
                },
                "FIFTEEN_MINUTE": {
                    "value": "FIFTEEN_MINUTE",
                    "description": "Maximum 200 days fifteen minute candlestick data is provided for one request.",
                },
                "THIRTY_MINUTE": {
                    "value": "THIRTY_MINUTE",
                    "description": "Maximum 200 days thirty minute candlestick data is provided for one request.",
                },
                "ONE_HOUR": {
                    "value": "ONE_HOUR",
                    "description": "Maximum 400 days one hour candlestick data is provided for one request.",
                },
                "ONE_DAY": {
                    "value": "ONE_DAY",
                    "description": "Maximum 2000 days one day candlestick data is provided for one request.",
                },
            }
        ),
    ],
    start_date: Annotated[str, Query(example="2023-09-08 12:00")],
    end_date: Annotated[str, Query(example="2023-09-09 12:00")],
):
    """
    Get the historical stock data for a given symbol. This endpoint provides
    the historical candle data of the  given stock symbol and candlestick
    interval for a particular time period from an External API in realtime.

    Parameters:
    -----------
    - **stock_symbol**:
        It must be a valid stock symbol that is registered in the NSE website
        eg: `TCS`, `RELIANCE`

    """

    stock_token, stock_symbol = validate_symbol_and_get_token(
        stock_exchange=ExchangeType.NSE, stock_symbol=stock_symbol
    )
    validated_interval = CandlestickInterval.validate_interval(interval)
    validated_start_date, validated_end_date = validate_dates(
        start_date, end_date, validated_interval, stock_symbol.split("-")[0]
    )
    payload = {
        "exchange": ExchangeType.NSE.name,
        "tradingsymbol": stock_symbol,
        "interval": validated_interval.name,
        "fromdate": validated_start_date.strftime("%Y-%m-%d %H:%M"),
        "todate": validated_end_date.strftime("%Y-%m-%d %H:%M"),
        "symboltoken": stock_token,
    }
    json_payload = json.dumps(payload)

    connection = get_endpoint_connection(
        payload=json_payload,
        request_method_type=RequestType.POST,
        url=CANDLE_DATA_URL,
    )
    res = connection.getresponse()
    data = json.loads(res.read().decode("utf-8"))
    if not data["data"]:
        raise UnkownException(error_message=data["message"], status_code=res.status)

    return process_smart_api_historical_stock_data(
        data["data"],
        stock_symbol,
        validated_interval,
        validated_start_date,
        validated_end_date,
    )
