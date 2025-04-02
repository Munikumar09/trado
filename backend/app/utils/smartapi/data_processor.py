from datetime import datetime, time
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.schemas.stock_model import (
    HistoricalStockDataBundle,
    HistoricalStockPriceInfo,
    SmartAPIStockPriceInfo,
)
from app.utils.common.logger import get_logger
from app.utils.common.types.reques_types import CandlestickInterval
from app.utils.smartapi.validator import check_data_availability, find_open_market_days

logger = get_logger(Path(__file__).name)


def process_smart_api_stock_data(
    stock_price_data: dict[str, Any]
) -> SmartAPIStockPriceInfo:
    """
    Processes the data from the SmartAPI and returns the processed data.

    Parameters:
    -----------
    stock_price_data: ``dict[str, Any]``
        The data from the SmartAPI to be processed.

    Returns:
    --------
    ``SmartAPIStockPriceInfo``
        The processed data from the SmartAPI as a SmartAPIStockPriceInfo object.

    """
    return SmartAPIStockPriceInfo(
        symbol=stock_price_data["tradingsymbol"],
        last_traded_price=stock_price_data["ltp"],
        day_open=stock_price_data["open"],
        day_low=stock_price_data["low"],
        day_high=stock_price_data["high"],
        change=stock_price_data["ltp"] - stock_price_data["close"],
        percent_change=(
            (stock_price_data["ltp"] - stock_price_data["close"])
            / stock_price_data["close"]
        )
        * 100,
        symbol_token=stock_price_data["symboltoken"],
        prev_day_close=stock_price_data["close"],
    )


def process_available_stock_data(
    historical_stock_data: tuple[list[list[Any]], str, str],
) -> list[HistoricalStockPriceInfo]:
    """Processes the available data from the SmartAPI and returns the processed data.

    Parameters:
    -----------
    historical_stock_data: ``tuple[list[list[Any]],str,str]``
        The data to be processed.

    Return:
    -------
    ``list[HistoricalStockPriceInfo]``
        Processed data as a HistoricalStockPriceInfo object.
    """
    return [
        HistoricalStockPriceInfo(
            timestamp=stock_data[0],
            open=stock_data[1],
            close=stock_data[2],
            low=stock_data[3],
            high=stock_data[4],
            volume=stock_data[5],
            stock_symbol=historical_stock_data[1],
            candle_interval=historical_stock_data[2],
        )
        for stock_data in historical_stock_data[0]
        if len(stock_data) > 0
    ]


def get_possible_timestamps_on_date(
    current_date: datetime,
    start_datetime: datetime,
    end_datetime: datetime,
    interval: CandlestickInterval,
) -> list[str]:
    """
    Finds the list of all possible timestamps of given interval for the given current
    date from start time which is either 9:15 or time in given start_datetime to end
    time which is either 15:29 or time in given end_datetime.

    Parameters:
    -----------
    current_date: ``datetime``
        The date for which to find the range of timestamps
    start_datetime: ``str``
        The initial datetime from which historical stock data should be retrieved
    end_datetime: ``str``
        The final datetime up to which historical stock data should be retrieved
    interval: ``CandlestickInterval``
        The interval of the candlestick

    Return:
    -------
    ``list[str]``
        The possible timestamps for given current date
    """
    if interval.name == "ONE_DAY":
        return [f"{current_date.strftime('%Y-%m-%d')}T00:00:00+05:30"]

    start_time = time(9, 15)
    end_time = time(15, 29)

    if start_datetime.date() == current_date.date():
        start_time = max(start_time, start_datetime.time())
    if end_datetime.date() == current_date.date():
        end_time = min(end_time, end_datetime.time())

    # Create a datetime range for every given interval from 9:15 to 15:29 on the particular day
    time_range = pd.date_range(
        start=f"{current_date.date()} {start_time.strftime('%H:%M')}:00+05:30",
        end=f"{current_date.date()} {end_time.strftime('%H:%M')}:00+05:30",
        freq=f"{interval.value[0]}min",
    )

    # Convert the datetime range to a list of timestamps of string type
    timestamps = [
        timestamp.strftime("%Y-%m-%dT%H:%M:%S+05:30") for timestamp in time_range
    ]

    return timestamps


def get_missing_timestamps(
    historical_stock_data: list[list[str]],
    stock_symbol: str,
    interval: CandlestickInterval,
    start_datetime: datetime,
    end_datetime: datetime,
) -> list[str]:
    """Finds the missing timestamps of given candlestick interval in the historical stock data
    of a stock between start_datetime and end_datetime.
    Parameters:
    -----------
    historical_stock_data: ``list[list[str]]``
        Available historical stock data points.
    stock_symbol: ``str``
        The symbol of the stock.
    interval: ``CandlestickInterval``
        The interval of the candlestick.
    start_datetime: ``datetime``
        The initial datetime from which to find timestamps of missing data points.
    end_datetime: ``datetime``
        The final datetime up to which to find timestamps of missing data points.

    Return:
    -------
    ``list[str]``
        list of missing timestamps in given historical stock data.
    """
    available_timestamps = pd.DataFrame(historical_stock_data)[0]
    all_possible_timestamps = []
    missing_timestamps = []

    try:
        start_datetime = check_data_availability(
            start_datetime, end_datetime, stock_symbol.split("-")[0], interval
        )
        open_dates = find_open_market_days(start_datetime, end_datetime)

        all_possible_timestamps = list(
            chain.from_iterable(
                get_possible_timestamps_on_date(
                    open_date, start_datetime, end_datetime, interval
                )
                for open_date in open_dates
            )
        )

        missing_timestamps = np.setdiff1d(
            all_possible_timestamps, available_timestamps
        ).tolist()
    except Exception as e:
        logger.error(e)

    return missing_timestamps


def process_smart_api_historical_stock_data(
    historical_stock_data: list[list[Any]],
    stock_symbol: str,
    interval: CandlestickInterval,
    start_datetime: datetime,
    end_datetime: datetime,
) -> HistoricalStockDataBundle:
    """Processes the available data points and find timestamps of missing data points of a stock
    between given start date and end date and returns them as a HistoricalStockDataBundle object.

    Parameters:
    -----------
    historical_stock_data: ``list[list[Any]]``
        Available historical stock data points.
    stock_symbol: ``str``
        The symbol of the stock.
    interval: ``CandlestickInterval``
        The interval of the candlestick.
    start_datetime: ``datetime``
        The initial datetime from which historical stock data should be retrieved.
    end_datetime: ``datetime``
        The final datetime up to which historical stock data should be retrieved.

    Return:
    -------
    ``HistoricalStockDataBundle``
        The processed available data points and timestamps of missing data points as a HistoricalStockDataBundle object.
    """
    processed_available_stock_data = process_available_stock_data(
        (
            historical_stock_data,
            stock_symbol,
            interval.name,
        )
    )
    missing_timestamps = get_missing_timestamps(
        historical_stock_data, stock_symbol, interval, start_datetime, end_datetime
    )

    return HistoricalStockDataBundle(
        available_stock_data=processed_available_stock_data,
        missing_timestamps=missing_timestamps,
    )
