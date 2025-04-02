import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

from app.utils.common.logger import get_logger
from app.utils.common.types.reques_types import CandlestickInterval
from app.utils.file_utils import create_dir, load_json_data, write_to_json_file
from tools.data_collector_tool.smartapi.constants import HISTORICAL_STOCK_DATA_URL

logger = get_logger(Path(__file__).name)


def get_historical_stock_data_url(
    stock_symbol: str, interval: str, start_date: str, end_date: str
) -> str:
    """
    Provides the url to the historical stock data endpoint.

    Parameters:
    -----------
    stock_symbol: ``str``
        The symbol of the stock
    interval: ``str``
        The interval of the candlestick
    start_date: ``str``
        The initial datetime from which historical stock data should be retrieved
    end_date: ``str``
        The final datetime up to which historical stock data should be retrieved

    Return:
    -------
    ``str``
        Url to the historical stock data endpoint.
    """
    return (
        f"{HISTORICAL_STOCK_DATA_URL}{stock_symbol}?interval={interval}"
        f"&start_date={start_date}&end_date={end_date}"
    )


def search_valid_date(
    start_date: datetime,
    end_date: datetime,
    stock_symbol: str,
    interval: CandlestickInterval,
) -> datetime:
    """It finds the valid month from where the availability of data starts for
    the given stock symbol and interval by using binary search method.

    Parameters:
    -----------
    start_date: ``datetime``
        Start date to search
    end_date: ``datetime``
        End date to search
    stock_symbol: ``str``
        The symbol of the stock

    Return:
    -------
    ``datetime``
        searched month from where the availability of data starts for the given
        stock symbol and interval.
    """
    valid_date = end_date

    with httpx.Client() as client:
        while start_date <= end_date:
            try:
                total_days = (end_date - start_date).days
                middle_date = start_date + timedelta(days=total_days // 2)
                first_day = middle_date.replace(day=3)
                last_day = (middle_date + timedelta(days=31)).replace(
                    day=1
                ) - timedelta(days=1)

                stocks_url = get_historical_stock_data_url(
                    stock_symbol,
                    interval.name,
                    f"{first_day.strftime('%Y-%m-%d')} 09:15",
                    f"{last_day.strftime('%Y-%m-%d')} 15:29",
                )
                response = client.get(stocks_url, timeout=60.0)

                if response.status_code == 200 and response.json():
                    end_date = first_day - timedelta(days=3)
                    valid_date = first_day.replace(day=1)
                else:
                    start_date = last_day + timedelta(days=1)
                time.sleep(0.3)
            except Exception as e:
                logger.error(e)
                start_date = last_day + timedelta(days=1)
                continue

    return valid_date


def dataframe_to_json_files(
    df: pd.DataFrame, dir_path: Path, interval: CandlestickInterval
):
    """
    Process the given dataframe and convert it into suitable data structure
    i.e dictionary which will be stored in json file.

    Parameters:
    -----------
    df: ``pd.DataFrame``
        pandas DataFrame to store into json files
    dir_path: ``str``
        Path of the destination directory to store json files
    """
    # Convert timestamp to datetime and extract year and day
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year
    df["day"] = df["timestamp"].dt.strftime("%Y-%m-%d")

    if interval.name == "ONE_DAY":
        grouped = df.groupby("year")

        # Iterate over each group
        for year, group in grouped:

            # Prepare data for JSON
            data = (
                group.set_index("day")
                .drop(columns=["year", "timestamp"])
                .to_dict(orient="index")
            )
            json_file_path = dir_path / f"{year}.json"

            if json_file_path.exists():
                stored_data = load_json_data(json_file_path)
                stored_data.update(data)

            write_to_json_file(json_file_path, stored_data)
    else:
        df["time"] = df["timestamp"].dt.strftime("%H:%M")

        # Group by year and day
        grouped = df.groupby(["year", "day"])

        # Iterate over each group
        for (year, day), group in grouped:

            # Create directory for the year if it doesn't exist
            year_dir = create_dir(dir_path / f"{year}")

            # Prepare data for JSON
            data = (
                group.set_index("time")
                .drop(columns=["year", "day", "timestamp"])
                .to_dict(orient="index")
            )

            # Write to JSON file
            json_file_path = year_dir / f"{day}.json"
            write_to_json_file(json_file_path, data)
