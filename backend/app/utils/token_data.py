from pathlib import Path

import pandas as pd

from app.data_layer.database.models.instrument_model import Instrument
from app.utils.common.logger import get_logger
from app.utils.common.types.financial_types import DataProviderType, ExchangeType
from app.utils.fetch_data import fetch_data
from app.utils.urls import SMARTAPI_TOKENS_URL, UPSTOX_TOKEN_URL

logger = get_logger(Path(__file__).name)


def get_smartapi_token_data() -> list[Instrument]:
    """
    Fetches the token data from the SMARTAPI_TOKENS_URL and processes it to return the processed data.
    It filters the data based on the exchange segment and instrument type and returns the processed data.
    The exchange segment used here is NSE and the instrument types are EQ, SM, BE, ST.
    Read more about the segments or groups in NSE
    from: https://www.nseindia.com/market-data/legend-of-series

    Returns:
    --------
    ``list[Instrument]``
        The processed token data as a list of Instrument objects
    """
    try:
        tokens_data = fetch_data(SMARTAPI_TOKENS_URL)
        df = pd.DataFrame(tokens_data)
        df = df[df["exch_seg"] == "NSE"]
        df = df[df["instrumenttype"] == ""]
        df["instrumenttype"] = df["symbol"].apply(lambda x: x.split("-")[-1])
        df = df[df["instrumenttype"].isin(["EQ", "SM", "BE", "ST"])]
        df["symbol"] = df["symbol"].apply(lambda x: x.split("-")[0])

        return [
            Instrument(
                token=token["token"],
                exchange_id=ExchangeType.NSE.value,
                data_provider_id=DataProviderType.SMARTAPI.value,
                symbol=token["symbol"],
                name=token["name"],
                instrument_type=token["instrumenttype"],
                expiry_date=token["expiry"],
                strike_price=token["strike"],
                lot_size=token["lotsize"],
                tick_size=token["tick_size"],
            )
            for token in df.to_dict("records")
        ]
    except Exception as e:
        logger.error("Error in get_smartapi_token_data: %s", e)
        return []


def get_uplink_token_data() -> list[Instrument]:
    """
    Fetches the token data from the UPSTOX_TOKEN_URL and processes it to return the processed data.
    It filters the data based on the segment and instrument type and returns the processed data.
    The segment used here is BSE_EQ and the instrument types are A, B, X, T, XT, M, MT.
    Read more about the segments or groups in BSE
    from: https://www.bseindia.com/markets/equity/EQReports/tra_trading.aspx

    Returns:
    --------
    ``list[Instrument]``
        The processed token data as a list of Instrument objects
    """
    try:
        df = pd.read_json(UPSTOX_TOKEN_URL, compression="gzip")
        df = df[df["segment"] == "BSE_EQ"]
        df = df[df["instrument_type"].isin(["A", "B", "X", "T", "XT", "M", "MT"])]
        df["instrument_type"] = df["segment"].apply(lambda x: x.split("_")[1])

        return [
            Instrument(
                token=token["instrument_key"],
                exchange_id=ExchangeType.BSE.value,
                data_provider_id=DataProviderType.UPLINK.value,
                symbol=token["trading_symbol"],
                name=token["name"],
                instrument_type=token["instrument_type"],
                lot_size=token["lot_size"],
                tick_size=token["tick_size"],
            )
            for token in df.to_dict("records")
        ]
    except Exception as e:
        logger.error("Error in process_upstox_token_data: %s", e)
        return []


def get_token_data(provider: DataProviderType) -> list[Instrument]:
    """
    This method is used to get the token data based on the data provider type.

    Parameters:
    ----------
    provider: ``DataProviderType``
        The data provider type based on which the token data should be processed

    Returns:
    --------
    ``list[Instrument]``
        The processed token data as a list of Instrument objects
    """
    provider_map = {
        DataProviderType.SMARTAPI: get_smartapi_token_data,
        DataProviderType.UPLINK: get_uplink_token_data,
    }

    fetch_func = provider_map.get(provider, lambda: [])

    return fetch_func()
