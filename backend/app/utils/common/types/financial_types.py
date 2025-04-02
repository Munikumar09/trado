from enum import Enum
from pathlib import Path
from typing import Optional

from app.utils.common.logger import get_logger

logger = get_logger(Path(__file__).name)


class DataProviderType(Enum):
    """
    DataProviders enumeration class to define the data providers in the system.
    """

    SMARTAPI = 1
    UPLINK = 2

    @classmethod
    def get_data_provider(
        cls, data_provider_symbol: str | int, verbose: bool = False
    ) -> Optional["DataProviderType"]:
        """
        This method is used to get the data provider type based on the data provider symbol or value.

        Parameters
        ----------
        data_provider_symbol: ``str`` | ``int``
            The data provider symbol or value

        Returns
        -------
        ``DataProvider``
            The data provider type based on the data provider symbol or value
        """
        try:
            if isinstance(data_provider_symbol, int):
                return cls(data_provider_symbol)

            return cls[data_provider_symbol.upper()]
        except Exception:
            if verbose:
                logger.error(
                    "Invalid data provider symbol or value [data_provider_symbol=%s]. possible values are: %s",
                    data_provider_symbol,
                    {data_provider.name: data_provider.value for data_provider in cls},
                )
            return None


class ExchangeType(Enum):
    """
    Exchanges enumeration class to define the exchanges in the system.
    """

    NSE = 1
    BSE = 2
    MCX = 3
    NFO = 4
    BFO = 5

    @classmethod
    def get_exchange(
        cls, exchange_symbol: str | int, verbose: bool = False
    ) -> Optional["ExchangeType"]:
        """
        This method is used to get the exchange type based on the exchange symbol or value.

        Parameters
        ----------
        exchange_symbol: ``str`` | ``int``
            The exchange symbol or value

        Returns
        -------
        ``ExchangeType``
            The exchange type based on the exchange symbol or value
        """
        try:
            if isinstance(exchange_symbol, int):
                return cls(exchange_symbol)

            return cls[exchange_symbol.upper()]
        except Exception:
            if verbose:
                logger.error(
                    "Invalid exchange symbol or value [exchange_symbol=%s]. possible values are: %s",
                    exchange_symbol,
                    {exchange.name: exchange.value for exchange in cls},
                )
            return None


class SecurityType(Enum):
    """
    SecurityType enumeration class to define the security types in the system.
    """

    EQUITY = "equity"
    DERIVATIVE = "derivative"
