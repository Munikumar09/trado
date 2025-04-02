from abc import ABC, abstractmethod
from typing import Optional

from omegaconf import DictConfig
from registrable import Registrable

from app.sockets.twisted_socket import MarketDataTwistedSocket
from app.utils.common.types.financial_types import ExchangeType


class WebsocketConnection(ABC, Registrable):
    """
    This is the base class for all the websocket connections. It provides the interface
    for the websocket connections to implement. The websocket connections are responsible
    for creating a connection to the respective websocket servers and subscribing to the
    required tokens to get the live data for the tokens from server. The subclasses of this
    class should implement the `get_tokens` method to return the tokens for the respective
    exchange and instrument type. The subclasses should also has a class method `from_cfg`
    to create the object from the configuration to abstract complex object creation logic.

    Attributes
    ----------
    websocket: ``MarketDatasetTwistedSocket``
        The websocket object to connect to the respective websocket
    """

    def __init__(self, websocket: MarketDataTwistedSocket):
        self.websocket = websocket

    @classmethod
    @abstractmethod
    def _get_tokens(
        cls,
        symbols: str | list[str] | None = None,
        exchange: ExchangeType = ExchangeType.NSE,
    ) -> dict[str, str]:
        """
        This method should be implemented by the subclasses to get the tokens for the
        respective exchange and given symbols. This method should return a dictionary
        containing the token and symbol mappings.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_cfg(cls, cfg: DictConfig) -> Optional["WebsocketConnection"]:
        """
        This method creates the object of the websocket connection from the configuration.
        """
        raise NotImplementedError
