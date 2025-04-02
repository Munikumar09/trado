# pylint: disable=no-value-for-parameter

from pathlib import Path
from typing import Optional

from omegaconf import DictConfig

from app.data_layer.database.crud.crud_utils import get_data_by_all_conditions
from app.data_layer.database.models import Instrument
from app.data_layer.streaming.streamer import Streamer
from app.sockets.connections.websocket_connection import WebsocketConnection
from app.sockets.twisted_sockets import UplinkSocket
from app.utils.common import init_from_cfg
from app.utils.common.logger import get_logger
from app.utils.common.types.financial_types import DataProviderType, ExchangeType

logger = get_logger(Path(__file__).name)


@WebsocketConnection.register("uplinksocket_connection")
class UplinkSocketConnection(WebsocketConnection):
    """
    This class is responsible for creating a connection to the UplinkSocket.
    It creates a connection to the UplinkSocket and subscribes to the tokens
    provided in the configuration.
    """

    @classmethod
    def _get_tokens(
        cls,
        symbols: str | list[str] | None = None,
        exchange: ExchangeType = ExchangeType.BSE,
    ) -> dict[str, str]:
        """
        Fetches the token-symbol mapping from the database based on the symbols provided.
        If no symbols are provided, it fetches all tokens from the SmartAPI data provider.

        Parameters
        ----------
        symbols: ``str | list[str] | None``, (defaults to None)
            List of symbols or a single symbol. If None, fetches all available tokens.
        exchange: ``ExchangeType``, (defaults to ExchangeType.BSE)
            The exchange for which the tokens are needed.

        Returns
        -------
        ``dict[str, str]``
            A dictionary containing token-symbol mappings. Eg: {"256265": "INFY"}
        """
        try:
            if symbols:
                if isinstance(symbols, str):
                    symbols = [symbols]
                instruments = [
                    get_data_by_all_conditions(
                        Instrument, symbol=symbol.upper(), exchange_id=exchange.value
                    )
                    for symbol in symbols
                ]
                instruments = [inst[0] for inst in instruments if inst]
            else:
                instruments = get_data_by_all_conditions(
                    Instrument, data_provider_id=DataProviderType.UPLINK.value
                )

            token_map = {inst.token: inst.symbol for inst in instruments}
            missing_symbols = set(symbols) - set(token_map.values()) if symbols else []

            if missing_symbols:
                logger.warning(
                    "Invalid symbols discarded: %s", ", ".join(missing_symbols)
                )

            return token_map

        except Exception as e:
            logger.error("Failed to fetch tokens: %s", str(e))
            return {}

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Optional["UplinkSocketConnection"]:
        try:
            exchange = ExchangeType.get_exchange(cfg.exchange_type)

            if exchange is None:
                logger.error("Invalid exchange type: %s", cfg.exchange_type)
                return None

            # Get the tokens to subscribe to
            tokens = cls._get_tokens(cfg.symbols, exchange)
            tokens = dict(tuple(tokens.items())[: cfg.num_tokens_per_instance])

            # If there are no tokens to subscribe to, log an error and return None.
            if not tokens:
                logger.error(
                    "Instance %d has no tokens to subscribe to, exiting...",
                    0,
                )
                return None

            # Initialize the callback to save the received data from the socket.
            save_data_callback = init_from_cfg(cfg.streaming, Streamer)

            smart_socket = UplinkSocket.initialize_socket(
                cfg.provider, save_data_callback
            )
            connection = cls(smart_socket)

            smart_socket.set_tokens(tokens)

            return connection
        except Exception as e:
            logger.error("Failed to initialize UplinkSocketConnection: %s", str(e))
            return None
