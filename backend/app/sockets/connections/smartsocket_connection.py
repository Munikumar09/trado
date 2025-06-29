# pylint: disable=no-value-for-parameter
from itertools import islice
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig

from app.data_layer.database.crud.crud_utils import get_data_by_all_conditions
from app.data_layer.database.models import Instrument
from app.data_layer.streaming.producer import Producer
from app.sockets.connections.websocket_connection import WebsocketConnection
from app.sockets.twisted_sockets import SmartSocket
from app.utils.common import init_from_cfg
from app.utils.common.logger import get_logger
from app.utils.common.types.financial_types import DataProviderType, ExchangeType
from app.utils.smartapi.smartsocket_types import EXCHANGETYPE_SMARTAPI_MAP

logger = get_logger(Path(__file__).name)


@WebsocketConnection.register("smartsocket_connection")
class SmartSocketConnection(WebsocketConnection):
    """
    This class is responsible for creating a connection to the SmartSocket.
    It creates a connection to the SmartSocket and subscribes to the tokens
    provided in the configuration.
    """

    @classmethod
    def _get_tokens(
        cls,
        symbols: str | list[str] | None = None,
        exchange: ExchangeType = ExchangeType.NSE,
    ) -> dict[str, str]:
        """
        Fetches the token-symbol mapping from the database based on the symbols provided.
        If no symbols are provided, it fetches all tokens from the SmartAPI data provider.

        Parameters
        ----------
        symbols: ``str | list[str] | None``, (defaults to None)
            List of symbols or a single symbol. If None, fetches all available tokens.
        exchange: ``ExchangeType``, (defaults to ExchangeType.NSE)
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
                    Instrument, data_provider_id=DataProviderType.SMARTAPI.value
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
    def from_cfg(cls, cfg: DictConfig) -> Optional["SmartSocketConnection"]:
        """
        Create a SmartSocketConnection instance from a configuration object.
        
        Initializes the connection by determining the appropriate exchange, retrieving and partitioning tokens for the current instance, updating the provider's correlation ID, setting up the streaming data callback, and configuring the SmartSocket with the relevant tokens. Returns None if initialization fails due to invalid configuration or other errors.
        
        Parameters:
            cfg (DictConfig): Configuration object containing connection, provider, and streaming settings.
        
        Returns:
            Optional[SmartSocketConnection]: The initialized SmartSocketConnection instance, or None if initialization fails.
        """
        try:
            connection_instance_num = cfg.get("current_connection_number", 0)
            num_tokens_per_instance = cfg.get("num_tokens_per_instance", 1000)

            # Get tokens before initializing SmartSocket
            exchange = ExchangeType.get_exchange(cfg.exchange_type)

            if exchange is None:
                logger.error("Invalid exchange type: %s, exiting...", cfg.exchange_type)
                return None

            tokens = cls._get_tokens(cfg.symbols, exchange)
            tokens = dict(
                islice(
                    tokens.items(),
                    connection_instance_num * num_tokens_per_instance,
                    (connection_instance_num + 1) * num_tokens_per_instance,
                )
            )

            if not tokens:
                logger.error(
                    "Instance %d has no tokens to subscribe to, exiting...",
                    connection_instance_num,
                )
                return None  # Exit early to avoid unnecessary initialization

            # Generate unique correlation ID per instance
            correlation_id = cfg.provider.correlation_id.replace(
                "_", str(connection_instance_num)
            )
            cfg.provider.correlation_id = correlation_id

            # Initialize callback function for streaming data
            save_data_callback = init_from_cfg(cfg.streaming, Producer)

            # Initialize SmartSocket only after confirming tokens exist
            smart_socket = SmartSocket.initialize_socket(
                cfg.provider, save_data_callback
            )
            connection = cls(smart_socket)

            tokens_list = [
                {
                    "exchangeType": EXCHANGETYPE_SMARTAPI_MAP[exchange].value,
                    "tokens": tokens,
                }
            ]
            smart_socket.set_tokens(tokens_list)

            return connection

        except Exception as e:
            logger.error("Failed to initialize SmartSocketConnection: %s", str(e))
            return None
