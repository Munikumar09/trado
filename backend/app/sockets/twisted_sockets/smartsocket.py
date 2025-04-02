# pylint: disable=too-many-arguments, too-many-instance-attributes
import json
import struct
import time
from pathlib import Path
from typing import Any, cast

from app.sockets.twisted_socket import MarketDataTwistedSocket
from app.sockets.websocket_client_protocol import MarketDataWebSocketClientProtocol
from app.utils.common.logger import get_logger
from app.utils.common.types.financial_types import DataProviderType
from app.utils.smartapi.connection import SmartApiConnection
from app.utils.smartapi.smartsocket_types import (
    SMARTAPI_EXCHANGETYPE_MAP,
    SmartAPIExchangeSegment,
    SubscriptionAction,
    SubscriptionMode,
)

logger = get_logger(Path(__file__).name, log_level="DEBUG")


class SmartSocket(MarketDataTwistedSocket):
    """
    SmartSocket is a class that connects to the SmartAPI WebSocket server and subscribes
    to the specified tokens. It receives the data for the subscribed tokens and parses
    the data to extract the required information for the subscribed tokens. The parsed
    data is then sent to the callback function for further processing or saving.

    Attributes
    ----------
    auth_token: ``str``
        The authorization token for the SmartAPI WebSocket server
    api_key: ``str``
        The API key for the SmartAPI WebSocket server connection
    client_code: ``str``
        The client code is the angel broking client id which is used to
        login to the angel broking account
    feed_token: ``str``
        The feed token is used to authenticate the user to the SmartAPI WebSocket server
    correlation_id: ``str``
        The correlation id is used to uniquely identify the WebSocket connection which
        is useful for debugging and logging purposes in multi-connection scenarios
    subscription_mode: ``SubscriptionMode``
        The subscription mode is used to specify the type of data to receive from
        the WebSocket server. The subscription mode can be either "quote", "snap_quote",
        or "full"
    on_data_save_callback: ``Callable[[str], None]``, ( default = None )
        The callback function that is called when the data is received from the
        WebSocket server
    debug: ``bool``, ( default = False )
        A flag to enable or disable the debug mode for the WebSocket connection.
        Enable this flag in development mode to get detailed logs for debugging
        purposes
    ping_interval: ``int``, ( default = 25 )
        The interval in seconds at which the ping message should be sent to the
        WebSocket server to keep the connection alive
    ping_message: ``str``, ( default = "ping" )
        The message to send to the WebSocket server to keep the connection alive
    """

    def __init__(
        self,
        auth_token: str,
        api_key: str,
        client_code: str,
        feed_token: str,
        correlation_id: str,
        subscription_mode: SubscriptionMode,
        on_data_save_callback,
        debug,
        ping_interval,
        ping_message,
    ):
        super().__init__(
            ping_interval=ping_interval, ping_message=ping_message, debug=debug
        )

        self.websocket_url = "wss://smartapisocket.angelone.in/smart-stream"
        self.little_endian_byte_order = "<"
        self.token_map: dict[str, tuple[str, SmartAPIExchangeSegment]] = {}

        self.headers = {
            "Authorization": auth_token,
            "x-api-key": api_key,
            "x-client-code": client_code,
            "x-feed-token": feed_token,
        }

        self.sanity_check()
        self.subscription_mode = subscription_mode
        self.correlation_id = correlation_id
        self.on_data_save_callback = on_data_save_callback
        self.subscribed_tokens: dict[str, int] = {}
        self._tokens: list[dict[str, int | list[str]]] = []

    def sanity_check(self):
        """
        Check if the headers are set correctly and raise an exception if
        any of the headers are empty.
        """
        for key, value in self.headers.items():
            assert value, f"{key} is empty"

    def set_tokens(
        self,
        token_data: (
            dict[str, int | dict[str, str]] | list[dict[str, int | dict[str, str]]]
        ),
    ):
        """
        Set the tokens to subscribe to the WebSocket connection.

        Parameters
        ----------
        tokens_with_exchanges: ``dict[str, int | dict[str, str]] | list[dict[str, int | dict[str, str]]]``
            A list of dictionaries containing the exchange type and the tokens to subscribe
            e.g., [{"exchangeType": 1, "tokens": {"token1": "name1", "token2": "name2"}}]
                or {"exchangeType": 1, "tokens": {"token1": "name1", "token2": "name2"}}
        """
        if isinstance(token_data, dict):
            token_data = [token_data]

        for token_exchange_info in token_data:
            # Check if the token exchange info is in the correct format
            if (
                "exchangeType" not in token_exchange_info
                or "tokens" not in token_exchange_info
                or not token_exchange_info["tokens"]
                or not token_exchange_info["exchangeType"]
            ):
                logger.error(
                    "Invalid token format %s, skipping token", token_exchange_info
                )
                continue

            exchange_type = SmartAPIExchangeSegment.get_exchange(
                cast(int, token_exchange_info["exchangeType"])
            )

            self._tokens.append(
                {
                    "exchangeType": exchange_type.value,
                    "tokens": list(cast(dict, token_exchange_info["tokens"]).keys()),
                }
            )
            self.token_map.update(
                {
                    k: (v, exchange_type)
                    for k, v in cast(dict, token_exchange_info["tokens"]).items()
                }
            )

    def _on_open(self, ws: MarketDataWebSocketClientProtocol):
        """
        This function is called when the WebSocket connection is opened.
        It sends a ping message to the server to keep the connection alive.
        When the connection is open, it also resubscribes to the tokens if
        it is not the first connection. If it is the first connection, it
        subscribes to the tokens.

        Parameters
        ----------
        ws: ``MarketDataWebSocketClientProtocol``
            The WebSocket client protocol object
        """
        if self.debug:
            logger.info("on open : %s", ws.state)

        if not self._is_first_connect:
            self.resubscribe()
        else:
            self.subscribe(self._tokens)

        self._is_first_connect = False

    def subscribe(self, subscription_data: list[dict[str, int | list[str]]]):
        """
        Subscribe to the specified tokens on the WebSocket connection.
        After subscribing, the WebSocket connection will receive data
        for the specified tokens. Based on the subscription mode, the
        received data will be different.
        Ref: https://smartapi.angelbroking.com/docs/WebSocket2

        Parameters
        ----------
        subscription_data: ``[dict[str, int | list[str]]]``
            A list of dictionaries containing the exchange type and the tokens to subscribe
            e.g., [{"exchangeType": 1, "tokens": ["token1", "token2"]},
                   {"exchangeType": 2, "tokens": ["token3", "token4"]}]
        """

        if self.debug:
            logger.debug("Subscribing to tokens: %s", subscription_data)

        if not subscription_data:
            logger.error("No tokens to subscribe")
            return False

        request_data = {
            "correlationID": self.correlation_id,
            "action": SubscriptionAction.SUBSCRIBE.value,
            "params": {
                "mode": self.subscription_mode.value,
                "tokenList": subscription_data,
            },
        }
        try:
            if self.ws:
                self.ws.sendMessage(json.dumps(request_data).encode("utf-8"))
            else:
                logger.error("WebSocket connection is not open")

            for tokens_with_exchange in subscription_data:
                for token in cast(list, tokens_with_exchange["tokens"]):
                    self.subscribed_tokens[token] = cast(
                        int, tokens_with_exchange["exchangeType"]
                    )

            return True
        except Exception as e:
            logger.error("Error while sending message: %s", e)
            self._close(reason=f"Error while sending message: {e}")
            raise e

    def unsubscribe(self, unsubscribe_data: list[str]):
        """
        Unsubscribe the specified tokens from the WebSocket connection.
        Once unsubscribed, the WebSocket connection will no longer receive
        data for the specified tokens.

        Parameters
        ----------
        tokens_to_unsubscribe: ``list[str]``
            A list of tokens to unsubscribe from the WebSocket connection
        """

        if not unsubscribe_data:
            logger.error("No tokens to unsubscribe")
            return False

        subscribed_tokens = []
        tokens_with_exchange: dict[int, list[str]] = {}
        tokens_not_subscribed = []

        for token in unsubscribe_data:
            exchange_type = self.subscribed_tokens.get(token)
            if exchange_type:
                subscribed_tokens.append(token)
                tokens_with_exchange.setdefault(exchange_type, []).append(token)
            else:
                tokens_not_subscribed.append(token)

        if tokens_not_subscribed:
            logger.error("Tokens not subscribed: %s", tokens_not_subscribed)

        tokens_to_unsubscribe = [
            {"exchangeType": exchange, "tokens": token_list}
            for exchange, token_list in tokens_with_exchange.items()
        ]

        request_data = {
            "correlationId": self.correlation_id,
            "action": SubscriptionAction.UNSUBSCRIBE.value,
            "params": {
                "mode": self.subscription_mode.value,
                "exchange": tokens_to_unsubscribe,
            },
        }

        try:
            if self.debug:
                logger.debug("Unsubscribing from tokens: %s", unsubscribe_data)

            if self.ws:
                self.ws.sendMessage(json.dumps(request_data).encode("utf-8"))
            else:
                logger.error("WebSocket connection is not open")

            for token in subscribed_tokens:
                self.subscribed_tokens.pop(token)

            return True
        except Exception as e:
            logger.error("Error while sending message to unsubscribe tokens: %s", e)
            self._close(reason=f"Error while sending message: {e}")
            raise

    def resubscribe(self):
        """
        Resubscribes to all previously subscribed tokens. It groups tokens by their
        exchange type and then  calls the subscribe method with the grouped tokens
        """
        tokens_with_exchange = {}

        for token, exchange_type in self.subscribed_tokens.items():
            tokens_with_exchange.setdefault(exchange_type, []).append(token)

        tokens_list = [
            {"exchangeType": exchange_type, "tokens": tokens}
            for exchange_type, tokens in tokens_with_exchange.items()
        ]

        if self.debug:
            logger.debug("Resubscribing to tokens: %s", tokens_list)

        return self.subscribe(tokens_list)

    def _unpack_data(self, binary_data, start, end, byte_format="I") -> tuple:
        """
        Unpack Binary Data to the integer according to the specified byte_format.
        This function returns the tuple
        """
        return struct.unpack(
            self.little_endian_byte_order + byte_format, binary_data[start:end]
        )

    def decode_data(self, binary_data: bytes) -> dict[str, Any]:
        """
        Parses binary data received from the websocket and returns a dictionary
        containing the parsed data.

        Parameters
        ----------
        binary_data: ``bytes``
            The binary data received from the websocket

        Returns
        -------
        parsed_data: ``dict[str, Any]``
            A dictionary containing the parsed data
        """

        parsed_data = {
            "subscription_mode": self._unpack_data(binary_data, 0, 1, byte_format="B")[
                0
            ],
            "token": binary_data[2:27].decode("utf-8").replace("\x00", ""),
            "sequence_number": self._unpack_data(binary_data, 27, 35, byte_format="q")[
                0
            ],
            "exchange_timestamp": self._unpack_data(
                binary_data, 35, 43, byte_format="q"
            )[0],
            "last_traded_price": self._unpack_data(
                binary_data, 43, 51, byte_format="q"
            )[0],
        }
        try:
            parsed_data["subscription_mode_val"] = (
                SubscriptionMode.get_subscription_mode(
                    parsed_data["subscription_mode"]
                ).name
            )

            if parsed_data["subscription_mode"] in [
                SubscriptionMode.QUOTE.value,
                SubscriptionMode.SNAP_QUOTE.value,
            ]:
                parsed_data["last_traded_quantity"] = self._unpack_data(
                    binary_data, 51, 59, byte_format="q"
                )[0]
                parsed_data["average_traded_price"] = self._unpack_data(
                    binary_data, 59, 67, byte_format="q"
                )[0]
                parsed_data["volume_trade_for_the_day"] = self._unpack_data(
                    binary_data, 67, 75, byte_format="q"
                )[0]
                parsed_data["total_buy_quantity"] = self._unpack_data(
                    binary_data, 75, 83, byte_format="d"
                )[0]
                parsed_data["total_sell_quantity"] = self._unpack_data(
                    binary_data, 83, 91, byte_format="d"
                )[0]
                parsed_data["open_price_of_the_day"] = self._unpack_data(
                    binary_data, 91, 99, byte_format="q"
                )[0]
                parsed_data["high_price_of_the_day"] = self._unpack_data(
                    binary_data, 99, 107, byte_format="q"
                )[0]
                parsed_data["low_price_of_the_day"] = self._unpack_data(
                    binary_data, 107, 115, byte_format="q"
                )[0]
                parsed_data["closed_price"] = self._unpack_data(
                    binary_data, 115, 123, byte_format="q"
                )[0]

            if parsed_data["subscription_mode"] == SubscriptionMode.SNAP_QUOTE.value:
                parsed_data["last_traded_timestamp"] = self._unpack_data(
                    binary_data, 123, 131, byte_format="q"
                )[0]
                parsed_data["open_interest"] = self._unpack_data(
                    binary_data, 131, 139, byte_format="q"
                )[0]
                parsed_data["open_interest_change_percentage"] = self._unpack_data(
                    binary_data, 139, 147, byte_format="q"
                )[0]
        except Exception as e:
            logger.exception("Error in parsing binary data: %s", e)

        return parsed_data

    def _on_message(
        self,
        ws: MarketDataWebSocketClientProtocol | None,
        payload: bytes | str,
        is_binary: bool,
    ):
        """
        This method is called whenever a message is received on the WebSocket
        connection. It decodes the payload, enriches the data with additional
        information, and triggers the data save callback if one is set.

        Parameters
        ----------
        ws: ``MarketDataWebScoketClientProtocol``
           The websocket client protocol instance
        payload: ``bytes | str``
            The raw message payload received from the WebSocket
        is_binary: ``bool``
            Flag indicating whether the payload is binary data
        """
        if is_binary:
            data = self.decode_data(cast(bytes, payload))
        else:
            data = json.loads(payload)

        data["symbol"] = self.token_map[data["token"]][0]
        data["retrieval_timestamp"] = str(time.time())
        data["data_provider_id"] = DataProviderType.SMARTAPI.value
        data["exchange_id"] = SMARTAPI_EXCHANGETYPE_MAP[
            self.token_map[data["token"]][1]
        ].value

        if self.debug:
            logger.debug("Received data: %s", data)

        if self.on_data_save_callback:
            self.on_data_save_callback(json.dumps(data))

    @staticmethod
    def initialize_socket(cfg, on_save_data_callback=None):
        """
        Initialize the SmartSocket connection with the specified configuration.
        """
        smartapi_connection = SmartApiConnection.get_connection()
        auth_token = smartapi_connection.get_auth_token()
        feed_token = smartapi_connection.api.getfeedToken()
        api_key = smartapi_connection.credentials.api_key
        client_code = smartapi_connection.credentials.client_id

        return SmartSocket(
            auth_token,
            api_key,
            client_code,
            feed_token,
            cfg.get("correlation_id", None),
            SubscriptionMode.get_subscription_mode(
                cfg.get("subscription_mode", "snap_quote")
            ),
            on_save_data_callback,
            debug=cfg.get("debug", False),
            ping_interval=cfg.get("ping_interval", 25),
            ping_message=cfg.get("ping_message", "ping"),
        )
