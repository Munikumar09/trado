# pylint: disable=too-many-arguments, too-many-instance-attributes, no-member
import json
import time
from pathlib import Path
from typing import Any, Callable, cast

import requests
from google.protobuf.json_format import MessageToDict

import app.sockets.twisted_sockets.uplink_data_decoder as decoder
from app.sockets.twisted_socket import MarketDataTwistedSocket
from app.sockets.websocket_client_protocol import MarketDataWebSocketClientProtocol
from app.utils.common.logger import get_logger
from app.utils.common.types.financial_types import DataProviderType, ExchangeType
from app.utils.credentials.uplink_credentials import UplinkCredentials
from app.utils.urls import UPLINK_WEBSOCKET_AUTH_URL

logger = get_logger(Path(__file__).name, log_level="DEBUG")


class UplinkSocket(MarketDataTwistedSocket):
    """
    UplinkSocket is a class that connects to the Uplink WebSocket server and subscribes
    to the specified tokens. It receives the data for the subscribed tokens and parses
    the data to extract the required information for the subscribed tokens. The parsed
    data is then sent to the callback function for further processing or saving.

    Attributes
    ----------
    websocket_url: ``str``
        The WebSocket URL for the Uplink WebSocket server connection
    guid: ``str``
        The guid id is used to uniquely identify the WebSocket connection which
        is useful for debugging and logging purposes in multi-connection scenarios
    subscription_mode: ``str``
        The subscription mode is used to specify the type of data to receive from
        the WebSocket server. The subscription mode can be either "ltpc", "option_geeks"
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
        websocket_url: str,
        guid: str,
        subscription_mode: str,
        on_data_save_callback: Callable[[str], None] | None,
        debug: bool,
        ping_interval: int,
        ping_message: str,
    ):
        super().__init__(
            ping_interval=ping_interval, ping_message=ping_message, debug=debug
        )
        self.websocket_url = websocket_url
        self.token_map: dict[str, str] = {}
        self.headers = {
            "accept": "application/json",
        }

        self.subscription_mode = subscription_mode
        self.guid = guid
        self.on_data_save_callback = on_data_save_callback
        self.subscribed_tokens: dict[str, str] = {}
        self._tokens: list[str] = []

    def set_tokens(
        self,
        token_data: dict[str, str],
    ):
        """
        Set the tokens to subscribe to the WebSocket connection.

        Parameters
        ----------
        token_data: ``dict[str, str]``
            A dictionaries containing the tokens and their symbols to subscribe
            eg:- {"token1": "name1", "token2": "name2"}
        """
        self._tokens = list(token_data.keys())
        self.token_map = token_data.copy()

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

    def subscribe(self, subscription_data: list[str]):
        """
        Subscribe to the specified tokens on the WebSocket connection.
        After subscribing, the WebSocket connection will receive data
        for the specified tokens. Based on the subscription mode, the
        received data will be different.
        Ref: https://upstox.com/developer/api-documentation/v3/get-market-data-feed

        Parameters
        ----------
        subscription_data: ``list[str]``
            A list of dictionaries containing the exchange type and the tokens to subscribe
            e.g., ["token1", "token2", ...]
        """
        valid_tokens, invalid_tokens = [], []

        for token in subscription_data:
            if token in self.token_map:
                valid_tokens.append(token)
            else:
                invalid_tokens.append(token)

        if invalid_tokens:
            logger.error(
                "Tokens not found in token map: %s. Please set tokens using set_tokens method",
                invalid_tokens,
            )

        if self.debug:
            logger.debug("Subscribing to tokens: %s", valid_tokens)

        if not valid_tokens:
            logger.error("No valid tokens to subscribe")
            return False

        request_data = {
            "guid": self.guid,
            "method": "sub",
            "data": {
                "mode": self.subscription_mode,
                "instrumentKeys": valid_tokens,
            },
        }

        try:
            if self.ws:
                self.ws.sendMessage(
                    json.dumps(request_data).encode("utf-8"), isBinary=True
                )
                self.subscribed_tokens.update(
                    {token: self.token_map[token] for token in valid_tokens}
                )
                return True

            logger.error("WebSocket connection is not open")
            return False

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
        unsubscribe_data: ``list[str]``
            A list of tokens to unsubscribe from the WebSocket connection
        """

        if not unsubscribe_data:
            logger.error("No tokens to unsubscribe")
            return False

        subscribed_tokens = []
        tokens_not_subscribed = []

        for token in unsubscribe_data:
            if token in self.subscribed_tokens:
                subscribed_tokens.append(token)
            else:
                tokens_not_subscribed.append(token)

        if tokens_not_subscribed:
            logger.error("Tokens not subscribed: %s", tokens_not_subscribed)

        if not subscribed_tokens:
            logger.warning("No subscribed tokens to unsubscribe in the list")
            return False

        request_data = {
            "guid": self.guid,
            "method": "unsub",
            "data": {
                "mode": self.subscription_mode,
                "instrumentKeys": subscribed_tokens,
            },
        }
        try:
            if self.debug:
                logger.debug("Unsubscribing from tokens: %s", unsubscribe_data)

            if self.ws:
                self.ws.sendMessage(
                    json.dumps(request_data).encode("utf-8"), isBinary=True
                )
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
        if self.debug:
            logger.debug("Resubscribing to tokens: %s", self.subscribed_tokens)

        if not self.subscribed_tokens:
            logger.debug("No tokens to resubscribe")
            return False

        return self.subscribe(self.subscribed_tokens)

    def decode_data(self, data: bytes) -> dict[str, Any]:
        """
        Decode the data received from the WebSocket connection. The data is
        decoded using the FeedResponse protobuf message.

        Parameters
        ----------
        data: ``bytes``
            The raw data received from the WebSocket connection

        Returns
        -------
        ``dict[str, Any]``
            A dictionary containing the decoded data

        """
        feed_response = decoder.FeedResponse()  # type: ignore
        feed_response.ParseFromString(data)

        return MessageToDict(feed_response)

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

        if "feeds" not in data:
            return

        for token, token_data in data["feeds"].items():
            if token_data.get("ltpc") is None:
                continue

            exchange_id = ExchangeType.get_exchange(token.split("|")[0].split("_")[0])
            if exchange_id is None:
                if self.debug:
                    logger.error("Exchange not found for token: %s", token)
                continue

            data_to_save = {
                "retrieval_timestamp": str(time.time()),
                "symbol": self.token_map.get(token, "unknown"),
                "exchange_id": exchange_id.value,
                "data_provider_id": DataProviderType.UPLINK.value,
                "last_traded_price": token_data["ltpc"].get("ltp", -1),
                "last_traded_timestamp": token_data["ltpc"].get("ltt", -1),
                "last_traded_quantity": token_data["ltpc"].get("ltq", -1),
                "close_price": token_data["ltpc"].get("cp", -1),
            }
            if self.on_data_save_callback:
                self.on_data_save_callback(json.dumps(data_to_save))

        if self.debug:
            logger.debug("Received data: %s", data)

    @staticmethod
    def initialize_socket(cfg, on_save_data_callback=None):
        """
        Initialize the UplinkSocket connection with the specified configuration.
        """
        credentials = UplinkCredentials.get_credentials()
        access_tokens = credentials.access_token

        if access_tokens is None:
            raise ValueError("Access token is missing")

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_tokens}",
        }
        api_response = requests.get(
            url=UPLINK_WEBSOCKET_AUTH_URL, headers=headers, timeout=10
        )

        if api_response.status_code != 200:
            raise ValueError(
                f"Failed to authorize with status code: {api_response.status_code}"
            )
        response = api_response.json()
        websocket_url = response["data"]["authorized_redirect_uri"]

        return UplinkSocket(
            websocket_url,
            cfg.get("guid", None),
            cfg.get("subscription_mode", "ltpc"),
            on_save_data_callback,
            debug=cfg.get("debug", False),
            ping_interval=cfg.get("ping_interval", 25),
            ping_message=cfg.get("ping_message", "ping"),
        )
