import json
import logging
from typing import cast

from autobahn.twisted.websocket import WebSocketServerFactory, WebSocketServerProtocol

from app.sockets.websocket_server_manager import ConnectionManager
from app.utils.asyncio_utils.coro_utils import fire_and_forgot

logger = logging.getLogger(__name__)


class StockTickerServerProtocol(WebSocketServerProtocol):
    """
    This class handles WebSocket connections for the stock ticker server. It manages the
    connection with the client, processes incoming messages, and sends messages back to the client.
    It is responsible for subscribing and unsubscribing to stock tokens based on the messages
    received from the client.
    """

    connection_manager: ConnectionManager | None = None

    def onOpen(self):
        """
        Called when the WebSocket connection is opened.
        """
        logger.info("WebSocket connection open: %s", self.peer)
        fire_and_forgot(self.connection_manager.connect(self))

    def handle_message(self, message: str | bytes):
        """
        Handle incoming messages from the client. This method is called when a message is received
        from the client. It decodes the message, parses it as JSON, and processes the action. The
        action can be either "subscribe" or "unsubscribe" for stock tokens.

        Parameters
        ----------
        message : ``str | bytes``
            The message received from the client
        """

        if isinstance(message, bytes):
            message = message.decode("utf-8")

        msg_data = json.loads(message)

        action = msg_data.get("action")
        stock_tokens = msg_data.get("stocks")

        if isinstance(stock_tokens, str):
            stock_tokens = [stock_tokens]

        if self.connection_manager is None:
            logger.warning("No connection_manager set for %s", self.peer)
            return

        if action == "subscribe" and stock_tokens:
            for token in stock_tokens:
                fire_and_forgot(self.connection_manager.handle_subscribe(self, token))

        elif action == "unsubscribe" and stock_tokens:
            for token in stock_tokens:
                fire_and_forgot(self.connection_manager.handle_unsubscribe(self, token))
        else:
            logger.warning("Invalid message format from %s: %s", self.peer, message)
            err_msg = {
                "type": "error",
                "message": "Invalid action or missing stock token",
            }
            fire_and_forgot(
                self.connection_manager.send_personal_message(err_msg, self)
            )

    def onMessage(self, payload: str | bytes, isBinary: bool):
        """
        Called when a message is received from the client. This method is responsible for handling
        incoming messages. It decodes the message, parses it as JSON, and processes the action.

        Parameters
        ----------
        payload : ``str | bytes``
            The message received from the client
        isBinary : ``bool``
            Indicates whether the message is binary or text
        """
        if isBinary:
            logger.warning("Received binary message from %s, ignoring.", self.peer)
            return

        try:
            self.handle_message(payload)

        except json.JSONDecodeError:
            logger.error(
                "Failed to decode JSON message from %s: %s",
                self.peer,
                cast(bytes, payload).decode("utf-8", errors="ignore"),
            )
            if self.connection_manager:
                err_msg = {"type": "error", "message": "Invalid JSON format"}
                fire_and_forgot(
                    self.connection_manager.send_personal_message(err_msg, self)
                )
        except Exception as e:
            logger.error("Error processing message from %s: %s", self.peer, e)
            if self.connection_manager:
                err_msg = {"type": "error", "message": "Internal server error"}
                fire_and_forgot(
                    self.connection_manager.send_personal_message(err_msg, self)
                )

    def onClose(self, wasClean: bool, code: int, reason: str):
        """
        Called when the WebSocket connection is closed. This method is responsible for handling
        cleanup and logging the disconnection details.

        Parameters
        ----------
        wasClean : ``bool``
            Indicates whether the connection was closed cleanly
        code : ``int``
            The status code indicating the reason for closure
        reason : ``str``
            The reason for closure from the client
        """
        logger.info(
            "WebSocket connection closed: %s (Clean: %s, Code: %d, Reason: '%s')",
            self.peer,
            wasClean,
            code,
            reason,
        )
        if self.connection_manager:
            fire_and_forgot(self.connection_manager.disconnect(self))


class StockTickerServerFactory(WebSocketServerFactory):
    """
    This factory creates instances of StockTickerServerProtocol for handling WebSocket connections.
    It initializes the connection manager and sets up the protocol for each new connection.

    Attributes
    ----------
    protocol : ``StockTickerServerProtocol``
        The protocol class to be used for handling WebSocket connections
    """

    protocol = StockTickerServerProtocol

    def __init__(self, url, connection_manager: ConnectionManager):
        super().__init__(url)

        self.connection_manager = connection_manager

        if connection_manager is None:
            raise ValueError(
                "connection_manager cannot be None. Please provide a valid connection manager."
            )

        logger.info(
            "StockTickerServerFactory initialized with a valid connection_manager and URL: %s",
            url,
        )

    def buildProtocol(self, addr):
        """
        Build a protocol instance for the given address. This method is called when a new WebSocket
        connection is established. It initializes the protocol with the connection manager.

        Parameters
        ----------
        addr : ``str``
            The address of the client connecting to the server

        Returns
        -------
        protocol : ``StockTickerServerProtocol``
            An instance of the StockTickerServerProtocol class
        """
        logger.debug("Building protocol for address: %s", addr)
        protocol = super().buildProtocol(addr)

        if protocol is not None and hasattr(protocol, "connection_manager"):

            protocol.connection_manager = self.connection_manager
            logger.debug(
                "Protocol for %s has been initialized with connection_manager", addr
            )

        return protocol
