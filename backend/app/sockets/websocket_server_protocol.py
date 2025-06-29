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
        Handles initialization when a WebSocket connection is established.
        
        Notifies the connection manager of the new connection.
        """
        logger.info("WebSocket connection open: %s", self.peer)
        fire_and_forgot(self.connection_manager.connect(self))

    def handle_message(self, message: str | bytes):
        """
        Processes a client message to subscribe or unsubscribe from stock tokens.
        
        Decodes and parses the incoming message, determines the requested action, and updates stock subscriptions accordingly. Sends an error message to the client if the action is invalid or required data is missing.
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
        Handles incoming text messages from the client, processing JSON commands for subscription management.
        
        Ignores binary messages. On receiving a text message, attempts to parse it as JSON and process the requested action. If the message is not valid JSON, or if an error occurs during processing, sends an appropriate error message back to the client.
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
        Handle cleanup and notify the connection manager when the WebSocket connection is closed.
        
        This method logs the closure details and asynchronously informs the connection manager to disconnect the client.
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
        """
        Initialize the WebSocket server factory with a URL and a connection manager.
        
        Raises:
            ValueError: If the connection manager is None.
        """
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
        Creates and initializes a StockTickerServerProtocol instance for a new WebSocket connection.
        
        Returns:
            StockTickerServerProtocol: The protocol instance associated with the new client connection.
        """
        logger.debug("Building protocol for address: %s", addr)
        protocol = super().buildProtocol(addr)

        if protocol is not None and hasattr(protocol, "connection_manager"):

            protocol.connection_manager = self.connection_manager
            logger.debug(
                "Protocol for %s has been initialized with connection_manager", addr
            )

        return protocol
