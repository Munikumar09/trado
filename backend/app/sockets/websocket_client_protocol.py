# pylint: disable=no-member, super-with-arguments
import time
from pathlib import Path

from autobahn.twisted.websocket import WebSocketClientProtocol
from autobahn.websocket.types import ConnectionResponse

from app.utils.common.logger import get_logger

logger = get_logger(Path(__file__).name, log_level="DEBUG")


class MarketDataWebSocketClientProtocol(WebSocketClientProtocol):
    """
    This class is a subclass of the `WebSocketClientProtocol` class from the `autobahn` library.
    It is used to create a WebSocket client protocol that can be used to connect to a WebSocket
    server and send and receive messages over the WebSocket connection.

    Attributes
    ----------
    ping_interval: ``int``
        The interval (in seconds) at which the client should send ping messages to the server
    ping_message: ``str``
        The text message that should be sent as a ping message to the server
    """

    _next_ping = None
    _next_pong_check = None
    _last_pong_time = None
    _last_ping_time = None

    def __init__(self, ping_interval, ping_message, *args, **kwargs):
        self.ping_interval = ping_interval
        self.ping_message = ping_message
        self.keepalive_interval = 2 * ping_interval
        super(MarketDataWebSocketClientProtocol, self).__init__(*args, **kwargs)

    def onConnect(self, response: ConnectionResponse):
        """
        This callback is triggered immediately after the WebSocket opening handshake is completed
        and a new WebSocket connection is established between the client and the server

        Parameters
        ----------
        response: ``ConnectionResponse``
            A ConnectionResponse object that contains information about the connection, including:
            - The IP address of the server (`response.peer`)
            - Response headers received from the server
            - WebSocket protocol details and more
        """
        if self.factory.debug:
            logger.debug("Connected to %s with %s", response.peer, response.protocol)

        self.factory.ws = self

        if self.factory.on_connect:
            self.factory.on_connect(self, response)

        self.factory.resetDelay()

    def onOpen(self):
        """
        This callback is triggered when the WebSocket connection has been established
        and is open for sending and receiving messages. This will be called after
        `onConnect`.
        """
        # Start the heartbeat messages to keep the connection alive
        self._loop_ping()
        self._loop_pong_check()

        if self.factory.debug:
            logger.debug("Connection Opened")

        if self.factory.on_open:
            self.factory.on_open(self)

    def onMessage(self, payload: bytes, isBinary: bool):
        """
        This callback is triggered when a WebSocket message is received from the server.

        If the message is a pong (heartbeat response), update the last pong timestamp.
        This method handles both binary and text messages.

        Parameters
        ----------
        payload: ``bytes``
            The received data, either in binary or text format.
        isBinary: ``bool``
            A flag indicating if the message is in binary format.
        """
        if isBinary:
            if self.factory.on_message:
                self.factory.on_message(self, payload, isBinary)
        else:
            try:
                message = payload.decode("utf-8")
            except UnicodeDecodeError:
                logger.error("Failed to decode message as UTF-8: %s", payload)
                return

            # Handle heartbeat pong response
            if message == "pong":
                if self.factory.debug and self._last_pong_time:
                    logger.debug(
                        "Last pong was received at %.2f seconds ago (%.4f)",
                        time.time() - self._last_ping_time,
                        self._last_pong_time,
                    )
                self._last_pong_time = time.time()

    # pylint: disable=arguments-renamed
    def onClose(self, was_clean: bool, code: int, reason: str):
        """
        This callback is triggered when the WebSocket connection is closed

        Parameters
        ----------
        was_clean: ``bool``
            A boolean flag that indicates whether the connection was closed cleanly
            meaning that the closing handshake was completed successfully
        code: ``int``
            The close status code sent by the server
        reason: ``str``
            The reason for closing the connection sent by the server
        """
        if self.factory.debug:
            logger.debug(
                "Closing the connection with code %s and reason %s, is closing clean %s",
                code,
                reason,
                was_clean,
            )
        if not was_clean:
            if self.factory.on_error:
                self.factory.on_error(self, code, reason)

        if self.factory.on_close:
            self.factory.on_close(self, code, reason)

        self._last_ping_time = None
        self._last_pong_time = None

        if self._next_ping and self._next_ping.active():
            self._next_ping.cancel()

        if self._next_pong_check and self._next_pong_check.active():
            self._next_pong_check.cancel()

    def _loop_ping(self):
        """
        This method is used to send a text-based ping message ("ping") to the server
        at regular intervals to keep the connection alive.
        """
        if self.factory.debug and self._last_ping_time:
            logger.debug(
                "Last ping was sent at %.2f seconds ago (%.4f)",
                time.time() - self._last_ping_time,
                self._last_ping_time,
            )

        if self.factory.debug and not self._last_ping_time:
            logger.debug("Sending heartbeat ping...")

        # Sending "ping" as a text message
        self.sendMessage(self.factory.ping_message.encode("utf-8"), isBinary=False)
        self._last_ping_time = time.time()

        self._next_ping = self.factory.reactor.callLater(
            self.ping_interval, self._loop_ping
        )

    def _loop_pong_check(self):
        """
        This method is used to check if the server is sending pong messages (text-based "pong")
        to keep the connection alive. If the server does not send a pong message within the
        specified interval, the connection is dropped and reconnected.
        """
        if self._last_pong_time:
            last_pong_diff = time.time() - self._last_pong_time

            if last_pong_diff > self.keepalive_interval:
                if self.factory.debug:
                    logger.debug(
                        "Last pong was received %s seconds ago. Dropping the connection to reconnect.",
                        last_pong_diff,
                    )
                self.dropConnection(abort=True)

        self._next_pong_check = self.factory.reactor.callLater(
            self.keepalive_interval, self._loop_pong_check
        )
