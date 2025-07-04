import asyncio
import json
import time
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, Set

from autobahn.twisted.websocket import WebSocketServerProtocol
from redis.asyncio import Redis

from app.cache.instrument_cache import get_stock_data
from app.core.singleton import Singleton
from app.utils.asyncio_utils.coro_utils import fire_and_forgot
from app.utils.common.logger import get_logger
from app.utils.constants import CHANNEL_PREFIX

logger = get_logger(Path(__file__).name)


class RedisPubSubError(Exception):
    """
    Base exception for Redis PubSub related errors. Serves as the parent class for all
    custom exceptions related to Redis Pub/Sub operations, providing a common exception
    type that can be caught to handle all Redis Pub/Sub specific errors.
    """


class RedisPubSubManager(metaclass=Singleton):
    """
    Manages Redis Pub/Sub functionality, allowing for subscription and unsubscription to
    channels. Handles incoming messages and invokes callbacks for subscribed channels.
    Implements monitoring for channel activity and automatically unsubscribes from inactive
    channels to conserve resources.

    Attributes
    ----------
    redis: ``Redis``
        The Redis client instance to use for Pub/Sub operations
    """

    def __init__(self, redis: Redis):
        """
        Initialize the RedisPubSubManager with a Redis client.
        """
        self.redis = redis
        self.subscribed_channels: Dict[str, Callable[[str, str], Coroutine]] = {}
        self.tasks: Dict[str, asyncio.Task] = {}
        self.channel_activity: Dict[str, float] = {}  # Track last activity time

    async def subscribe(
        self, channel: str, callback: Callable[[str, str], Coroutine]
    ) -> bool:
        """
        Subscribes to a Redis channel and sets a callback. Creates a separate task to listen
        for messages on the channel and invoke the callback when a message is received. Tracks
        channel activity for monitoring purposes.

        Parameters
        ----------
        channel: ``str``
            The Redis channel to subscribe to
        callback: ``Callable[[str, str], Coroutine]``
            The callback function to invoke when a message is received on the channel

        Returns
        -------
        ``bool``
            True if subscription was successful, False if already subscribed

        Raises
        ------
        ``ValueError``
            If the channel name is invalid
        ``RedisPubSubError``
            If there's an error subscribing to the channel
        """
        if not channel or not isinstance(channel, str):
            raise ValueError("Invalid channel name provided.")

        if channel in self.subscribed_channels:
            logger.debug("Already subscribed to channel %s", channel)
            return False

        try:
            self.redis.ping()
            self.subscribed_channels[channel] = callback
            self.channel_activity[channel] = time.time()  # Track subscription time

            # Start a new task to listen for messages on the channel
            def done_callback(t):
                self._handle_task_done(channel, t)

            task = fire_and_forgot(
                self._listen_channel(channel, callback), done_callback
            )

            # Store the task for later cancellation
            self.tasks[channel] = task

            logger.info("Subscribed to Redis channel: %s", channel)
            return True

        except Exception as e:
            if channel in self.subscribed_channels:
                del self.subscribed_channels[channel]
            if channel in self.channel_activity:
                del self.channel_activity[channel]

            logger.error("Failed to subscribe to channel %s: %s", channel, e)
            raise RedisPubSubError(
                f"Failed to subscribe to channel {channel}: {e}"
            ) from e

    def _handle_task_done(self, channel: str, task: asyncio.Task):
        """
        Handle cleanup when a listener task completes. Manages resource cleanup for
        completed tasks, whether they completed normally, were cancelled, or failed
        with an exception. This prevents resource leaks from abandoned tasks.

        Parameters
        ----------
        channel: ``str``
            The Redis channel associated with the completed task
        task: ``asyncio.Task``
            The completed asyncio task object
        """
        if channel in self.tasks and self.tasks[channel] == task:
            if task.cancelled():
                logger.debug("Task for channel %s was cancelled", channel)
            elif task.exception():
                logger.error(
                    "Task for channel %s failed with exception: %s",
                    channel,
                    task.exception(),
                )
            else:
                logger.debug("Task for channel %s completed normally", channel)

            # Clean up resources even if the task ended unexpectedly
            if channel in self.tasks:
                del self.tasks[channel]
            if channel in self.subscribed_channels and not task.cancelled():
                del self.subscribed_channels[channel]
            if channel in self.channel_activity:
                del self.channel_activity[channel]

    async def unsubscribe(self, channel: str) -> bool:
        """
        Unsubscribes from a Redis channel and cleans up associated resources. Cancels the
        task listening to the channel and removes all related tracking information from
        the manager's data structures.

        Parameters
        ----------
        channel: ``str``
            The Redis channel to unsubscribe from

        Returns
        -------
        ``bool``
            True if unsubscription was successful, False if not subscribed
        """
        if channel not in self.subscribed_channels:
            logger.debug("Not subscribed to %s, nothing to unsubscribe", channel)
            return False

        # Cancel the task if it's still running
        if channel in self.tasks:
            task = self.tasks[channel]
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    logger.warning(
                        "Cancellation of task for %s timed out or was cancelled",
                        channel,
                    )
                except Exception as e:
                    logger.error(
                        "Error during task cancellation for %s: %s", channel, e
                    )

        # Remove the channel from tracking data structures
        if channel in self.subscribed_channels:
            del self.subscribed_channels[channel]
        if channel in self.tasks:
            del self.tasks[channel]
        if channel in self.channel_activity:
            del self.channel_activity[channel]

        logger.info("Unsubscribed from Redis channel: %s", channel)
        return True

    async def _listen_channel(
        self, channel: str, callback: Callable[[str, str], Coroutine]
    ):
        """
        Listens for messages on a Redis channel and invokes callbacks. Creates a Redis
        PubSub subscription and continuously listens for messages, invoking the provided
        callback when messages are received. Handles cancellation and cleanup gracefully.

        Parameters
        ----------
        channel: ``str``
            The Redis channel to listen to
        callback: ``Callable[[str, str], Coroutine]``
            The callback function to invoke when a message is received on the channel
        """
        pubsub = self.redis.pubsub()
        try:
            await pubsub.subscribe(channel)
            logger.info("Listening for messages on channel: %s", channel)

            async for message in pubsub.listen():
                if message["type"] == "message":

                    # Update last activity time
                    self.channel_activity[channel] = time.time()
                    try:
                        data = message["data"]
                        if isinstance(data, bytes):
                            data = data.decode("utf-8")
                        await callback(channel, data)
                    except Exception as e:
                        logger.error("Error in callback for channel %s: %s", channel, e)
                        # Continue listening despite callback errors
        except asyncio.CancelledError:
            logger.info("Listener for channel %s was cancelled", channel)
            raise  # Re-raise to properly handle task cancellation
        except Exception as e:
            logger.error("Error in listener for channel %s: %s", channel, e)
            raise  # Re-raise to mark the task as failed
        finally:
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.close()
                logger.debug("Cleaned up pubsub resources for %s", channel)
            except Exception as e:
                logger.error("Error cleaning up pubsub for channel %s: %s", channel, e)

    async def close(self):
        """
        Close all subscriptions and clean up resources. Gracefully shuts down all channel
        subscriptions and cleans up associated resources. This should be called during
        application shutdown to ensure proper cleanup.
        """
        logger.info("Closing all Redis PubSub subscriptions")

        # Create a copy of channels to avoid modification during iteration
        channels = list(self.subscribed_channels.keys())

        for channel in channels:
            await self.unsubscribe(channel)


class ConnectionManager(metaclass=Singleton):
    """
    Manages WebSocket connections and subscriptions for stock updates. Handles client
    connections, disconnections, and subscription management. Implements rate limiting,
    connection monitoring, and efficient  message broadcasting to subscribed clients.

    Attributes
    ----------
    pubsub_manager: ``RedisPubSubManager``
        The RedisPubSubManager instance to use for managing Redis Pub/Sub
    """

    def __init__(self, pubsub_manager: RedisPubSubManager):
        """
        Initialize the ConnectionManager with a RedisPubSubManager.
        """
        self.active_connections: Dict[str, WebSocketServerProtocol] = {}
        self.subscriptions: Dict[str, Set[str]] = {}
        self.client_subscriptions: Dict[str, Set[str]] = {}
        self.pubsub_manager = pubsub_manager

    def _get_client_id(self, client: WebSocketServerProtocol) -> str:
        """
        Returns the client ID based on the client's connection. Uses the peer property
        of the WebSocketServerProtocol to create a unique identifier for each client
        connection.

        Parameters
        ----------
        client: ``WebSocketServerProtocol``
            The WebSocket client connection
        Returns
        -------
        ``str``
            The unique client ID
        """
        return client.peer

    async def connect(self, client: WebSocketServerProtocol):
        """
        Handles a new client connection and initializes their subscription set.
        This method is called when a client successfully connects to the WebSocket.
        Parameters
        ----------
        client: ``WebSocketServerProtocol``
            The WebSocket client connection
        """

        client_id = self._get_client_id(client)

        # Add the client to the active connections and initialize their subscription set
        self.active_connections[client_id] = client
        self.client_subscriptions[client_id] = set()
        logger.info("Client connected: %s", client_id)

    async def disconnect(self, client: WebSocketServerProtocol):
        """
        Handles the disconnection of a client and cleans up their subscriptions.
        This method is called when a client disconnects from the WebSocket.

        Parameters
        ----------
        client: ``WebSocketServerProtocol``
            The WebSocket client connection

        Raises
        ------
        ``KeyError``
            If the client ID is not found in the active connections
        """
        client_id = self._get_client_id(client)

        # Remove the client from active connections
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            subscribed_stocks = self.client_subscriptions.pop(client_id, set())

            # Clean up subscriptions for the client
            for stock in subscribed_stocks:
                if stock in self.subscriptions:
                    self.subscriptions[stock].discard(client_id)

                    # If no clients are subscribed to the stock, unsubscribe from Redis
                    # and remove the stock from the subscriptions
                    if not self.subscriptions[stock]:
                        del self.subscriptions[stock]
                        await self.pubsub_manager.unsubscribe(
                            f"{CHANNEL_PREFIX}{stock}"
                        )
            logger.info(
                "Client disconnected: %s. Cleaned up subscriptions: %s",
                client_id,
                subscribed_stocks,
            )
        else:
            logger.warning("Attempted to disconnect unknown client: %s", client_id)

    async def redis_callback(self, channel: str, message: str):
        """
        Callback function to handle incoming messages from Redis channel. This function is
        called when a message is received on a subscribed channel. It decodes the message
        and broadcasts it to all subscribers of the channel.

        Parameters
        ----------
        channel: ``str``
            The Redis channel that received the message
        message: ``str``
            The actual message received from the Redis channel
        """
        try:
            payload = json.loads(message)

            await self.broadcast_to_subscribers(
                channel.replace(CHANNEL_PREFIX, ""), payload
            )
        except Exception as e:
            logger.error("Error in Redis callback for %s: %s", channel, e)

    async def handle_subscribe(self, client: WebSocketServerProtocol, stock_token: str):
        """
        Handles a subscription request from a client. This method is called when a client
        sends a subscription request for a specific stock token. It subscribes the client
        to the stock token and sends an acknowledgment message back to the client.

        Parameters
        ----------
        client: ``WebSocketServerProtocol``
            The WebSocket client connection
        stock_token: ``str``
            The stock token to subscribe to

        Raises
        ------
        ``KeyError``
            If the client ID is not found in the active connections
        ``ValueError``
            If the stock token is invalid or empty
        """
        client_id = self._get_client_id(client)

        # Check if the given stock token is not empty
        if not stock_token:
            logger.warning("Empty stock token received from %s", client_id)
            await self.send_personal_message(
                {"type": "error", "message": "Invalid stock token"}, client
            )
            return

        stock_token = stock_token.upper()

        # Add the stock token to the client's subscriptions
        self.client_subscriptions.setdefault(client_id, set()).add(stock_token)

        # Add the client ID to the stock token's subscriptions
        self.subscriptions.setdefault(stock_token, set()).add(client_id)

        # Send acknowledgment message to the client
        await self.send_personal_message(
            {"type": "subscription_ack", "stock": stock_token}, client
        )

        # Subscribe to the Redis channel only for the first client requesting this stock
        # This prevents duplicate subscriptions and ensures efficient message handling.
        # Once subscribed, an active listener (task) will listen for messages on the channel
        # and invoke the callback when a message is received.
        if len(self.subscriptions[stock_token]) == 1:
            await self.pubsub_manager.subscribe(
                f"{CHANNEL_PREFIX}{stock_token}", self.redis_callback
            )

        # Fetch the latest stock data from the cache and send it to the client. This ensures
        # that the client receives the most recent data immediately after subscribing.
        current_data = await get_stock_data(stock_token, self.pubsub_manager.redis)

        if current_data:
            await self.send_personal_message(
                {"type": "stock_update", "data": current_data}, client
            )
        else:
            await self.send_personal_message(
                {
                    "type": "stock_update",
                    "stock": stock_token,
                    "data": None,
                    "message": "No current data available",
                },
                client,
            )

    async def handle_unsubscribe(
        self, client: WebSocketServerProtocol, stock_token: str
    ):
        """
        Handles an un-subscription request from a client. This method is called when a client
        sends an un-subscription request for a specific stock token. It unsubscribes the client
        from the stock token and sends an acknowledgment message back to the client.

        Parameters
        ----------
        client: ``WebSocketServerProtocol``
            The WebSocket client connection
        stock_token: ``str``
            The stock token to unsubscribe from

        Raises
        ------
        ``KeyError``
            If the client ID is not found in the active connections
        ``ValueError``
            If the stock token is invalid or empty
        """
        client_id = self._get_client_id(client)

        # Check if the given stock token is not empty
        if not stock_token:
            logger.warning(
                "Empty stock token received for unsubscribe from %s", client_id
            )
            await self.send_personal_message(
                {"type": "error", "message": "Invalid stock token"}, client
            )
            return

        stock_token = stock_token.upper()

        # Remove the stock token from the client's subscriptions
        if client_id in self.client_subscriptions:
            self.client_subscriptions[client_id].discard(stock_token)

        # Remove the client ID from the stock token's subscriptions
        if stock_token in self.subscriptions:
            self.subscriptions[stock_token].discard(client_id)

            # If no clients are subscribed to the stock, unsubscribe from Redis
            # and remove the stock from the subscriptions
            if not self.subscriptions[stock_token]:
                del self.subscriptions[stock_token]
                await self.pubsub_manager.unsubscribe(f"{CHANNEL_PREFIX}{stock_token}")

        logger.info("Client %s unsubscribed from %s", client_id, stock_token)

        # Send acknowledgment message to the client for un-subscription
        await self.send_personal_message(
            {"type": "unsubscription_ack", "stock": stock_token}, client
        )

    async def send_personal_message(
        self, message: Dict[str, Any], client: WebSocketServerProtocol
    ):
        """
        Sends a personal message to a specific client.

        Parameters
        ----------
        message: ``Dict[str, Any]``
            The message to be sent to the client
        client: ``WebSocketServerProtocol``
            The WebSocket client connection
        """

        try:
            payload = json.dumps(message).encode("utf-8")
            client.sendMessage(payload, isBinary=False)
        except Exception as e:
            logger.error(
                "Failed to send message to %s: %s", self._get_client_id(client), e
            )

    async def broadcast_to_subscribers(self, stock_token: str, data: Dict[str, Any]):
        """
        Broadcasts stock updates to all subscribers of a specific stock token.
        This method is called when a message is received on a subscribed Redis channel.
        It sends the message to all clients subscribed to the stock token.

        Parameters
        ----------
        stock_token: ``str``
            The stock token to broadcast the message to
        data: ``Dict[str, Any]``
            The data to be sent to the subscribers
        """
        stock_token = stock_token.upper()
        if stock_token not in self.subscriptions:
            return

        message = {"type": "stock_update", "data": data}
        payload = json.dumps(message).encode("utf-8")
        disconnected_clients = set()

        for client_id in list(self.subscriptions[stock_token]):
            client = self.active_connections.get(client_id)

            # Check if the client is still connected, if not, remove it from the subscription
            if client:
                try:
                    client.sendMessage(payload, isBinary=False)
                except Exception as e:
                    logger.error("Broadcast error to %s: %s", client_id, e)
                    disconnected_clients.add(client_id)
            else:
                disconnected_clients.add(client_id)

        for client_id in disconnected_clients:
            # Remove the disconnected clients from the subscription
            self.subscriptions[stock_token].discard(client_id)

            if client_id in self.client_subscriptions:
                self.client_subscriptions[client_id].discard(stock_token)

            if not self.subscriptions[stock_token]:
                del self.subscriptions[stock_token]
                await self.pubsub_manager.unsubscribe(f"{CHANNEL_PREFIX}{stock_token}")

    async def close(self):
        """
        Closes the WebSocket connection and cleans up resources.
        """
        for client_id in list(self.active_connections.keys()):
            client = self.active_connections[client_id]
            await self.send_personal_message({"type": "close"}, client)
            client.close()
        self.active_connections.clear()
        self.subscriptions.clear()
        self.client_subscriptions.clear()
