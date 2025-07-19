import asyncio
from typing import TYPE_CHECKING

from redis.asyncio import Redis
from twisted.internet.interfaces import IListeningPort

from app.core.singleton import Singleton
from app.sockets.websocket_server_manager import ConnectionManager

if TYPE_CHECKING:
    from app.data_layer.streaming.consumers.kafka_consumer import KafkaConsumer


class AppState(metaclass=Singleton):
    """
    Holds the application state and configuration. This class is a singleton that provides a
    centralized place to manage application state, including Kafka consumer tasks, WebSocket
    server connections, and Redis client instances.

    Attributes
    ----------
    kafka_consumer_task: ``asyncio.Task | None``
        The asyncio task running the Kafka consumer.
    websocket_server_port: ``IListeningPort | None``
        The Twisted listening port for the WebSocket server.
    websocket_server_running: ``bool``
        Flag indicating if the WebSocket server is running.
    redis_client: ``Any | None``
        The Redis client instance.
    connection_manager: ``ConnectionManager | None``
        The WebSocket connection manager.
    startup_complete: ``bool``
        Flag indicating if the application startup is complete.
    """

    kafka_consumer_task: asyncio.Task | None = None
    kafka_consumer: "KafkaConsumer | None" = None
    websocket_server_port: IListeningPort | None = None
    websocket_server_running: bool = False
    redis_client: Redis | None = None
    connection_manager: ConnectionManager | None = None
    startup_complete: bool = False
