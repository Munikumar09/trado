import asyncio
from typing import Any

from twisted.internet.interfaces import IListeningPort

from app.core.singleton import Singleton
from app.sockets.websocket_server_manager import ConnectionManager


class AppState(metaclass=Singleton):
    """
    Holds the application state and configuration. This class is a singleton that provides a
    centralized place to manage application state, including Kafka consumer tasks, WebSocket
    server connections, and Redis client instances.
    """

    kafka_consumer_task: asyncio.Task | None = None
    websocket_server_port: IListeningPort | None = None
    websocket_server_running: bool = False
    redis_client: Any | None = None
    connection_manager: ConnectionManager | None = None
    startup_complete: bool = False
