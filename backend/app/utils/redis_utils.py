from pathlib import Path
from typing import Optional

import redis
import redis.asyncio as async_redis

from app.core.singleton import Singleton
from app.utils.common.logger import get_logger
from app.utils.fetch_data import get_env_var

logger = get_logger(Path(__file__).name)


class RedisConnection(metaclass=Singleton):
    """
    Redis connection manager class. This class is a singleton that manages the connection to a
    Redis server. It provides methods to get a connection and close it.

    Attributes
    ----------
    host : ``str``
        The host or IP address of the Redis server
    port : ``int``
        The port of the Redis server
    db : ``int``
        The database number to use for the connection
    decode_responses : ``bool``
        Whether to decode responses from Redis as UTF-8 strings
    socket_timeout : ``int``
        The timeout for socket operations in seconds
    socket_connect_timeout : ``int``
        The timeout for socket connection operations in seconds
    retry_on_timeout : ``bool``
        Whether to retry on timeout errors
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        decode_responses: bool = True,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        retry_on_timeout: bool = True,
    ) -> None:
        """
        Initialize Redis connection parameters, using environment variables as defaults if not provided.
        
        Parameters:
            host (str, optional): Redis server hostname. Defaults to the value of the REDIS_HOST environment variable if not specified.
            port (int, optional): Redis server port. Defaults to the value of the REDIS_PORT environment variable if not specified.
            db (int, optional): Redis database number. Defaults to the value of the REDIS_DB environment variable if not specified.
            decode_responses (bool): Whether to decode responses from Redis. Defaults to True.
            socket_timeout (int): Timeout in seconds for socket operations. Defaults to 5.
            socket_connect_timeout (int): Timeout in seconds for establishing a connection. Defaults to 5.
            retry_on_timeout (bool): Whether to retry commands if a timeout occurs. Defaults to True.
        """
        self.host: str = host or get_env_var("REDIS_HOST")
        self.port: int = int(port or get_env_var("REDIS_PORT"))
        self.db: int = int(db or get_env_var("REDIS_DB"))
        self.decode_responses = decode_responses
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.retry_on_timeout = retry_on_timeout


class RedisSyncConnection(RedisConnection):
    """
    Singleton class to manage a synchronous Redis connection.
    """

    connection_pool: Optional[redis.ConnectionPool] = None

    def get_connection(self) -> redis.Redis:
        """
        Returns a synchronous Redis client from the connection pool, initializing the pool if necessary.
        
        The connection is verified with a ping command before being returned.
        
        Returns:
            redis.Redis: A synchronous Redis client instance.
        
        Raises:
            redis.ConnectionError: If unable to connect to the Redis server.
            redis.TimeoutError: If the connection attempt times out.
            redis.ResponseError: If the Redis server returns an error during the ping test.
        """
        if self.connection_pool is None:
            self.connection_pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=self.decode_responses,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                retry_on_timeout=self.retry_on_timeout,
            )
        connection = redis.Redis(connection_pool=self.connection_pool)
        connection.ping()  # test connection
        return connection

    def close_connection(self) -> None:
        """
        Closes the synchronous Redis connection pool and releases associated resources.
        """
        if self.connection_pool:
            self.connection_pool.disconnect()
            self.connection_pool = None
            logger.info("Redis connection closed.")


class RedisAsyncConnection(RedisConnection):
    """
    Singleton class to manage an asynchronous Redis connection.
    """

    connection_pool: Optional[async_redis.ConnectionPool] = None

    async def get_connection(self) -> async_redis.Redis:
        """
        Obtain an asynchronous Redis client using a managed connection pool.
        
        Returns:
            An `async_redis.Redis` client instance connected to the configured Redis server.
        
        Raises:
            async_redis.ConnectionError: If the connection to Redis fails.
            async_redis.TimeoutError: If the connection times out.
            async_redis.ResponseError: If the Redis server returns an error.
        """

        if self.connection_pool is None:
            self.connection_pool = async_redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=self.decode_responses,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                retry_on_timeout=self.retry_on_timeout,
            )
        connection = async_redis.Redis(connection_pool=self.connection_pool)
        # await connection.ping()
        return connection

    async def close_connection(self) -> None:
        """
        Closes the asynchronous Redis connection pool and releases associated resources.
        """
        if self.connection_pool:
            await self.connection_pool.disconnect()
            self.connection_pool = None
            logger.info("Redis connection closed.")
