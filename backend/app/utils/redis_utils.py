from pathlib import Path

import redis
import redis.asyncio as async_redis

from app.utils.common.logger import get_logger
from app.utils.fetch_data import get_required_env_var

logger = get_logger(Path(__file__).name)


# Initialize Redis client
def init_redis_client(is_async: bool = True) -> redis.Redis | async_redis.Redis:
    """
    Initialize Redis client with asynchronous support. Based on the value of `is_async`,
    the function initializes either an asynchronous or synchronous Redis client.

    Parameters
    ----------
    is_async: ``bool``, ( defaults = True )
        If True, initializes an asynchronous Redis client

    Returns
    -------
    ``redis.Redis | redis.asyncio.Redis``
        The Redis client instance

    Raises
    ------
    `` redis.ConnectionError``
        If the connection to Redis fails
    """
    redis_host = get_required_env_var("REDIS_HOST")
    redis_port = int(get_required_env_var("REDIS_PORT"))
    redis_db = get_required_env_var("REDIS_DB")

    try:
        pool: redis.ConnectionPool | async_redis.ConnectionPool
        if is_async:
            pool = async_redis.ConnectionPool(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )
            return async_redis.Redis(connection_pool=pool)

        pool = redis.ConnectionPool(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True,
        )
        return redis.Redis(connection_pool=pool)
    except redis.ConnectionError as e:
        logger.error(
            "Failed to connect to Redis at %s:%d (db: %s): %s",
            redis_host,
            redis_port,
            redis_db,
            str(e),
        )
        raise
