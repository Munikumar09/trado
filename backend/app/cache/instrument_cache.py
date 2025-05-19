import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError

from app.utils.common.logger import get_logger
from app.utils.constants import CHANNEL_PREFIX
from app.utils.redis_utils import RedisAsyncConnection

logger = get_logger(Path(__file__).name)

formats = [
    "%Y-%m-%dT%H:%M:%S.%f%z",  # ISO 8601 with microseconds and timezone
    "%Y-%m-%dT%H:%M:%S%z",  # ISO 8601 without microseconds, with timezone
    "%Y-%m-%dT%H:%M:%S.%f",  # ISO 8601 with microseconds, no timezone (assume UTC)
    "%Y-%m-%dT%H:%M:%S",  # ISO 8601 without microseconds, no timezone (assume UTC)
    "%Y-%m-%d %H:%M:%S.%f",  # Space separated with microseconds (assume UTC)
    "%Y-%m-%d %H:%M:%S",  # Space separated without microseconds (assume UTC)
]


@lru_cache(maxsize=128)
def parse_timestamp(timestamp_str: str) -> datetime | None:
    """
    Parses a timestamp string into a timezone-aware UTC datetime object with format caching.
    Handles various ISO formats and Unix timestamps (seconds and milliseconds).

    Parameters
    ----------
    timestamp_str: ``str``
        The timestamp string to parse, which can be in various formats:
        - ISO 8601 format with or without timezone
        - Space-separated date and time
        - Unix timestamp in seconds or milliseconds

    Returns
    -------
    datetime: ``datetime | None``
        A timezone-aware UTC datetime object if parsing is successful,
        None if the timestamp cannot be parsed with any known format
    """
    dt = None
    for fmt in formats:
        try:
            dt = datetime.strptime(timestamp_str, fmt)

            # Ensure timezone awareness (assume UTC if not specified)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                # Convert to UTC if it has another timezone
                dt = dt.astimezone(timezone.utc)

            return dt
        except (ValueError, TypeError):
            continue

    # Try parsing as Unix timestamp (seconds or milliseconds)
    try:
        ts_float = float(timestamp_str)

        # Heuristic: If the number is very large, assume milliseconds.
        # Timestamps around 2001 and later in ms
        if ts_float > 1e11:
            dt = datetime.fromtimestamp(ts_float / 1000, tz=timezone.utc)
        else:
            dt = datetime.fromtimestamp(ts_float, tz=timezone.utc)
        return dt
    except (ValueError, TypeError):
        logger.warning("Could not parse timestamp: %s as a float.", timestamp_str)

    logger.warning("Could not parse timestamp: %s with known formats.", timestamp_str)

    return None


class CacheUpdateError(Exception):
    """
    Exception raised when stock cache update operations fail.

    This exception indicates problems with updating stock data in Redis,
    such as connection issues, invalid data formats, or concurrent modification
    conflicts.
    """


def _should_update(current_timestamp, new_timestamp):
    return current_timestamp is None or new_timestamp > current_timestamp


def _validate_inputs(cache_key: str, cache_data: dict) -> tuple[bool, str]:
    """
    Validates the inputs for cache update operation.
    Checks if the cache key and data are valid, and if the timestamp is present.
    Returns a tuple indicating whether the inputs are valid and the timestamp string.

    Parameters
    ----------
    cache_key: ``str``
        The Redis key under which the stock data is stored
    cache_data: ``dict``
        The stock data to cache, which must include 'last_traded_timestamp' field

    Returns
    -------
    ``tuple[bool, str]``
        A tuple where the first element is a boolean indicating if the inputs are valid,
        and the second element is the timestamp string or an error message
    """
    if not cache_key or not isinstance(cache_key, str):
        logger.error("Invalid cache key provided")
        return False, "invalid_key"
    if not cache_data or not isinstance(cache_data, dict):
        logger.error("Invalid cache data provided")
        return False, "invalid_data"
    timestamp_str = cache_data.get("last_traded_timestamp")
    if not timestamp_str:
        logger.warning("Missing timestamp for %s. Skipping update.", cache_key)
        return False, "missing_timestamp"
    return True, timestamp_str


def _extract_current_timestamp(current_data_str, cache_key):
    current_timestamp = None
    current_timestamp_str = None
    if current_data_str:
        try:
            current_data = json.loads(current_data_str)
            current_timestamp_str = current_data.get("last_traded_timestamp")
            if current_timestamp_str:
                current_timestamp = parse_timestamp(current_timestamp_str)
        except json.JSONDecodeError:
            logger.warning(
                "Could not decode existing cache data for %s. Overwriting.",
                cache_key,
            )
        except Exception as e:
            logger.error(
                "Error processing existing cache data for %s: %s. Overwriting.",
                cache_key,
                e,
            )
    return current_timestamp, current_timestamp_str


async def update_stock_cache(
    cache_key: str, cache_data: dict, redis_client=None
) -> bool:
    """
    Updates the stock data cache using atomic check-and-set operation.

    Updates the Redis cache for a stock if the new data has a more recent timestamp.
    Uses Redis transaction to ensure atomic operations and prevent race conditions.
    Adds metadata such as processing timestamp and cache update time to the stored data.

    Parameters
    ----------
    cache_key: ``str``
        The Redis key under which the stock data is stored, typically prefixed with
        a channel identifier.
    cache_data: ``dict``
        The stock data to cache, which must include 'last_traded_timestamp' field.
        Other fields are preserved as-is in the cache.
    redis_client: ``redis.asyncio.Redis | None``
        Optional Redis client to use for the operation. If not provided,
        a new connection will be created from the global connection pool.

    Returns
    -------
    ``bool``
        True if the cache was updated successfully (i.e., data was newer),
        False if update was skipped (validation failure, older timestamp, or race condition)

    Raises
    ------
    ``CacheUpdateError``
        If there's a serious error that prevents cache updating beyond simple validation
        failures, such as Redis connection issues or JSON serialization problems
    """
    valid, timestamp_str = _validate_inputs(cache_key, cache_data)

    if not valid:
        return False

    try:
        if redis_client is None:
            redis_client = await RedisAsyncConnection().get_connection()
    except Exception as e:
        error_msg = f"Failed to get Redis connection for update: {e}"
        logger.error(error_msg)
        raise CacheUpdateError(error_msg) from e

    new_timestamp = parse_timestamp(timestamp_str)
    if not new_timestamp:
        logger.warning(
            "Invalid timestamp format for %s: '%s'. Skipping update.",
            cache_key,
            timestamp_str,
        )
        return False

    try:
        async with redis_client.pipeline(transaction=True) as pipe:
            await pipe.watch(cache_key)
            current_data_str = await pipe.get(cache_key)
            current_timestamp, current_timestamp_str = _extract_current_timestamp(
                current_data_str, cache_key
            )

            if _should_update(current_timestamp, new_timestamp):
                new_data = {
                    **cache_data,
                    "processed_timestamp_utc": new_timestamp.isoformat(),
                    "cache_updated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                pipe.multi()
                pipe.set(cache_key, json.dumps(new_data))
                result = await pipe.execute()
                if result and result[0]:
                    logger.debug(
                        "Cache updated for %s with timestamp %s",
                        cache_key,
                        timestamp_str,
                    )
                    return True
                logger.warning(
                    "Cache update transaction likely failed for %s (result: %s). Key might have changed.",
                    cache_key,
                    result,
                )
                return False

            logger.debug(
                "Skipping update for %s. New timestamp '%s' (%s) is not later than cached '%s' (%s)",
                cache_key,
                timestamp_str,
                new_timestamp,
                current_timestamp_str,
                current_timestamp,
            )
            return False

    except redis.WatchError:
        logger.warning(
            "WatchError for %s. Another client modified the key. Update skipped for this message.",
            cache_key,
        )
        return False
    except RedisConnectionError as e:
        error_msg = f"Redis connection error during update for {cache_key}: {e}"
        logger.error(error_msg)
        raise CacheUpdateError(error_msg) from e
    except Exception as e:
        error_msg = f"Error updating cache for {cache_key}: {e}"
        logger.error(error_msg)
        raise CacheUpdateError(error_msg) from e


async def get_stock_data(stock_name: str) -> dict[str, Any] | None:
    """
    Retrieves the latest stock data from the Redis cache.

    Fetches stock data by constructing a cache key from the stock name
    and retrieving the corresponding JSON data from Redis.

    Parameters
    ----------
    stock_name: ``str``
        The name/symbol of the stock to retrieve from the cache
        (e.g., "RELIANCE", "INFY")

    Returns
    -------
    data: ``dict[str, Any] | None``
        A dictionary containing the latest stock data including price,
        timestamp, and other relevant fields if found in the cache.
        None if the stock data is not in the cache or the stock name is invalid.

    Raises
    ------
    ``CacheUpdateError``
        If there's a serious error accessing Redis or parsing the cache data,
        such as connection issues or invalid JSON format
    """
    if not stock_name or not isinstance(stock_name, str):
        logger.error("Invalid stock name provided")
        return None

    try:
        r = await RedisAsyncConnection().get_connection()
        cache_key = f"{CHANNEL_PREFIX}{stock_name}"
        data_str = await r.get(cache_key)

        if data_str:
            return json.loads(data_str)

        return None
    except RuntimeError as e:
        error_msg = f"RuntimeError retrieving cache for {stock_name}: {e}"
        logger.error(error_msg)
        raise CacheUpdateError(error_msg) from e
    except RedisConnectionError as e:
        error_msg = f"Redis connection error during retrieval for {stock_name}: {e}"
        logger.error(error_msg)
        raise CacheUpdateError(error_msg) from e
    except json.JSONDecodeError as e:
        error_msg = f"Failed to decode cached JSON for {stock_name}: {e}"
        logger.error(error_msg)
        raise CacheUpdateError(error_msg) from e
    except Exception as e:
        error_msg = f"Error retrieving cache for {stock_name}: {e}"
        logger.error(error_msg)
        raise CacheUpdateError(error_msg) from e
