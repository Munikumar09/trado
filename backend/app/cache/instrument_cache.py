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
    Parse a timestamp string into a timezone-aware UTC datetime object.
    
    Attempts multiple ISO 8601 and space-separated datetime formats, with or without microseconds and timezone information. If no format matches, tries to interpret the string as a Unix timestamp in seconds or milliseconds. Returns None if parsing fails.
     
    Parameters:
        timestamp_str (str): The timestamp string to parse, supporting ISO 8601, space-separated formats, or Unix timestamps.
    
    Returns:
        datetime | None: A UTC-aware datetime object if parsing succeeds, otherwise None.
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
    """
    Determine whether the cache should be updated based on timestamp comparison.
    
    Returns True if there is no current timestamp or if the new timestamp is more recent than the current one.
    """
    return current_timestamp is None or new_timestamp > current_timestamp


def _validate_inputs(cache_key: str, cache_data: dict) -> tuple[bool, str]:
    """
    Validate cache key and data for a cache update operation.
    
    Checks that the cache key is a non-empty string, the cache data is a dictionary, and that it contains a 'last_traded_timestamp' field.
    
    Returns:
        tuple[bool, str]: A tuple where the first element is True and the second is the timestamp string if validation succeeds; otherwise, False and an error message indicating the validation failure.
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
    """
    Extracts and parses the current timestamp from cached JSON data.
    
    Attempts to decode the provided JSON string, retrieve the 'last_traded_timestamp' field, and parse it into a UTC datetime object. Returns a tuple containing the parsed datetime (or None if unavailable) and the original timestamp string.
    """
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
    Atomically updates stock data in the Redis cache if the new data has a more recent timestamp.
    
    Validates input, parses timestamps, and uses a Redis transaction with optimistic locking to ensure the cache is only updated if the new data is newer than the existing cached data. Adds metadata fields for processing and cache update times in UTC ISO 8601 format. Returns True if the cache was updated, or False if the update was skipped due to validation failure, older timestamp, or concurrent modification.
    
    Returns:
        bool: True if the cache was updated, False if skipped.
    
    Raises:
        CacheUpdateError: If a Redis connection error, JSON serialization issue, or other serious failure occurs.
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
    Retrieve the latest stock data for a given stock name from the Redis cache.
    
    Attempts to fetch and parse the cached JSON data for the specified stock. Returns the data as a dictionary if found, or None if the stock is not present or the name is invalid.
    
    Returns:
        dict[str, Any] | None: Parsed stock data dictionary if available, otherwise None.
    
    Raises:
        CacheUpdateError: If a Redis connection error, JSON decoding error, or other serious retrieval issue occurs.
    """
    if not stock_name or not isinstance(stock_name, str):
        logger.error("Invalid stock name provided")
        return None

    try:
        connection = RedisAsyncConnection()
        r = await connection.get_connection()
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
