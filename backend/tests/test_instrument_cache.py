# pylint: disable= protected-access
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.cache import instrument_cache
from app.utils.constants import CHANNEL_PREFIX
from app.utils.redis_utils import RedisAsyncConnection

# Helper for valid timestamp
VALID_TIMESTAMP = datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@pytest.fixture(autouse=True)
def mock_redis(mocker):
    """
    Fixture to mock RedisAsyncConnection.get_connection method.
    """
    mock_redis = AsyncMock()
    mocker.patch(
        "app.utils.redis_utils.RedisAsyncConnection.get_connection",
        return_value=mock_redis,
    )
    return mock_redis


@pytest.mark.parametrize(
    "ts_str,expected",
    [
        (
            "2024-05-20T12:34:56.123456+0000",
            datetime(2024, 5, 20, 12, 34, 56, 123456, tzinfo=timezone.utc),
        ),
        (
            "2024-05-20T12:34:56+0000",
            datetime(2024, 5, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
        (
            "2024-05-20T12:34:56.123456",
            datetime(2024, 5, 20, 12, 34, 56, 123456, tzinfo=timezone.utc),
        ),
        ("2024-05-20T12:34:56", datetime(2024, 5, 20, 12, 34, 56, tzinfo=timezone.utc)),
        (
            "2024-05-20 12:34:56.123456",
            datetime(2024, 5, 20, 12, 34, 56, 123456, tzinfo=timezone.utc),
        ),
        ("2024-05-20 12:34:56", datetime(2024, 5, 20, 12, 34, 56, tzinfo=timezone.utc)),
        (
            str(
                int(datetime(2024, 5, 20, 12, 34, 56, tzinfo=timezone.utc).timestamp())
            ),
            datetime(2024, 5, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
        (
            str(
                int(
                    datetime(2024, 5, 20, 12, 34, 56, tzinfo=timezone.utc).timestamp()
                    * 1000
                )
            ),
            datetime(2024, 5, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
        # Negative timestamp (before epoch)
        (
            str(int(datetime(1960, 1, 1, tzinfo=timezone.utc).timestamp())),
            datetime(1960, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        ),
        # Zero timestamp (epoch)
        ("0", datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        # Timestamp with timezone offset
        (
            "2024-05-20T12:34:56+05:30",
            datetime(2024, 5, 20, 7, 4, 56, tzinfo=timezone.utc),
        ),
    ],
)
def test_parse_timestamp_valid_formats(ts_str: str, expected: datetime) -> None:
    """
    Test parse_timestamp with various valid timestamp formats.
    """
    dt = instrument_cache.parse_timestamp(ts_str)
    assert dt == expected


def test_parse_timestamp_invalid() -> None:
    """
    Test parse_timestamp returns None for invalid inputs.
    """
    assert instrument_cache.parse_timestamp(None) is None
    assert instrument_cache.parse_timestamp("") is None
    assert instrument_cache.parse_timestamp("not-a-date") is None

    # Test with whitespace string
    assert instrument_cache.parse_timestamp("   ") is None


def test_should_update() -> None:
    """
    Test _should_update logic for timestamp comparisons.
    """
    now = datetime.now(timezone.utc)
    later = now + timedelta(seconds=1)

    assert instrument_cache._should_update(None, now)
    assert instrument_cache._should_update(now, later)
    assert not instrument_cache._should_update(later, now)
    assert not instrument_cache._should_update(now, now)


def test_validate_inputs() -> None:
    """
    Test _validate_inputs for various valid and invalid input scenarios.
    """
    # Valid
    valid, ts = instrument_cache._validate_inputs(
        "key", {"last_traded_timestamp": VALID_TIMESTAMP}
    )
    assert valid and ts == VALID_TIMESTAMP

    # Invalid key
    valid, msg = instrument_cache._validate_inputs(
        None, {"last_traded_timestamp": VALID_TIMESTAMP}  # type: ignore[arg-type]
    )
    assert not valid and msg == "invalid_key"

    valid, msg = instrument_cache._validate_inputs(
        123, {"last_traded_timestamp": VALID_TIMESTAMP}  # type: ignore[arg-type]
    )
    assert not valid and msg == "invalid_key"

    # Invalid data
    valid, msg = instrument_cache._validate_inputs("key", None)  # type: ignore[arg-type]
    assert not valid and msg == "invalid_data"

    valid, msg = instrument_cache._validate_inputs("key", "bad")  # type: ignore[arg-type]
    assert not valid and msg == "invalid_data"

    # Missing timestamp
    valid, msg = instrument_cache._validate_inputs("key", {"foo": 1})
    assert not valid and msg == "missing_timestamp"

    # Empty dict
    valid, msg = instrument_cache._validate_inputs("key", {})
    assert not valid and msg == "invalid_data"

    # Data missing last_traded_timestamp key
    valid, msg = instrument_cache._validate_inputs("key", {"other": 123})
    assert not valid and msg == "missing_timestamp"


def test_extract_current_timestamp() -> None:
    """
    Test _extract_current_timestamp for valid, missing, and invalid JSON data.
    """
    # Valid
    data = json.dumps({"last_traded_timestamp": VALID_TIMESTAMP})
    ts, ts_str = instrument_cache._extract_current_timestamp(data, "key")
    assert ts.isoformat() == VALID_TIMESTAMP
    assert ts_str == VALID_TIMESTAMP

    # No timestamp
    data = json.dumps({"foo": 1})
    ts, ts_str = instrument_cache._extract_current_timestamp(data, "key")
    assert ts is None and ts_str is None

    # Invalid JSON
    ts, ts_str = instrument_cache._extract_current_timestamp("not-json", "key")
    assert ts is None and ts_str is None

    # Empty string
    ts, ts_str = instrument_cache._extract_current_timestamp("", "key")
    assert ts is None and ts_str is None

    # Missing key in JSON
    data = json.dumps({"another_key": 123})
    ts, ts_str = instrument_cache._extract_current_timestamp(data, "key")
    assert ts is None and ts_str is None


@pytest.mark.asyncio
async def test_update_stock_cache_valid(mocker):
    """
    Test update_stock_cache for valid update, skip, and newer timestamp scenarios.
    """

    cache_key = "test_update_stock_cache_valid"

    # STEP 1: Prepare mock pipeline object
    mock_pipe = MagicMock()
    mock_pipe.watch = AsyncMock()
    mock_pipe.multi = MagicMock()
    mock_pipe.set = MagicMock()
    mock_pipe.execute = AsyncMock()

    # STEP 2: Create async context manager mock for redis_client.pipeline()
    mock_pipeline_cm = AsyncMock()
    mock_pipeline_cm.__aenter__.return_value = mock_pipe
    mock_pipeline_cm.__aexit__.return_value = None

    # STEP 3: Create redis_client mock and patch pipeline()
    mock_redis = mocker.Mock()
    mock_redis.pipeline.return_value = mock_pipeline_cm

    # CASE 1: No existing data → should update
    mock_pipe.get = AsyncMock(return_value=None)  # No data in cache
    mock_pipe.execute.return_value = [True]  # Successful transaction

    cache_data = {"last_traded_timestamp": VALID_TIMESTAMP, "foo": 1}
    result = await instrument_cache.update_stock_cache(
        cache_key, cache_data, redis_client=mock_redis
    )
    assert result is True

    # CASE 2: Same timestamp → should skip
    cached_data = {
        "last_traded_timestamp": VALID_TIMESTAMP,
        "foo": 1,
    }
    mock_pipe.get = AsyncMock(return_value=json.dumps(cached_data))

    result = await instrument_cache.update_stock_cache(
        cache_key, cache_data, redis_client=mock_redis
    )
    assert result is False

    # CASE 3: Newer timestamp → should update
    newer_timestamp = (datetime.now(timezone.utc) + timedelta(seconds=10)).isoformat()
    cached_data["last_traded_timestamp"] = VALID_TIMESTAMP  # older one in cache
    mock_pipe.get = AsyncMock(return_value=json.dumps(cached_data))
    mock_pipe.execute.return_value = [True]  # Success again

    new_data = {
        "last_traded_timestamp": newer_timestamp,
        "foo": 2,
    }
    result = await instrument_cache.update_stock_cache(
        cache_key, new_data, redis_client=mock_redis
    )
    assert result is True


@pytest.mark.asyncio
async def test_update_stock_cache_invalid() -> None:
    """
    Test update_stock_cache for invalid input and invalid timestamp.
    """
    redis_connection = RedisAsyncConnection()
    r = await redis_connection.get_connection()

    # Invalid input
    result = await instrument_cache.update_stock_cache(None, {}, r)  # type: ignore[arg-type]
    assert not result

    # Invalid timestamp
    cache_key = "test_update_stock_cache_invalid"
    cache_data = {"last_traded_timestamp": "not-a-date"}
    result = await instrument_cache.update_stock_cache(
        cache_key, cache_data, redis_client=r
    )
    assert not result

    await redis_connection.close_connection()


@pytest.mark.asyncio
async def test_update_stock_cache_missing_timestamp() -> None:
    """
    Test update_stock_cache with missing last_traded_timestamp in cache_data.
    """
    redis_connection = RedisAsyncConnection()
    r = await redis_connection.get_connection()
    cache_key = "test_update_stock_cache_missing_timestamp"
    cache_data = {"foo": 1}
    result = await instrument_cache.update_stock_cache(
        cache_key, cache_data, redis_client=r
    )
    assert not result

    await r.delete(cache_key)
    await redis_connection.close_connection()


@pytest.mark.asyncio
async def test_get_stock_data(mock_redis) -> None:
    """
    Test get_stock_data for not found, valid, invalid stock name, and invalid JSON.
    """
    redis_connection = RedisAsyncConnection()
    r = await redis_connection.get_connection()
    stock_name = "INFY"
    cache_key = f"{CHANNEL_PREFIX}{stock_name}"
    mock_redis.get = AsyncMock(return_value=None)
    await r.delete(cache_key)

    # Not found
    result = await instrument_cache.get_stock_data(stock_name, r)
    assert result is None

    # Insert and get
    data = {"foo": 1, "last_traded_timestamp": VALID_TIMESTAMP}
    await r.set(cache_key, json.dumps(data))

    mock_redis.get = AsyncMock(return_value=json.dumps(data))
    result = await instrument_cache.get_stock_data(stock_name, r)
    assert result is not None
    assert result["foo"] == 1

    # Invalid stock name
    result = await instrument_cache.get_stock_data(None, r)  # type: ignore[arg-type]
    assert result is None

    await r.delete(cache_key)
    await redis_connection.close_connection()

    # Invalid json data
    cache_key = f"{CHANNEL_PREFIX}INVALID_JSON"
    await r.set(cache_key, "not-json")

    mock_redis.get = AsyncMock(return_value="not-json")

    with pytest.raises(instrument_cache.CacheUpdateError) as e:
        await instrument_cache.get_stock_data("INVALID_JSON", r)

    assert (
        str(e.value)
        == "Failed to decode cached JSON for INVALID_JSON: Expecting value: line 1 column 1 (char 0)"
    )

    # Test get_stock_data with empty string as stock name.
    result = await instrument_cache.get_stock_data("", r)
    assert result is None
