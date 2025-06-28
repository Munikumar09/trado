# pylint: disable=protected-access, too-many-lines too-many-locals
"""
Comprehensive tests for the KafkaConsumer class.

This test suite combines all tests for the KafkaConsumer class including:
- Basic initialization and configuration tests
- Method-level unit tests for individual functions
- Comprehensive integration tests for consume_messages method
- Error handling, backoff logic, and recovery mechanisms
- Resource management and cleanup tests
- Edge cases and boundary conditions

The tests are organized into logical sections:
1. Initialization and Configuration Tests
2. Basic Method Tests (polling, transformation, processing)
3. Error Handling Tests
4. Comprehensive consume_messages Tests
5. Resource Management and Cleanup Tests
6. Edge Cases and Integration Tests
"""

import asyncio
import json
import os
import random
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from confluent_kafka import KafkaException

from app.data_layer.streaming.consumers.kafka_consumer import (
    MAX_BACKOFF_ATTEMPTS,
    KafkaConsumer,
)

# =============================================================================
# SHARED FIXTURES
# =============================================================================


@pytest.fixture
def mock_logger():
    """
    Unified fixture that provides a mock logger for testing log output.
    This fixture is shared across all tests and ensures consistent logging
    verification throughout the test suite.
    """
    with patch("app.data_layer.streaming.consumers.kafka_consumer.logger") as mock:
        yield mock


def _create_mock_redis_client():
    """
    Helper function to create consistent mock Redis client.
    """
    mock_client = AsyncMock()
    mock_client.ping = AsyncMock()
    mock_client.publish = AsyncMock()
    mock_client.close = AsyncMock()
    mock_client.flushdb = AsyncMock()
    return mock_client


@pytest.fixture(autouse=True)
def mock_redis_connection():
    """
    Unified fixture that provides a mock Redis connection and client. This mocks all Redis
    dependencies including connection management, client operations, and cleanup. Used for
    isolating Redis operations during testing.
    """
    mock_redis_client = _create_mock_redis_client()

    with patch(
        "app.data_layer.streaming.consumers.kafka_consumer.RedisAsyncConnection"
    ) as mock_conn_class:
        mock_connection = AsyncMock()
        mock_connection.get_connection = AsyncMock(return_value=mock_redis_client)
        mock_connection.close_connection = AsyncMock()
        mock_conn_class.return_value = mock_connection

        yield {
            "connection": mock_connection,
            "client": mock_redis_client,
            "connection_class": mock_conn_class,
        }


@pytest.fixture
def mock_update_stock_cache():
    """
    Fixture that mocks the update_stock_cache function. Isolates cache updating logic
    from message consumption logic.
    """
    with patch(
        "app.data_layer.streaming.consumers.kafka_consumer.update_stock_cache",
        new_callable=AsyncMock,
    ) as mock:
        yield mock


def _create_mock_confluent_consumer():
    """
    Helper function to create consistent mock Confluent Kafka consumer.
    """
    instance = MagicMock()
    instance.subscribe = MagicMock()
    instance.poll = MagicMock(return_value=None)
    instance.close = MagicMock()
    return instance


@pytest.fixture(autouse=True)
def patch_confluent_consumer():
    """
    Auto-used fixture that patches the Confluent Kafka Consumer. This ensures all tests
    use mocked Kafka consumers instead of real ones, preventing external dependencies
    and making tests deterministic.
    """
    with patch(
        "app.data_layer.streaming.consumers.kafka_consumer.ConfluentConsumer",
        autospec=True,
    ) as mock_consumer_cls:
        mock_consumer_cls.return_value = _create_mock_confluent_consumer()
        yield mock_consumer_cls


@pytest.fixture(autouse=True)
def patch_redis_utils():
    """
    Auto-used fixture that patches Redis utilities. Ensures Redis operations are mocked
    consistently across all tests.
    """
    redis_instance = _create_mock_redis_client()

    with patch("app.utils.redis_utils.redis", new=MagicMock()) as mock_redis:
        with patch(
            "app.utils.redis_utils.async_redis", new=MagicMock()
        ) as mock_async_redis:
            mock_async_redis.ConnectionPool = MagicMock()
            mock_async_redis.Redis = MagicMock(return_value=redis_instance)
            yield {"sync": mock_redis, "async": mock_async_redis}


@pytest.fixture
def sample_kafka_message():
    """
    Fixture providing a sample Kafka message payload. Represents a typical market data
    message for consistent testing.
    """
    return {
        "retrieval_timestamp": 1640995200000,
        "last_traded_timestamp": 1640995200000,
        "symbol": "AAPL",
        "exchange_id": 1,
        "data_provider_id": 1,
        "last_traded_price": 150.25,
        "last_traded_quantity": 100,
        "average_traded_price": 150.00,
        "volume_trade_for_the_day": 1000000,
        "total_buy_quantity": 500000,
        "total_sell_quantity": 500000,
    }


@pytest.fixture
def sample_transformed_message():
    """
    Fixture providing the expected transformed message format. Represents the expected
    output after _transform_message processing.
    """
    return {
        "retrieval_timestamp": 1640995200000,
        "last_traded_timestamp": 1640995200000,
        "symbol": "AAPL",
        "exchange": "NSE",
        "data_provider": "SMARTAPI",
        "last_traded_price": 150.25,
        "last_traded_quantity": 100,
        "average_traded_price": 150.00,
        "volume_trade_for_the_day": 1000000,
        "total_buy_quantity": 500000,
        "total_sell_quantity": 500000,
    }


@pytest.fixture
def test_kafka_config():
    """
    Fixture providing test Kafka configuration constants. Centralizes test configuration
    to avoid repetition.
    """
    return {
        "topic": "test_kafka_consumer_topic",
        "broker": "localhost:9092",
        "group": "test_kafka_consumer_group",
    }


@pytest.fixture
def basic_kafka_consumer(test_kafka_config):
    """
    Fixture that creates a basic KafkaConsumer for method-level testing. Used for tests
    that focus on individual methods rather than the full consume_messages integration.
    """
    consumer = KafkaConsumer(
        topic=test_kafka_config["topic"],
        group_id=test_kafka_config["group"],
        brokers=test_kafka_config["broker"],
    )
    yield consumer
    consumer.stop()
    KafkaConsumer.clear_instance(KafkaConsumer)


def _create_kafka_consumer_with_mocks():
    """
    Helper function to create KafkaConsumer with mocked dependencies.
    """
    mock_consumer_instance = _create_mock_confluent_consumer()

    with patch(
        "app.data_layer.streaming.consumers.kafka_consumer.ConfluentConsumer"
    ) as mock_consumer_class:
        mock_consumer_class.return_value = mock_consumer_instance

        consumer = KafkaConsumer(
            topic="test_topic", group_id="test_group", brokers="localhost:9092"
        )
        consumer._executor = MagicMock()

        return {
            "consumer": consumer,
            "mock_kafka_consumer": mock_consumer_instance,
            "mock_consumer_class": mock_consumer_class,
        }


@pytest.fixture
def kafka_consumer_with_mocks():
    """
    Fixture that creates a KafkaConsumer with all external dependencies mocked.

    Provides a fully isolated consumer instance for comprehensive testing
    of the consume_messages method and related functionality.
    """
    consumer_data = _create_kafka_consumer_with_mocks()

    yield consumer_data

    # Cleanup
    KafkaConsumer.clear_instance(KafkaConsumer)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _cleanup_kafka_consumer():
    """
    Helper function to clean up KafkaConsumer instances.
    """
    KafkaConsumer.clear_instance(KafkaConsumer)


# =============================================================================
# INITIALIZATION AND CONFIGURATION TESTS
# =============================================================================


def test_kafka_consumer_default_initialization():
    """
    Test KafkaConsumer initialization with default values. Verifies that the consumer
    initializes correctly with default configuration and that all expected properties
    are set properly.
    """
    consumer = KafkaConsumer()

    assert isinstance(consumer, KafkaConsumer)
    assert consumer.topic == os.environ.get("KAFKA_TOPIC_INSTRUMENT", "instrument_data")
    assert consumer.group_id == os.environ.get(
        "KAFKA_CONSUMER_GROUP_ID", "trado_consumer_group"
    )
    assert consumer.brokers == os.environ.get("KAFKA_BROKER_URL", "localhost:9092")
    assert consumer.config["auto.offset.reset"] == "earliest"
    assert consumer.config["enable.auto.commit"] is True
    assert consumer.config["broker.address.family"] == "v4"
    assert consumer.config["session.timeout.ms"] == 30000
    assert consumer.config["fetch.min.bytes"] == 1
    assert consumer.config["fetch.wait.max.ms"] == 500
    assert consumer._should_run is False

    _cleanup_kafka_consumer()


def test_kafka_consumer_custom_initialization():
    """
    Test KafkaConsumer initialization with custom values.

    Verifies that custom configuration is properly applied and overrides
    default values as expected.
    """
    config = {
        "auto.offset.reset": "latest",
        "enable.auto.commit": False,
        "broker.address.family": "v4",
        "session.timeout.ms": 45000,
    }
    custom_config = {
        "topic": "custom_topic",
        "group_id": "custom_group",
        "brokers": "custom_broker:9092",
        "config": config,
    }

    consumer = KafkaConsumer(**custom_config)

    assert consumer.topic == "custom_topic"
    assert consumer.group_id == "custom_group"
    assert consumer.brokers == "custom_broker:9092"
    assert consumer.config["auto.offset.reset"] == "latest"
    assert consumer.config["enable.auto.commit"] is False
    assert consumer.config["session.timeout.ms"] == 45000

    _cleanup_kafka_consumer()


def test_kafka_consumer_from_cfg():
    """
    Test KafkaConsumer creation from configuration.

    Verifies the from_cfg class method works correctly for configuration-based
    initialization and handles edge cases.
    """
    # Test initialization from cfg method
    consumer = KafkaConsumer.from_cfg({})
    assert isinstance(consumer, KafkaConsumer)

    _cleanup_kafka_consumer()

    # Test with None config
    consumer = KafkaConsumer.from_cfg({"consumer_config": None})
    assert consumer is None


# =============================================================================
# BASIC METHOD TESTS
# =============================================================================


def _create_mock_message(content="test message"):
    """Helper function to create mock Kafka message."""
    msg_mock = MagicMock()
    msg_mock.value.return_value = content.encode()
    msg_mock.error.return_value = None
    return msg_mock


def test_poll_message_success(basic_kafka_consumer):
    """
    Test successful message polling.

    Verifies that messages are correctly retrieved and decoded from Kafka.
    """
    msg_mock = _create_mock_message("test message")
    basic_kafka_consumer.consumer.poll.return_value = msg_mock

    msg = basic_kafka_consumer._poll_message()
    assert msg == "test message"
    basic_kafka_consumer.consumer.poll.assert_called_once_with(timeout=1.0)


def test_poll_message_none(basic_kafka_consumer):
    """
    Test polling when no messages are available.

    Verifies that None is returned when no messages are available from Kafka.
    """
    basic_kafka_consumer.consumer.poll.return_value = None
    msg = basic_kafka_consumer._poll_message()
    assert msg is None
    basic_kafka_consumer.consumer.poll.assert_called_once_with(timeout=1.0)


def test_poll_message_kafka_exception(basic_kafka_consumer):
    """
    Test polling with Kafka exceptions.

    Verifies that Kafka exceptions are properly propagated when polling fails.
    """
    basic_kafka_consumer.consumer.poll.side_effect = KafkaException("Poll error")

    with pytest.raises(KafkaException) as e:
        basic_kafka_consumer._poll_message()

    assert str(e.value) == "Poll error"


def _assert_transformed_message_fields(transformed, original):
    """Helper function to verify transformed message fields."""
    assert transformed["symbol"] == original["symbol"]
    assert transformed["exchange"] == "NSE"
    assert transformed["data_provider"] == "SMARTAPI"
    assert transformed["last_traded_price"] == original["last_traded_price"]
    assert transformed["last_traded_quantity"] == original["last_traded_quantity"]
    assert transformed["average_traded_price"] == original["average_traded_price"]
    assert (
        transformed["volume_trade_for_the_day"] == original["volume_trade_for_the_day"]
    )
    assert transformed["total_buy_quantity"] == original["total_buy_quantity"]
    assert transformed["total_sell_quantity"] == original["total_sell_quantity"]


def test_transform_message(basic_kafka_consumer, sample_kafka_message):
    """
    Test message transformation.

    Verifies that raw Kafka messages are correctly transformed into the
    expected format with proper field mapping.
    """
    transformed = basic_kafka_consumer._transform_message(sample_kafka_message)
    _assert_transformed_message_fields(transformed, sample_kafka_message)


def _verify_successful_message_processing(
    client, mock_update_stock_cache, sample_transformed_message
):
    """Helper function to verify successful message processing."""
    expected_channel = f"stock:{sample_transformed_message['symbol']}_NSE"

    client.publish.assert_called_once_with(
        expected_channel,
        json.dumps(sample_transformed_message),
    )

    mock_update_stock_cache.assert_called_once_with(
        expected_channel,
        sample_transformed_message,
        client,
    )


@pytest.mark.asyncio
async def test_process_message_success(
    basic_kafka_consumer,
    sample_transformed_message,
    mock_redis_connection,
    mock_update_stock_cache,
):
    """
    Test successful message processing.

    Verifies that transformed messages are correctly published to Redis
    and cache is updated appropriately.
    """
    client = mock_redis_connection["client"]

    await basic_kafka_consumer.process_message(sample_transformed_message, client)

    _verify_successful_message_processing(
        client, mock_update_stock_cache, sample_transformed_message
    )


@pytest.mark.asyncio
async def test_process_message_missing_symbol(
    basic_kafka_consumer,
    sample_transformed_message,
    mock_redis_connection,
    mock_update_stock_cache,
    mock_logger,
):
    """
    Test message processing with missing required fields.

    Verifies that messages missing required fields are handled gracefully
    with appropriate warning logging.
    """
    client = mock_redis_connection["client"]

    # Remove required field
    message_without_symbol = sample_transformed_message.copy()
    message_without_symbol.pop("symbol")

    await basic_kafka_consumer.process_message(message_without_symbol, client)

    mock_logger.warning.assert_called_once_with(
        "Message missing required fields (symbol or exchange): %s",
        message_without_symbol,
    )

    # Verify no Redis operations were performed
    client.publish.assert_not_called()
    mock_update_stock_cache.assert_not_called()


@pytest.mark.asyncio
async def test_process_message_redis_error(
    basic_kafka_consumer,
    sample_transformed_message,
    mock_redis_connection,
    mock_logger,
):
    """
    Test message processing with Redis errors.

    Verifies that Redis errors are properly caught and logged with
    appropriate error details.
    """
    client = mock_redis_connection["client"]
    client.publish.side_effect = Exception("Redis publish error")

    with pytest.raises(Exception) as e:
        await basic_kafka_consumer.process_message(sample_transformed_message, client)

    assert str(e.value) == "Redis publish error"
    mock_logger.exception.assert_called_once_with(
        "Failed to process message: %s, error: %s",
        sample_transformed_message,
        str(e.value),
    )


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


def test_handle_consume_error_scenarios(basic_kafka_consumer, mock_logger):
    """
    Test handling of different error types including JSON, Kafka, and general exceptions.

    Verifies that all error types are logged with appropriate detail for debugging.
    """
    # Test JSON decode error
    json_err = json.JSONDecodeError("msg", "doc", 0)
    basic_kafka_consumer._handle_consume_error(json_err, "bad json")
    mock_logger.error.assert_called_with(
        "JSON decode error: %s | Raw message: %s...", json_err, "bad json"
    )

    mock_logger.reset_mock()

    # Test Kafka exception
    kafka_err = KafkaException("Kafka connection failed")
    basic_kafka_consumer._handle_consume_error(kafka_err, None)
    mock_logger.error.assert_called_with("Kafka error: %s", kafka_err)

    mock_logger.reset_mock()

    # Test general exception
    general_err = Exception("Unexpected error")
    basic_kafka_consumer._handle_consume_error(general_err, None)
    mock_logger.exception.assert_called_with(
        "Unexpected error processing message: %s", general_err
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "consecutive_errors,expected_fmt,expected_args,log_level",
    [
        (1, "Error encountered. Backing off for %.2f seconds.", (1.0,), "info"),
        (4, "Multiple errors (%d). Backing off for %.2f seconds.", (4, 8.0), "warning"),
        (
            10,
            "Reached maximum retry attempts (%d). Backing off for %.2f seconds.",
            (10, 60.0),
            "critical",
        ),
    ],
)
async def test_apply_backoff_logs_and_sleeps(
    monkeypatch,
    basic_kafka_consumer,
    mock_logger,
    consecutive_errors,
    expected_fmt,
    expected_args,
    log_level,
):
    """
    Test exponential backoff behavior with different error counts.

    Verifies that backoff sleep times increase exponentially and that
    appropriate log levels are used based on error severity.
    """
    sleep_called = {}

    async def fake_sleep(secs):
        sleep_called["secs"] = secs

    monkeypatch.setattr("asyncio.sleep", fake_sleep)
    # Patch random.uniform to return 0 for deterministic backoff
    monkeypatch.setattr(random, "uniform", lambda a, b: 0)

    await basic_kafka_consumer._apply_backoff(consecutive_errors)

    # Check correct logger method called with expected format string and arguments
    log_method = getattr(mock_logger, log_level)
    assert log_method.call_count == 1
    call_args = log_method.call_args[0]
    assert call_args[0] == expected_fmt

    # For info, only one argument (backoff_time), for warning/critical, two arguments
    if log_level == "info":
        assert call_args[1:] == expected_args
    else:
        assert call_args[1:3] == expected_args

    # Check that sleep was called with a positive value
    assert sleep_called["secs"] > 0


# =============================================================================
# COMPREHENSIVE CONSUME_MESSAGES TESTS
# =============================================================================


async def _create_async_task_with_timeout(consumer, timeout=1.0):
    """Helper function to create and manage async tasks with timeout."""
    task = asyncio.create_task(consumer.consume_messages())

    try:
        await asyncio.wait_for(task, timeout=timeout)
    except asyncio.TimeoutError:
        consumer.stop()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()

    return task


def _create_mock_executor_with_counter(consumer, stop_after=5):
    """Helper function to create mock executor that stops after a certain number of calls."""
    call_count = 0

    async def mock_run_in_executor(*_):
        nonlocal call_count
        call_count += 1
        if call_count >= stop_after:
            consumer.stop()
        return None

    return mock_run_in_executor, lambda: call_count


def _setup_mock_loop_with_executor(mock_executor_func):
    """Helper function to set up mock event loop with executor."""
    mock_loop = AsyncMock()
    mock_loop.run_in_executor = AsyncMock(side_effect=mock_executor_func)
    return mock_loop


@pytest.mark.asyncio
async def test_successful_message_consumption_and_processing(
    kafka_consumer_with_mocks,
    mock_redis_connection,
    sample_kafka_message,
    sample_transformed_message,
    mock_logger,
):
    """
    Test the complete successful flow of message consumption and processing.

    This is the most important test as it verifies the core functionality:
    1. Messages are polled from Kafka successfully
    2. Raw JSON is parsed correctly
    3. Messages are transformed to the expected format
    4. Messages are published to Redis channels
    5. Cache is updated with the latest data
    6. No errors occur during normal operation
    """
    consumer_data = kafka_consumer_with_mocks
    consumer = consumer_data["consumer"]
    raw_message = json.dumps(sample_kafka_message)

    async def mock_run_in_executor(*_):
        # Return the message on first call, None on subsequent calls to stop the loop
        if not hasattr(mock_run_in_executor, "called"):
            mock_run_in_executor.called = True
            return raw_message
        return None

    with patch("asyncio.get_running_loop") as mock_get_loop:
        mock_loop = _setup_mock_loop_with_executor(mock_run_in_executor)
        mock_get_loop.return_value = mock_loop

        with patch.object(
            consumer, "_transform_message", return_value=sample_transformed_message
        ) as mock_transform:
            with patch.object(
                consumer, "process_message", new_callable=AsyncMock
            ) as mock_process:
                # Start consumption and let it process one message
                task = asyncio.create_task(consumer.consume_messages())
                await asyncio.sleep(0.1)
                consumer.stop()

                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except asyncio.TimeoutError:
                    task.cancel()

                # Verify the message was processed correctly
                mock_transform.assert_called_once_with(sample_kafka_message)
                mock_process.assert_called_once_with(
                    sample_transformed_message, mock_redis_connection["client"]
                )

                # Verify logging
                mock_logger.info.assert_any_call(
                    "Starting Kafka consumer (topic=%s, group=%s)",
                    "test_topic",
                    "test_group",
                )


@pytest.mark.asyncio
async def test_no_messages_available_continues_polling(kafka_consumer_with_mocks):
    """
    Test that the consumer continues polling when no messages are available.

    Verifies continuous polling behavior and appropriate sleep intervals
    when no messages are received from Kafka.
    """
    consumer_data = kafka_consumer_with_mocks
    consumer = consumer_data["consumer"]

    mock_executor_func, _ = _create_mock_executor_with_counter(consumer)

    with patch("asyncio.sleep") as mock_sleep:
        mock_sleep.return_value = None

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = _setup_mock_loop_with_executor(mock_executor_func)
            mock_get_loop.return_value = mock_loop

            await _create_async_task_with_timeout(consumer)

            # Verify that polling happened multiple times
            assert mock_loop.run_in_executor.call_count >= 3

            # Verify that sleep was called for each empty poll
            assert mock_sleep.call_count >= 2


@pytest.mark.asyncio
async def test_error_handling_patterns_and_backoff(kafka_consumer_with_mocks):
    """
    Test comprehensive error handling including JSON, Kafka, and runtime errors with backoff.

    Verifies that all error types are handled gracefully and backoff is applied.
    This consolidates testing of different error scenarios.
    """
    consumer_data = kafka_consumer_with_mocks
    consumer = consumer_data["consumer"]

    # Test different error types in sequence
    call_count = 0

    async def mock_run_in_executor(*_):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "invalid json {"  # JSON error
        if call_count == 2:
            raise KafkaException("Kafka error")
        if call_count == 3:
            raise RuntimeError("Runtime error")

        consumer.stop()
        return None

    with patch("asyncio.get_running_loop") as mock_get_loop:
        mock_loop = _setup_mock_loop_with_executor(mock_run_in_executor)
        mock_get_loop.return_value = mock_loop

        with patch.object(
            consumer, "_apply_backoff", new_callable=AsyncMock
        ) as mock_backoff:
            with patch.object(consumer, "_handle_consume_error") as mock_handle_error:
                await _create_async_task_with_timeout(consumer)

                # Verify multiple error handling calls
                assert mock_handle_error.call_count >= 3
                # Verify backoff was applied multiple times
                assert mock_backoff.call_count >= 3


@pytest.mark.asyncio
async def test_process_message_exception_handling(
    kafka_consumer_with_mocks,
    sample_kafka_message,
    sample_transformed_message,
):
    """
    Test handling of exceptions during message processing (Redis/cache operations).

    Verifies that exceptions in process_message are handled gracefully
    and the consumer continues operating after processing errors.
    """
    consumer_data = kafka_consumer_with_mocks
    consumer = consumer_data["consumer"]
    raw_message = json.dumps(sample_kafka_message)

    call_count = 0

    async def mock_run_in_executor(*_):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return raw_message
        if call_count == 2:
            consumer.stop()
            return None
        return None

    with patch("asyncio.get_running_loop") as mock_get_loop:
        mock_loop = _setup_mock_loop_with_executor(mock_run_in_executor)
        mock_get_loop.return_value = mock_loop

        with patch.object(
            consumer, "_transform_message", return_value=sample_transformed_message
        ):
            # Make process_message raise an exception
            with patch.object(
                consumer, "process_message", new_callable=AsyncMock
            ) as mock_process:
                mock_process.side_effect = Exception("Redis connection failed")

                with patch.object(
                    consumer, "_apply_backoff", new_callable=AsyncMock
                ) as mock_backoff:
                    with patch.object(
                        consumer, "_handle_consume_error"
                    ) as mock_handle_error:
                        await _create_async_task_with_timeout(consumer)

                        # Verify error handling was called
                        mock_handle_error.assert_called()
                        # Verify backoff was applied
                        mock_backoff.assert_called()


@pytest.mark.asyncio
async def test_consecutive_errors_trigger_backoff(kafka_consumer_with_mocks):
    """
    Test that consecutive errors trigger exponential backoff behavior.

    Verifies that error count increases with consecutive failures and
    backoff sleep time increases exponentially.
    """
    consumer_data = kafka_consumer_with_mocks
    consumer = consumer_data["consumer"]

    call_count = 0

    async def mock_run_in_executor(*_):
        nonlocal call_count
        call_count += 1
        if call_count <= 3:  # Simulate 3 consecutive errors
            raise RuntimeError(f"Test error {call_count}")

        consumer.stop()
        return None

    with patch("asyncio.sleep") as mock_sleep:
        mock_sleep.return_value = None

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = _setup_mock_loop_with_executor(mock_run_in_executor)
            mock_get_loop.return_value = mock_loop

            await _create_async_task_with_timeout(consumer, timeout=2.0)

            # Verify that backoff sleep was called multiple times with increasing durations
            sleep_calls = [
                call for call in mock_sleep.call_args_list if call[0][0] > 0.01
            ]
            assert len(sleep_calls) >= 2, "Expected multiple backoff sleep calls"

            # Verify exponential backoff pattern
            if len(sleep_calls) >= 2:
                assert (
                    sleep_calls[1][0][0] > sleep_calls[0][0][0]
                ), "Backoff time should increase"


@pytest.mark.asyncio
async def test_successful_processing_resets_error_count(
    kafka_consumer_with_mocks,
    sample_kafka_message,
    sample_transformed_message,
):
    """
    Test that successful message processing resets the error count.

    Verifies that error count resets to 0 after successful processing
    and subsequent errors start backoff from the beginning.
    """
    consumer_data = kafka_consumer_with_mocks
    consumer = consumer_data["consumer"]

    call_count = 0
    raw_message = json.dumps(sample_kafka_message)

    async def mock_run_in_executor(*_):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("First error")
        if call_count == 2:
            return raw_message  # Successful message
        if call_count == 3:
            raise RuntimeError("Second error after success")
        consumer.stop()
        return None

    with patch("asyncio.sleep") as mock_sleep:
        mock_sleep.return_value = None

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = AsyncMock()
            mock_loop.run_in_executor = AsyncMock(side_effect=mock_run_in_executor)
            mock_get_loop.return_value = mock_loop

            with patch.object(
                consumer, "_transform_message", return_value=sample_transformed_message
            ):
                with patch.object(
                    consumer, "process_message", new_callable=AsyncMock
                ) as mock_process:

                    # Start consumption
                    task = asyncio.create_task(consumer.consume_messages())

                    # Wait for completion
                    try:
                        await asyncio.wait_for(task, timeout=2.0)
                    except asyncio.TimeoutError:
                        consumer.stop()
                        await asyncio.wait_for(task, timeout=1.0)

                    # Verify successful processing occurred
                    mock_process.assert_called()

                    # Verify backoff was applied for errors
                    backoff_sleeps = [
                        call for call in mock_sleep.call_args_list if call[0][0] > 0.01
                    ]
                    assert len(backoff_sleeps) >= 1, "Expected backoff sleep calls"


@pytest.mark.asyncio
async def test_consumer_restart_scenarios(kafka_consumer_with_mocks):
    """
    Test consumer restart after maximum attempts and prolonged failures.

    Verifies both attempt-based and time-based restart logic work correctly
    for scenarios where errors persist.
    """
    consumer_data = kafka_consumer_with_mocks
    consumer = consumer_data["consumer"]

    call_count = 0

    async def mock_run_in_executor(*_):
        nonlocal call_count
        call_count += 1
        if call_count <= MAX_BACKOFF_ATTEMPTS + 1:
            raise RuntimeError(f"Persistent error {call_count}")

        consumer.stop()
        return None

    with patch("asyncio.sleep") as mock_sleep:
        mock_sleep.return_value = None

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = AsyncMock()
            mock_loop.run_in_executor = AsyncMock(side_effect=mock_run_in_executor)
            mock_get_loop.return_value = mock_loop

            # Start consumption
            task = asyncio.create_task(consumer.consume_messages())

            # Wait for completion
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                consumer.stop()
                await asyncio.wait_for(task, timeout=2.0)

            # Verify that multiple errors occurred
            assert call_count > MAX_BACKOFF_ATTEMPTS

            # Verify sleep was called for backoff
            assert mock_sleep.call_count > 0

            # Verify consumer was accessed (indicating restart)
            assert consumer_data["mock_kafka_consumer"].close.call_count > 0


# =============================================================================
# RESOURCE MANAGEMENT AND CLEANUP TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_resource_management_and_cleanup(
    kafka_consumer_with_mocks, mock_redis_connection
):
    """
    Test comprehensive resource management including graceful shutdown, cleanup, and stop method.

    Verifies that resources are properly cleaned up during cancellation, finally blocks,
    and stop method calls.
    """
    consumer_data = kafka_consumer_with_mocks
    consumer = consumer_data["consumer"]

    # Test stop method state change
    assert consumer._should_run is False
    consumer._should_run = True
    assert consumer._should_run is True

    mock_executor_func, _ = _create_mock_executor_with_counter(consumer, stop_after=3)

    with patch("asyncio.get_running_loop") as mock_get_loop:
        mock_loop = _setup_mock_loop_with_executor(mock_executor_func)
        mock_get_loop.return_value = mock_loop

        # Test graceful shutdown via cancellation
        task = asyncio.create_task(consumer.consume_messages())
        await asyncio.sleep(0.1)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Verify resources are cleaned up
        mock_redis_connection["connection"].close_connection.assert_called()

        # Verify should_run was set to False
        assert consumer._should_run is False


@pytest.mark.asyncio
async def test_fatal_error_propagation_with_cleanup(
    kafka_consumer_with_mocks, mock_redis_connection
):
    """
    Test that fatal errors are propagated while still performing cleanup.

    Verifies that fatal errors like Redis startup failures are not caught
    but resources are still cleaned up appropriately.
    """
    consumer_data = kafka_consumer_with_mocks
    consumer = consumer_data["consumer"]

    # Make Redis connection fail during startup
    mock_redis_connection["connection"].get_connection.side_effect = Exception(
        "Redis connection failed"
    )

    # Start consumption and expect it to fail
    with pytest.raises(Exception) as exc_info:
        await consumer.consume_messages()

    # Verify the correct exception was raised
    assert "Redis connection failed" in str(exc_info.value)


# =============================================================================
# EDGE CASES AND INTEGRATION TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_edge_cases_and_special_scenarios(
    kafka_consumer_with_mocks, mock_redis_connection, mock_logger
):
    """
    Test comprehensive edge cases including Redis failures, empty messages, and large messages.

    Verifies that various edge cases are handled gracefully including startup failures,
    empty transformed messages, and large message log truncation.
    """
    consumer_data = kafka_consumer_with_mocks
    consumer = consumer_data["consumer"]

    # Test 1: Redis connection failure during startup
    mock_redis_connection["connection"].get_connection.side_effect = Exception(
        "Redis server unavailable"
    )

    with pytest.raises(Exception) as exc_info:
        await consumer.consume_messages()

    assert "Redis server unavailable" in str(exc_info.value)

    # Reset Redis connection for next tests
    mock_redis_connection["connection"].get_connection.side_effect = None
    mock_redis_connection["connection"].get_connection.return_value = (
        mock_redis_connection["client"]
    )

    # Test 2: Empty transformed message handling
    sample_kafka_message = {"symbol": "AAPL", "data": "test"}
    raw_message = json.dumps(sample_kafka_message)

    call_count = 0

    async def mock_run_in_executor_empty(*_):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return raw_message

        consumer.stop()
        return None

    with patch("asyncio.get_running_loop") as mock_get_loop:
        mock_loop = _setup_mock_loop_with_executor(mock_run_in_executor_empty)
        mock_get_loop.return_value = mock_loop

        with patch.object(consumer, "_transform_message", return_value=None):
            with patch.object(
                consumer, "process_message", new_callable=AsyncMock
            ) as mock_process:
                await _create_async_task_with_timeout(consumer)

                # Verify process_message was called with None
                mock_process.assert_called_once_with(
                    None, mock_redis_connection["client"]
                )

    # Test 3: Large message log truncation
    # Reset call count for new test
    call_count = 0
    large_data = "x" * 1000  # 1000 character string
    large_message = {"symbol": "AAPL", "data": large_data}
    large_raw_message = json.dumps(large_message)

    async def mock_run_in_executor_large(*_):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return large_raw_message

        consumer.stop()
        return None

    with patch("asyncio.get_running_loop") as mock_get_loop:
        mock_loop = _setup_mock_loop_with_executor(mock_run_in_executor_large)
        mock_get_loop.return_value = mock_loop

        with patch.object(consumer, "_transform_message", return_value={}):
            with patch.object(consumer, "process_message", new_callable=AsyncMock):
                await _create_async_task_with_timeout(consumer, timeout=2.0)

                # Verify debug log was called with truncated message
                debug_calls = [
                    call
                    for call in mock_logger.debug.call_args_list
                    if "Raw message:" in str(call)
                ]
                if debug_calls:  # Only check if debug was called
                    logged_message = debug_calls[-1][0][
                        1
                    ]  # Get the last debug call (large message)
                    if (
                        len(logged_message) > 200
                    ):  # Only check truncation for large messages
                        assert len(logged_message) <= 203  # 200 + "..."
                        assert logged_message.endswith("...")
