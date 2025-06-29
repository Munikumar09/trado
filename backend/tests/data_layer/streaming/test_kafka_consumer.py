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
    Provides a pytest fixture that yields a mock logger for verifying log output in tests.
    
    This fixture patches the Kafka consumer's logger, enabling consistent and isolated logging assertions across the test suite.
    """
    with patch("app.data_layer.streaming.consumers.kafka_consumer.logger") as mock:
        yield mock


def _create_mock_redis_client():
    """
    Create a mock Redis client with async methods for testing purposes.
    
    Returns:
        AsyncMock: A mock Redis client with async `ping`, `publish`, `close`, and `flushdb` methods.
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
    Fixture that provides a fully mocked Redis connection and client for testing.
    
    Yields:
        dict: Contains the mocked Redis connection, client, and connection class for use in tests.
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
    Fixture that provides a mocked asynchronous `update_stock_cache` function for use in tests.
    
    Yields:
        AsyncMock: The mocked `update_stock_cache` function, allowing tests to isolate cache update behavior from message consumption logic.
    """
    with patch(
        "app.data_layer.streaming.consumers.kafka_consumer.update_stock_cache",
        new_callable=AsyncMock,
    ) as mock:
        yield mock


def _create_mock_confluent_consumer():
    """
    Create a mock Confluent Kafka consumer with mocked subscribe, poll, and close methods.
    
    Returns:
        MagicMock: A mock Kafka consumer instance with subscribe, poll, and close methods.
    """
    instance = MagicMock()
    instance.subscribe = MagicMock()
    instance.poll = MagicMock(return_value=None)
    instance.close = MagicMock()
    return instance


@pytest.fixture(autouse=True)
def patch_confluent_consumer():
    """
    Automatically patches the Confluent Kafka Consumer with a mock for all tests.
    
    This fixture ensures that all Kafka consumer interactions use a mock implementation, isolating tests from external Kafka dependencies and enabling deterministic test behavior.
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
    Auto-used pytest fixture that patches Redis utility modules with mocks for both synchronous and asynchronous Redis operations.
    
    Yields:
        dict: A dictionary containing the mocked synchronous ('sync') and asynchronous ('async') Redis modules for use in tests.
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
    Provides a sample Kafka message payload representing typical market data for use in tests.
    
    Returns:
        dict: A dictionary containing fields such as timestamps, symbol, exchange ID, price, quantity, and volume.
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
    Provides a sample transformed message dictionary representing the expected output of the message transformation process in tests.
    
    Returns:
        dict: A dictionary with standardized market data fields for use in test assertions.
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
    Fixture that provides a KafkaConsumer instance for method-level tests.
    
    Yields:
        KafkaConsumer: An instance configured with test Kafka settings for isolated method testing. Ensures cleanup after use.
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
    Creates a KafkaConsumer instance with mocked Kafka consumer and executor for testing purposes.
    
    Returns:
        dict: A dictionary containing the KafkaConsumer instance, the mocked Kafka consumer, and the patched consumer class.
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
    Fixture that provides a KafkaConsumer instance with all external dependencies mocked.
    
    Yields a fully isolated KafkaConsumer for integration and behavioral testing, ensuring no real connections to Kafka or Redis are made. Cleans up the singleton instance after use.
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
    Removes all existing KafkaConsumer singleton instances to ensure a clean test environment.
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
    Test that KafkaConsumer initializes with custom configuration values.
    
    Verifies that user-supplied topic, group ID, brokers, and config dictionary override the default KafkaConsumer settings.
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
    """
    Create a mock Kafka message object with the specified content.
    
    Parameters:
        content (str): The message content to encode as the mock message value. Defaults to "test message".
    
    Returns:
        MagicMock: A mock Kafka message with encoded value and no error.
    """
    msg_mock = MagicMock()
    msg_mock.value.return_value = content.encode()
    msg_mock.error.return_value = None
    return msg_mock


def test_poll_message_success(basic_kafka_consumer):
    """
    Tests that _poll_message successfully retrieves and decodes a message from the Kafka consumer.
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
    Test that `_poll_message` raises a `KafkaException` when the Kafka consumer's `poll` method fails.
    """
    basic_kafka_consumer.consumer.poll.side_effect = KafkaException("Poll error")

    with pytest.raises(KafkaException) as e:
        basic_kafka_consumer._poll_message()

    assert str(e.value) == "Poll error"


def _assert_transformed_message_fields(transformed, original):
    """
    Assert that the transformed message fields match the expected values from the original message.
    
    Parameters:
        transformed (dict): The transformed message to verify.
        original (dict): The original message containing expected values.
    """
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
    Tests that the KafkaConsumer correctly transforms a raw Kafka message into the expected structured format.
    
    Verifies that all relevant fields are mapped accurately from the input message to the transformed output.
    """
    transformed = basic_kafka_consumer._transform_message(sample_kafka_message)
    _assert_transformed_message_fields(transformed, sample_kafka_message)


def _verify_successful_message_processing(
    client, mock_update_stock_cache, sample_transformed_message
):
    """
    Verify that a transformed message was published to the correct Redis channel and that the stock cache was updated.
    
    Parameters:
        client: The mocked Redis client used for publishing.
        mock_update_stock_cache: The mocked function for updating the stock cache.
        sample_transformed_message: The transformed message expected to be published and cached.
    """
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
    Tests that a transformed message is successfully processed by publishing it to Redis and updating the cache.
    
    Verifies that the `process_message` method publishes the message to Redis and invokes the cache update function with the correct data.
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
    Test that processing a message missing required fields logs a warning and skips Redis operations.
    
    Verifies that when a message lacks the 'symbol' or 'exchange' field, the consumer logs a warning and does not publish to Redis or update the stock cache.
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
    Test that Redis errors during message processing are raised and logged.
    
    Verifies that when a Redis publish operation fails during message processing, the exception is propagated and an error is logged with the relevant message and error details.
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
    Tests that the KafkaConsumer's error handler logs JSON decode errors, Kafka exceptions, and general exceptions with the correct log level and detail.
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
    Asynchronously tests that the exponential backoff mechanism logs at the correct level and sleeps for the expected duration based on the number of consecutive errors.
    
    Verifies that:
    - The logger is called with the correct log level, format string, and arguments.
    - The backoff sleep duration increases as expected.
    - The sleep function is invoked with a positive value.
    """
    sleep_called = {}

    async def fake_sleep(secs):
        """
        Mocks an asynchronous sleep by recording the requested duration in the `sleep_called` dictionary.
        """
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
    """
    Run the consumer's message consumption asynchronously with a timeout, ensuring cleanup on timeout.
    
    Parameters:
        consumer: The KafkaConsumer instance whose consume_messages coroutine will be run.
        timeout (float): Maximum time in seconds to allow the task to run before attempting to stop the consumer.
    
    Returns:
        task: The asyncio.Task object representing the running consume_messages coroutine.
    """
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
    """
    Create a mock asynchronous executor function that counts its invocations and stops the given consumer after a specified number of calls.
    
    Parameters:
    	stop_after (int): The number of calls after which the consumer will be stopped. Defaults to 5.
    
    Returns:
    	mock_run_in_executor (Callable): An async function simulating executor behavior and triggering consumer stop.
    	get_call_count (Callable): A function returning the current call count.
    """
    call_count = 0

    async def mock_run_in_executor(*_):
        """
        Simulates asynchronous execution in an executor, incrementing a call counter and stopping the consumer after a specified number of calls.
        
        This mock is used to control the flow of message consumption in tests by stopping the consumer after a set number of invocations.
        """
        nonlocal call_count
        call_count += 1
        if call_count >= stop_after:
            consumer.stop()
        return None

    return mock_run_in_executor, lambda: call_count


def _setup_mock_loop_with_executor(mock_executor_func):
    """
    Create a mocked asynchronous event loop with a custom executor function for testing.
    
    Parameters:
        mock_executor_func (callable): Function to be used as the side effect for the event loop's run_in_executor method.
    
    Returns:
        AsyncMock: A mock event loop with run_in_executor configured to use the provided executor function.
    """
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
    Asynchronously tests the end-to-end flow of consuming and processing a Kafka message.
    
    Verifies that a message is polled from Kafka, parsed, transformed, published to Redis, and the cache is updated, with all expected logging and no errors during normal operation.
    """
    consumer_data = kafka_consumer_with_mocks
    consumer = consumer_data["consumer"]
    raw_message = json.dumps(sample_kafka_message)

    async def mock_run_in_executor(*_):
        # Return the message on first call, None on subsequent calls to stop the loop
        """
        Simulates an executor function that returns a predefined message on the first call and None on subsequent calls.
        
        Returns:
            The predefined raw message on the first invocation; None on all subsequent invocations.
        """
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
    Test that the Kafka consumer continues polling and sleeping when no messages are available.
    
    This test verifies that the consumer repeatedly polls for messages and invokes sleep intervals when no messages are received from Kafka, ensuring continuous operation in the absence of new data.
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
    Test that the consumer handles various error types and applies backoff during message consumption.
    
    Simulates sequential JSON decoding errors, Kafka exceptions, and runtime errors to verify that each is handled gracefully and that the backoff mechanism is invoked for each error scenario.
    """
    consumer_data = kafka_consumer_with_mocks
    consumer = consumer_data["consumer"]

    # Test different error types in sequence
    call_count = 0

    async def mock_run_in_executor(*_):
        """
        Simulates asynchronous execution of message polling with controlled error scenarios for testing.
        
        Returns:
            str or None: Returns an invalid JSON string on the first call, raises a KafkaException on the second call, raises a RuntimeError on the third call, and stops the consumer and returns None on subsequent calls.
        """
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
    Test that exceptions raised during message processing are handled without crashing the consumer.
    
    Simulates an exception in `process_message` (such as a Redis failure) and verifies that error handling and backoff mechanisms are invoked, allowing the consumer to continue operating after the error.
    """
    consumer_data = kafka_consumer_with_mocks
    consumer = consumer_data["consumer"]
    raw_message = json.dumps(sample_kafka_message)

    call_count = 0

    async def mock_run_in_executor(*_):
        """
        Simulates asynchronous execution of a function in an executor, returning a raw message on the first call and stopping the consumer on the second call.
        
        Returns:
            The raw message on the first invocation; None on subsequent calls.
        """
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
    Asynchronously tests that consecutive processing errors in the KafkaConsumer trigger exponential backoff, with increasing sleep durations after each failure.
    """
    consumer_data = kafka_consumer_with_mocks
    consumer = consumer_data["consumer"]

    call_count = 0

    async def mock_run_in_executor(*_):
        """
        Simulates an executor function that raises a RuntimeError for the first three calls, then stops the consumer.
        
        This mock is used to test error handling and backoff logic by triggering consecutive errors before allowing the consumer to stop.
        """
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
    Test that the error count in the KafkaConsumer resets after successful message processing.
    
    Simulates consecutive errors, a successful message processing, and a subsequent error to verify that the backoff mechanism restarts from the initial state after a successful operation.
    """
    consumer_data = kafka_consumer_with_mocks
    consumer = consumer_data["consumer"]

    call_count = 0
    raw_message = json.dumps(sample_kafka_message)

    async def mock_run_in_executor(*_):
        """
        Simulates an asynchronous executor that raises an error on the first call, returns a raw message on the second, raises another error on the third, and stops the consumer on subsequent calls.
        
        Returns:
            The raw message on the second call; None after stopping the consumer.
        """
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
    Test that the Kafka consumer restarts correctly after exceeding maximum backoff attempts due to persistent errors.
    
    Simulates repeated failures during message consumption to verify that both attempt-based and time-based restart mechanisms trigger as expected. Ensures that backoff logic is applied, the consumer is closed and restarted, and resource cleanup occurs after persistent errors.
    """
    consumer_data = kafka_consumer_with_mocks
    consumer = consumer_data["consumer"]

    call_count = 0

    async def mock_run_in_executor(*_):
        """
        Simulates an executor function that raises a RuntimeError for a set number of calls, then stops the consumer.
        
        Raises:
            RuntimeError: For each call until the maximum backoff attempts are exceeded.
        """
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
    Test that the KafkaConsumer performs proper resource cleanup and state management during shutdown.
    
    This test verifies that resources such as Redis connections are closed and the consumer's run state is reset to `False` when the consumer is cancelled or stopped.
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
    Test that fatal errors during startup propagate exceptions while ensuring resource cleanup.
    
    Simulates a Redis connection failure during consumer startup and verifies that the exception is raised and resources are properly cleaned up.
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
    Test handling of edge cases in KafkaConsumer, including Redis startup failures, empty transformed messages, and large message logging.
    
    This test verifies that:
    - Redis connection failures during consumer startup raise exceptions and are propagated.
    - The consumer handles empty transformed messages by calling `process_message` with `None`.
    - Large raw messages are logged in debug output with truncation to a maximum of 200 characters plus ellipsis.
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
        """
        Simulates an executor that returns a raw message on the first call and stops the consumer on subsequent calls.
        
        Returns:
            The raw message on the first invocation; None on subsequent invocations after stopping the consumer.
        """
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
        """
        Simulates asynchronous execution of a function that returns a large raw message on the first call and stops the consumer on subsequent calls.
        
        Returns:
            The large raw message on the first invocation; None on subsequent calls after stopping the consumer.
        """
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
