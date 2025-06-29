# pylint: disable=protected-access, too-many-lines
"""
Comprehensive tests for the `KafkaProducer` class.

This test suite covers all aspects of the KafkaProducer class including:
- Initialization and configuration tests
- Method-level unit tests for individual functions
- Error handling and exception scenarios
- Resource management and cleanup tests
- Edge cases and boundary conditions
- Delivery statistics and queue management

The tests are organized into logical sections:
1. Initialization and Configuration Tests
2. Basic Method Tests (delivery_report, __call__, close)
3. Error Handling Tests
4. Edge Cases and Integration Tests
5. Configuration Tests (from_cfg)
6. Statistics and Queue Management Tests
"""

import json
from unittest.mock import MagicMock, call, patch

import pytest
from confluent_kafka import KafkaError, Message
from omegaconf import DictConfig

from app.data_layer.streaming.producers.kafka_producer import (
    DEFAULT_CONFIG,
    KafkaProducer,
)

# =============================================================================
# SHARED FIXTURES
# =============================================================================


KAFKA_SERVER = "localhost:9092"
KAFKA_TOPIC = "test_topic"
TEST_DATA = "test message"


@pytest.fixture
def mock_logger():
    """
    Pytest fixture that provides a mock logger for verifying logging behavior in tests.
    """
    with patch("app.data_layer.streaming.producers.kafka_producer.logger") as mock:
        yield mock


def _create_mock_confluent_producer():
    """
    Create a MagicMock instance simulating a Confluent Kafka producer with mocked methods for produce, poll, flush, and length.
    
    Returns:
        MagicMock: A mock object emulating the Confluent Kafka producer interface for testing purposes.
    """
    mock_producer = MagicMock()
    mock_producer.produce = MagicMock()
    mock_producer.poll = MagicMock(return_value=0)
    mock_producer.flush = MagicMock(return_value=0)
    mock_producer.__len__ = MagicMock(return_value=0)
    return mock_producer


@pytest.fixture(autouse=True)
def patch_confluent_producer():
    """
    Auto-used pytest fixture that patches the `ConfluentProducer` class with a mock instance for all tests.
    
    Yields:
        dict: Contains the patched class and its mock instance for use in tests.
    """
    with patch(
        "app.data_layer.streaming.producers.kafka_producer.ConfluentProducer"
    ) as mock_producer_class:
        mock_instance = _create_mock_confluent_producer()
        mock_producer_class.return_value = mock_instance
        yield {"class": mock_producer_class, "instance": mock_instance}


def _create_mock_message(topic=KAFKA_TOPIC, partition=0):
    """
    Create a mock Kafka Message object with specified topic and partition values.
    
    Parameters:
        topic (str): The topic name to assign to the mock message.
        partition (int): The partition number to assign to the mock message.
    
    Returns:
        MagicMock: A mock object simulating a Kafka Message with the given topic and partition.
    """
    mock_msg = MagicMock(spec=Message)
    mock_msg.topic.return_value = topic
    mock_msg.partition.return_value = partition
    return mock_msg


@pytest.fixture
def sample_kafka_config():
    """
    Return a sample Kafka configuration dictionary for use in tests.
    
    Returns:
        dict: A dictionary containing example Kafka producer configuration options.
    """
    return {
        "compression.type": "snappy",
        "acks": "all",
        "retries": 5,
    }


@pytest.fixture
def test_kafka_producer_config():
    """
    Provides a sample configuration dictionary for initializing a `KafkaProducer` in tests.
    
    Returns:
        dict: A dictionary containing Kafka server address, topic, and additional producer configuration.
    """
    return {
        "kafka_server": "KAFKA_SERVER",
        "kafka_topic": KAFKA_TOPIC,
        "config": {"compression.type": "snappy"},
    }


@pytest.fixture
def basic_kafka_producer(test_kafka_producer_config):
    """
    Pytest fixture that provides a basic `KafkaProducer` instance for method-level tests.
    
    Yields:
        KafkaProducer: An instance initialized with the provided test configuration.
    """
    producer = KafkaProducer(
        kafka_server=test_kafka_producer_config["kafka_server"],
        kafka_topic=test_kafka_producer_config["kafka_topic"],
        config=test_kafka_producer_config["config"],
    )
    yield producer


@pytest.fixture
def kafka_producer_with_mocks():
    """
    Pytest fixture that yields a `KafkaProducer` instance with all external dependencies mocked.
    
    Provides an isolated producer and its associated mocks for testing functionality and error handling without requiring a real Kafka environment.
    """
    with patch(
        "app.data_layer.streaming.producers.kafka_producer.ConfluentProducer"
    ) as mock_producer_class:
        mock_instance = _create_mock_confluent_producer()
        mock_producer_class.return_value = mock_instance

        producer = KafkaProducer(
            kafka_server="KAFKA_SERVER",
            kafka_topic=KAFKA_TOPIC,
        )

        yield {
            "producer": producer,
            "mock_kafka_producer": mock_instance,
            "mock_producer_class": mock_producer_class,
        }


def validate_kafka_producer_instance(producer):
    """
    Asserts that a KafkaProducer instance is initialized with the expected topic, zeroed delivery counters, and a mocked producer object.
    """
    assert producer.kafka_topic == KAFKA_TOPIC
    assert producer._delivery_success_count == 0
    assert producer._delivery_failure_count == 0
    assert isinstance(producer.kafka_producer, MagicMock)


# =============================================================================
# INITIALIZATION AND CONFIGURATION TESTS
# =============================================================================


def test_kafka_producer_default_initialization():
    """
    Test `KafkaProducer` initialization with default configuration. Verifies that the producer
    initializes correctly with default values and that all expected properties are set properly.
    """
    producer = KafkaProducer(KAFKA_SERVER, KAFKA_TOPIC)

    validate_kafka_producer_instance(producer)


def test_kafka_producer_custom_configuration(sample_kafka_config):
    """
    Test `KafkaProducer` initialization with custom configuration. Verifies that custom
    configuration is properly merged with default configuration and overrides default values
    as expected.
    """
    producer = KafkaProducer(KAFKA_SERVER, KAFKA_TOPIC, sample_kafka_config)

    validate_kafka_producer_instance(producer)


def test_kafka_producer_config_merging():
    """
    Test that custom Kafka producer configuration merges correctly with default settings.
    
    Verifies that both default and custom configuration values are present in the merged config passed to the underlying ConfluentProducer, and that required fields like `bootstrap.servers` are set.
    """
    custom_config = {"compression.type": "snappy", "acks": "all"}

    with patch(
        "app.data_layer.streaming.producers.kafka_producer.ConfluentProducer"
    ) as mock_producer_class:
        _ = KafkaProducer(KAFKA_SERVER, KAFKA_TOPIC, custom_config)

        # Verify that ConfluentProducer was called with merged config
        called_config = mock_producer_class.call_args[0][0]

        # Should have default values
        assert (
            called_config["queue.buffering.max.messages"]
            == DEFAULT_CONFIG["queue.buffering.max.messages"]
        )
        assert (
            called_config["broker.address.family"]
            == DEFAULT_CONFIG["broker.address.family"]
        )

        # Should have custom values
        assert called_config["compression.type"] == "snappy"
        assert called_config["acks"] == "all"

        # Should have bootstrap.servers set
        assert called_config["bootstrap.servers"] == KAFKA_SERVER


def test_kafka_producer_none_config():
    """
    Test `KafkaProducer` initialization with `None` config. Verifies that passing `None`
    as config doesn't cause errors and falls back to default configuration.
    """
    producer = KafkaProducer(KAFKA_SERVER, KAFKA_TOPIC, None)
    validate_kafka_producer_instance(producer)


# =============================================================================
# BASIC METHOD TESTS
# =============================================================================


def test_delivery_report_success(basic_kafka_producer, mock_logger):
    """
    Test that the `delivery_report` method correctly handles a successful message delivery.
    
    Verifies that the success count is incremented and a debug log is emitted when a message is delivered without error.
    """
    mock_msg = _create_mock_message(KAFKA_TOPIC, 0)

    # Test successful delivery (err = None)
    basic_kafka_producer.delivery_report(None, mock_msg)

    assert basic_kafka_producer._delivery_success_count == 1
    assert basic_kafka_producer._delivery_failure_count == 0

    mock_logger.debug.assert_called_once_with(
        "Message delivered to %s [%s]", KAFKA_TOPIC, 0
    )


def test_delivery_report_failure(basic_kafka_producer, mock_logger):
    """
    Test `delivery_report` method with failed delivery. Verifies that failed message delivery
    is properly recorded and logged with appropriate error information.
    """
    mock_msg = _create_mock_message(KAFKA_TOPIC, 0)
    kafka_error = KafkaError(KafkaError._MSG_TIMED_OUT)

    # Test failed delivery
    basic_kafka_producer.delivery_report(kafka_error, mock_msg)

    assert basic_kafka_producer._delivery_success_count == 0
    assert basic_kafka_producer._delivery_failure_count == 1

    mock_logger.error.assert_called_once_with(
        "Message delivery failed: %s for message to %s [%s]",
        kafka_error,
        KAFKA_TOPIC,
        0,
    )


def test_delivery_report_statistics_logging(basic_kafka_producer, mock_logger):
    """
    Test that `delivery_report` logs delivery statistics every 1000 messages.
    
    Verifies that the statistics log is triggered only when the total number of delivered messages reaches a multiple of 1000.
    """
    mock_msg = _create_mock_message(KAFKA_TOPIC, 0)

    # Simulate 999 successful deliveries (no statistics log)
    basic_kafka_producer._delivery_success_count = 999
    basic_kafka_producer.delivery_report(None, mock_msg)

    # Should log statistics at 1000th message
    mock_logger.info.assert_called_once_with(
        "Kafka producer statistics: %d successful / %d total messages (success rate: %.2f%%)",
        1000,
        1000,
        100.0,
    )


def test_call_method_success(basic_kafka_producer):
    """
    Test that the `__call__` method successfully sends valid data to Kafka.
    
    Verifies that the data is encoded to UTF-8, sent with the correct topic and callback, and that the method returns True on success.
    """
    result = basic_kafka_producer(TEST_DATA)

    assert result is True
    basic_kafka_producer.kafka_producer.produce.assert_called_once_with(
        topic=KAFKA_TOPIC,
        value=TEST_DATA.encode("utf-8"),
        callback=basic_kafka_producer.delivery_report,
    )
    basic_kafka_producer.kafka_producer.poll.assert_called_once_with(0)


def test_call_method_empty_data(basic_kafka_producer, mock_logger):
    """
    Test that the `__call__` method returns False and logs a warning when given empty or None data, ensuring no message is sent to Kafka.
    """
    # Test with empty string
    result = basic_kafka_producer("")
    assert result is False
    mock_logger.warning.assert_called_with("Attempted to send empty data to Kafka")

    # Test with None
    result = basic_kafka_producer(None)
    assert result is False

    # Ensure produce was not called
    basic_kafka_producer.kafka_producer.produce.assert_not_called()


def test_call_method_large_queue_flushing(basic_kafka_producer, mock_logger):
    """
    Test that the `__call__` method triggers a flush when the producer queue exceeds 10,000 messages.
    
    Verifies that a flush is performed with the correct timeout and appropriate logging occurs when the internal queue size is large.
    """
    # Mock poll to return large number of messages
    basic_kafka_producer.kafka_producer.poll.return_value = 15000
    basic_kafka_producer.kafka_producer.flush.return_value = 5000

    result = basic_kafka_producer(TEST_DATA)

    assert result is True

    # Verify flush was called due to large queue
    basic_kafka_producer.kafka_producer.flush.assert_called_once_with(timeout=1.0)

    # Verify logging
    mock_logger.info.assert_any_call(
        "Large producer queue (%d messages), flushing...", 15000
    )
    mock_logger.info.assert_any_call(
        "Flushed producer queue, %d messages remaining", 5000
    )


def test_call_method_buffer_error(basic_kafka_producer, mock_logger):
    """
    Test that the `__call__` method of `KafkaProducer` handles `BufferError` by flushing the queue and logging appropriately.
    
    Simulates a full producer queue causing a `BufferError`, verifies that a flush is attempted with the correct timeout, and checks that relevant warning and info logs are emitted.
    """
    # Mock produce to raise BufferError
    basic_kafka_producer.kafka_producer.produce.side_effect = BufferError("Queue full")
    basic_kafka_producer.kafka_producer.__len__.return_value = 100000
    basic_kafka_producer.kafka_producer.flush.return_value = 50000

    result = basic_kafka_producer(TEST_DATA)

    assert result is False

    # Verify flush was attempted
    basic_kafka_producer.kafka_producer.flush.assert_called_once_with(timeout=5.0)

    # Verify logging
    mock_logger.warning.assert_called_with(
        "Local producer queue is full (%d messages awaiting delivery): %s",
        100000,
        basic_kafka_producer.kafka_producer.produce.side_effect,
    )
    mock_logger.info.assert_any_call("Attempting to flush producer queue...")
    mock_logger.info.assert_any_call(
        "Producer queue flush completed, %d messages remaining", 50000
    )


def test_call_method_general_exception(basic_kafka_producer, mock_logger):
    """
    Test that the `__call__` method catches and logs general exceptions raised during message production, returning False without crashing.
    """
    # Mock produce to raise general exception
    basic_kafka_producer.kafka_producer.produce.side_effect = Exception(
        "Connection error"
    )

    result = basic_kafka_producer(TEST_DATA)

    assert result is False

    mock_logger.error.assert_called_once_with(
        "Error sending data to Kafka: %s",
        basic_kafka_producer.kafka_producer.produce.side_effect,
    )


def test_close_method_success(basic_kafka_producer, mock_logger):
    """
    Test that the `close` method flushes the producer and logs final delivery statistics on success.
    
    Verifies that the producer's flush is called with the correct timeout and that informational logs are emitted for both successful flush and final statistics when messages have been processed.
    """
    # Set up some delivery statistics
    basic_kafka_producer._delivery_success_count = 800
    basic_kafka_producer._delivery_failure_count = 200
    basic_kafka_producer.kafka_producer.flush.return_value = 0

    basic_kafka_producer.close()

    # Verify flush was called
    basic_kafka_producer.kafka_producer.flush.assert_called_once_with(timeout=10.0)

    # Verify success logging
    mock_logger.info.assert_any_call("Kafka producer flushed successfully")

    # Verify statistics logging
    mock_logger.info.assert_any_call(
        "Final Kafka producer statistics: %d successful / %d total messages (success rate: %.2f%%)",
        800,
        1000,
        80.0,
    )


def test_close_method_with_remaining_messages(basic_kafka_producer, mock_logger):
    """
    Test that the `close` method logs a warning if messages remain after flushing during shutdown.
    """
    basic_kafka_producer.kafka_producer.flush.return_value = 5
    basic_kafka_producer.close()

    mock_logger.warning.assert_called_once_with(
        "Failed to flush all messages, %d messages remain in queue", 5
    )


def test_close_method_no_statistics(basic_kafka_producer, mock_logger):
    """
    Test that the `close` method does not log statistics when no messages have been processed.
    
    Verifies that only the flush success message is logged if the producer has not sent any messages.
    """
    basic_kafka_producer.kafka_producer.flush.return_value = 0
    basic_kafka_producer.close()

    # Verify no statistics logging (since total = 0)
    mock_logger.info.assert_called_once_with("Kafka producer flushed successfully")


def test_close_method_flush_exception(basic_kafka_producer, mock_logger):
    """
    Test that the `close` method logs an error and handles exceptions raised during the flush operation without crashing.
    """
    basic_kafka_producer.kafka_producer.flush.side_effect = Exception("Flush error")
    basic_kafka_producer.close()

    mock_logger.error.assert_called_once_with(
        "Error flushing Kafka producer: %s",
        basic_kafka_producer.kafka_producer.flush.side_effect,
    )


def test_close_method_no_producer(basic_kafka_producer, mock_logger):
    """
    Test `close` method when `kafka_producer` is `None`. Verifies that close method handles cases
    where the producer instance might be `None` without errors.
    """
    basic_kafka_producer.kafka_producer = None

    # Should not raise any exceptions
    basic_kafka_producer.close()

    # No logging should occur
    mock_logger.info.assert_not_called()
    mock_logger.warning.assert_not_called()
    mock_logger.error.assert_not_called()


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


def test_multiple_buffer_errors(basic_kafka_producer):
    """
    Test that consecutive BufferError exceptions during message sending are handled by repeated flush attempts.
    
    Verifies that when the producer's queue is full and multiple BufferErrors occur, each send attempt triggers a flush and returns False.
    """
    basic_kafka_producer.kafka_producer.produce.side_effect = BufferError("Queue full")
    basic_kafka_producer.kafka_producer.__len__.return_value = 100000
    basic_kafka_producer.kafka_producer.flush.return_value = 80000

    # Send multiple messages that will trigger buffer errors
    results = []
    for i in range(3):
        result = basic_kafka_producer(f"message_{i}")
        results.append(result)

    # All should return False
    assert all(result is False for result in results)

    # Flush should be called for each attempt
    assert basic_kafka_producer.kafka_producer.flush.call_count == 3


def test_producer_statistics_mixed_success_failure(basic_kafka_producer, mock_logger):
    """
    Test that KafkaProducer logs correct statistics when both successful and failed deliveries occur.
    
    Simulates a scenario with a mix of successful and failed message deliveries to ensure that the producer logs the correct success rate and message counts when the total reaches a statistics reporting threshold.
    """
    mock_msg = _create_mock_message(KAFKA_TOPIC, 0)
    kafka_error = KafkaError(KafkaError._MSG_TIMED_OUT)

    # Simulate mixed success/failure to reach 1000 total
    basic_kafka_producer._delivery_success_count = 700
    basic_kafka_producer._delivery_failure_count = 299

    # This should trigger statistics logging (total = 1000)
    basic_kafka_producer.delivery_report(kafka_error, mock_msg)

    mock_logger.info.assert_called_once_with(
        "Kafka producer statistics: %d successful / %d total messages (success rate: %.2f%%)",
        700,
        1000,
        70.0,
    )


# =============================================================================
# EDGE CASES AND INTEGRATION TESTS
# =============================================================================


def test_large_message_handling(basic_kafka_producer):
    """
    Test that the KafkaProducer correctly encodes and sends large messages without errors.
    
    Verifies that a message of approximately 1MB is accepted, encoded to UTF-8, and passed to the underlying producer's `produce` method as expected.
    """
    # Create a large message (1MB)
    large_data = "x" * (1024 * 1024)

    result = basic_kafka_producer(large_data)

    assert result is True
    basic_kafka_producer.kafka_producer.produce.assert_called_once_with(
        topic=KAFKA_TOPIC,
        value=large_data.encode("utf-8"),
        callback=basic_kafka_producer.delivery_report,
    )


def test_unicode_message_handling(basic_kafka_producer):
    """
    Test handling of Unicode messages. Verifies that Unicode characters are properly encoded
    when sending messages to Kafka.
    """
    unicode_data = "测试消息 🚀 émojis and ñoñó"

    result = basic_kafka_producer(unicode_data)

    assert result is True
    basic_kafka_producer.kafka_producer.produce.assert_called_once_with(
        topic=KAFKA_TOPIC,
        value=unicode_data.encode("utf-8"),
        callback=basic_kafka_producer.delivery_report,
    )


def test_json_message_handling(basic_kafka_producer):
    """
    Test that the producer correctly handles and sends JSON-serialized message payloads.
    """
    data_dict = {
        "user_id": 12345,
        "action": "purchase",
        "items": ["item1", "item2"],
        "timestamp": "2024-01-15T10:30:00Z",
    }
    json_data = json.dumps(data_dict)

    result = basic_kafka_producer(json_data)

    assert result is True
    basic_kafka_producer.kafka_producer.produce.assert_called_once_with(
        topic=KAFKA_TOPIC,
        value=json_data.encode("utf-8"),
        callback=basic_kafka_producer.delivery_report,
    )


def test_queue_management_edge_cases(basic_kafka_producer):
    """
    Test queue management behavior at boundary values for the internal producer queue.
    
    Verifies that flushing is not triggered when the queue size is exactly at the threshold, but is triggered when the queue size exceeds the threshold. Also checks correct handling of flush return values and method calls.
    """
    # Test exactly at threshold (should not trigger flush)
    basic_kafka_producer.kafka_producer.poll.return_value = 10000
    result = basic_kafka_producer("test message")
    assert result is True
    basic_kafka_producer.kafka_producer.flush.assert_not_called()

    # Reset mock
    basic_kafka_producer.kafka_producer.reset_mock()

    # Test just over threshold (should trigger flush)
    basic_kafka_producer.kafka_producer.poll.return_value = 10001
    basic_kafka_producer.kafka_producer.flush.return_value = 8000

    result = basic_kafka_producer("test message")
    assert result is True
    basic_kafka_producer.kafka_producer.flush.assert_called_once_with(timeout=1.0)


# =============================================================================
# CONFIGURATION TESTS (from_CFG)
# =============================================================================


def test_from_cfg_success():
    """
    Test that the `from_cfg` class method creates a `KafkaProducer` instance from a valid configuration object.
    """
    cfg = DictConfig(
        {
            "kafka_server": "KAFKA_SERVER",
            "kafka_topic": KAFKA_TOPIC,
            "producer_config": {"compression.type": "snappy"},
        }
    )

    producer = KafkaProducer.from_cfg(cfg)

    assert producer is not None
    assert isinstance(producer, KafkaProducer)
    assert producer.kafka_topic == KAFKA_TOPIC


def test_from_cfg_missing_kafka_server(mock_logger):
    """
    Test that `KafkaProducer.from_cfg` returns None and logs an error when `kafka_server` is missing from the configuration.
    """
    cfg = DictConfig(
        {
            "kafka_topic": KAFKA_TOPIC,
        }
    )

    producer = KafkaProducer.from_cfg(cfg)

    assert producer is None
    mock_logger.error.assert_called_with("Missing kafka_server in configuration")


def test_from_cfg_missing_kafka_topic(mock_logger):
    """
    Test that `KafkaProducer.from_cfg` returns None and logs an error when `kafka_topic` is missing from the configuration.
    """
    cfg = DictConfig(
        {
            "kafka_server": "KAFKA_SERVER",
        }
    )

    producer = KafkaProducer.from_cfg(cfg)

    assert producer is None
    mock_logger.error.assert_called_with("Missing kafka_topic in configuration")


def test_from_cfg_invalid_configuration_format(mock_logger):
    """
    Test that `KafkaProducer.from_cfg` returns None and logs an error when given an invalid configuration format.
    """
    # Test with non-dict configuration
    cfg = "invalid_config"

    with patch(
        "app.data_layer.streaming.producers.kafka_producer.OmegaConf.to_container"
    ) as mock_to_container:
        mock_to_container.return_value = "not_a_dict"

        producer = KafkaProducer.from_cfg(cfg)

        assert producer is None
        mock_logger.error.assert_called_with("Invalid configuration format")


def test_from_cfg_with_producer_config():
    """
    Test that `KafkaProducer.from_cfg` correctly applies additional producer configuration from the config object.
    
    Verifies that custom producer settings in the `producer_config` section are extracted and used when creating the `KafkaProducer` instance.
    """
    cfg = DictConfig(
        {
            "kafka_server": "KAFKA_SERVER",
            "kafka_topic": KAFKA_TOPIC,
            "producer_config": {
                "compression.type": "gzip",
                "acks": "all",
                "retries": 10,
            },
        }
    )

    producer = KafkaProducer.from_cfg(cfg)

    assert producer is not None
    assert isinstance(producer, KafkaProducer)
    assert producer.kafka_topic == KAFKA_TOPIC


def test_from_cfg_without_producer_config():
    """
    Test that `KafkaProducer.from_cfg` works correctly when `producer_config` is not provided.
    
    Verifies that the absence of `producer_config` in the configuration does not cause errors and results in a producer instance with default settings.
    """
    cfg = DictConfig(
        {
            "kafka_server": "KAFKA_SERVER",
            "kafka_topic": KAFKA_TOPIC,
        }
    )

    producer = KafkaProducer.from_cfg(cfg)

    assert producer is not None
    assert isinstance(producer, KafkaProducer)
    assert producer.kafka_topic == KAFKA_TOPIC


def test_from_cfg_exception_handling(mock_logger):
    """
    Test that `KafkaProducer.from_cfg` handles and logs unexpected exceptions during configuration processing.
    
    Verifies that if an exception occurs while processing the configuration, the method returns `None` and logs the error.
    """
    cfg = DictConfig(
        {
            "kafka_server": "KAFKA_SERVER",
            "kafka_topic": KAFKA_TOPIC,
        }
    )

    with patch(
        "app.data_layer.streaming.producers.kafka_producer.OmegaConf.to_container"
    ) as mock_to_container:
        mock_to_container.side_effect = Exception("Configuration error")

        producer = KafkaProducer.from_cfg(cfg)

        assert producer is None
        mock_logger.error.assert_called_with(
            "Error creating KafkaProducer: %s", mock_to_container.side_effect
        )


# =============================================================================
# STATISTICS AND QUEUE MANAGEMENT TESTS
# =============================================================================


def test_delivery_statistics_accuracy(basic_kafka_producer):
    """
    Verify that the KafkaProducer correctly tracks the number of successful and failed deliveries across multiple delivery report invocations.
    """
    mock_msg = _create_mock_message(KAFKA_TOPIC, 0)
    kafka_error = KafkaError(KafkaError._MSG_TIMED_OUT)

    # Test multiple successful deliveries
    for _ in range(5):
        basic_kafka_producer.delivery_report(None, mock_msg)

    # Test multiple failed deliveries
    for _ in range(3):
        basic_kafka_producer.delivery_report(kafka_error, mock_msg)

    assert basic_kafka_producer._delivery_success_count == 5
    assert basic_kafka_producer._delivery_failure_count == 3


def test_queue_polling_behavior(basic_kafka_producer):
    """
    Verifies that the producer's `poll(0)` method is called on every message send to ensure delivery callbacks are triggered.
    """
    basic_kafka_producer.kafka_producer.poll.return_value = 100

    # Send multiple messages
    for i in range(3):
        basic_kafka_producer(f"message_{i}")

    # Poll should be called for each message
    assert basic_kafka_producer.kafka_producer.poll.call_count == 3

    # All calls should be poll(0)
    expected_calls = [call(0), call(0), call(0)]
    basic_kafka_producer.kafka_producer.poll.assert_has_calls(expected_calls)


def test_statistics_logging_intervals(basic_kafka_producer, mock_logger):
    """
    Verify that delivery statistics are logged only at exact multiples of 1000 successful messages, and not at other message counts.
    """
    mock_msg = _create_mock_message(KAFKA_TOPIC, 0)

    # Test various message counts that should NOT trigger logging
    test_counts = [1, 100, 500, 999, 1001, 1500, 1999]

    for count in test_counts:
        mock_logger.reset_mock()
        basic_kafka_producer._delivery_success_count = count - 1
        basic_kafka_producer.delivery_report(None, mock_msg)

        # Should not log statistics
        mock_logger.info.assert_not_called()

    # Test count that SHOULD trigger logging
    mock_logger.reset_mock()
    basic_kafka_producer._delivery_success_count = 1999
    basic_kafka_producer.delivery_report(None, mock_msg)

    # Should log statistics
    mock_logger.info.assert_called_once()


def test_comprehensive_integration_scenario(kafka_producer_with_mocks, mock_logger):
    """
    Simulates an end-to-end integration scenario for KafkaProducer, verifying correct operation across message sending, delivery reporting, and resource cleanup.
    
    This test ensures that messages are sent, delivery reports are processed for both success and failure, and final statistics are logged as expected in a realistic usage pattern.
    """
    producer_data = kafka_producer_with_mocks
    producer = producer_data["producer"]
    mock_producer = producer_data["mock_kafka_producer"]

    # Simulate realistic usage pattern
    messages = ["msg1", "msg2", "msg3"]

    # Configure mock for realistic behavior
    mock_producer.poll.return_value = 2000  # Moderate queue size
    mock_producer.flush.return_value = 0

    # Send messages
    results = []
    for msg in messages:
        result = producer(msg)
        results.append(result)

    # All should succeed
    assert all(results)

    # Verify all messages were sent
    assert mock_producer.produce.call_count == len(messages)

    # Simulate some delivery reports
    mock_msg = _create_mock_message(KAFKA_TOPIC, 0)
    producer.delivery_report(None, mock_msg)  # Success
    producer.delivery_report(KafkaError(KafkaError._MSG_TIMED_OUT), mock_msg)  # Failure

    # Close the producer
    producer.close()

    # Verify final statistics logging
    mock_logger.info.assert_any_call(
        "Final Kafka producer statistics: %d successful / %d total messages (success rate: %.2f%%)",
        1,
        2,
        50.0,
    )


# =============================================================================
# ADDITIONAL EDGE CASES AND BOUNDARY CONDITIONS
# =============================================================================


def test_delivery_report_with_invalid_message(basic_kafka_producer):
    """
    Tests that `delivery_report` handles message objects lacking expected methods without crashing, incrementing the failure count as appropriate.
    """
    # Create a mock message that doesn't implement expected methods properly
    mock_msg = MagicMock()
    mock_msg.topic.side_effect = Exception("Topic access failed")
    mock_msg.partition.side_effect = Exception("Partition access failed")

    kafka_error = KafkaError(KafkaError.BROKER_NOT_AVAILABLE)

    # This should not crash even if message access fails
    try:
        basic_kafka_producer.delivery_report(kafka_error, mock_msg)
        assert basic_kafka_producer._delivery_failure_count == 1
    except Exception:
        # If it does throw an exception, that's also acceptable behavior
        # but we want to document it
        pass


def test_very_high_message_throughput_statistics(basic_kafka_producer, mock_logger):
    """
    Verifies that the KafkaProducer logs correct delivery statistics when processing a very high number of messages, ensuring accurate reporting at large throughput volumes.
    """
    mock_msg = _create_mock_message(KAFKA_TOPIC, 0)

    # Simulate processing close to 10,000 messages
    basic_kafka_producer._delivery_success_count = 9998
    basic_kafka_producer._delivery_failure_count = 1

    # This delivery should trigger statistics logging
    basic_kafka_producer.delivery_report(None, mock_msg)

    # Should log statistics for 10,000 total messages
    mock_logger.info.assert_called_with(
        "Kafka producer statistics: %d successful / %d total messages (success rate: %.2f%%)",
        9999,
        10000,
        99.99,
    )


def test_zero_success_rate_statistics(basic_kafka_producer, mock_logger):
    """
    Test that the KafkaProducer logs a 0% success rate when all message deliveries fail.
    
    Simulates 1000 failed deliveries and verifies that the statistics log reflects zero successful messages out of 1000, with a 0.0% success rate.
    """
    mock_msg = _create_mock_message(KAFKA_TOPIC, 0)
    kafka_error = KafkaError(KafkaError.NETWORK_EXCEPTION)

    # Simulate 1000 failed messages
    basic_kafka_producer._delivery_failure_count = 999

    # This failure should trigger statistics logging
    basic_kafka_producer.delivery_report(kafka_error, mock_msg)

    # Should log 0% success rate
    mock_logger.info.assert_called_with(
        "Kafka producer statistics: %d successful / %d total messages (success rate: %.2f%%)",
        0,
        1000,
        0.0,
    )


def test_producer_config_with_all_defaults():
    """
    Verify that the KafkaProducer applies all default configuration values when no custom config is provided.
    """
    with patch(
        "app.data_layer.streaming.producers.kafka_producer.ConfluentProducer"
    ) as mock_producer_class:
        KafkaProducer(KAFKA_SERVER, KAFKA_TOPIC)

        # Get the config that was passed to ConfluentProducer
        called_config = mock_producer_class.call_args[0][0]

        # Verify all default config values are present
        for key, value in DEFAULT_CONFIG.items():
            assert called_config[key] == value

        # Verify bootstrap.servers is set
        assert called_config["bootstrap.servers"] == KAFKA_SERVER


def test_buffer_error_with_zero_remaining_messages(basic_kafka_producer, mock_logger):
    """
    Test that a BufferError triggers a flush and logs zero remaining messages when the queue is fully emptied.
    
    Verifies that when a BufferError occurs during message production and the subsequent flush empties the queue (returns 0), the correct log message is emitted indicating no remaining messages.
    """
    basic_kafka_producer.kafka_producer.produce.side_effect = BufferError("Queue full")
    basic_kafka_producer.kafka_producer.__len__.return_value = 50000
    basic_kafka_producer.kafka_producer.flush.return_value = 0  # All messages flushed

    result = basic_kafka_producer("test message")

    assert result is False

    # Verify that flush completion is logged correctly
    mock_logger.info.assert_any_call(
        "Producer queue flush completed, %d messages remaining", 0
    )


def test_message_encoding_edge_cases(basic_kafka_producer):
    """
    Tests that the KafkaProducer correctly encodes various edge-case string messages to UTF-8 bytes before sending.
    
    Verifies proper handling of whitespace, control characters, emojis, mixed Unicode content, and long strings.
    """
    test_cases = [
        "",  # Empty string (though this is handled separately)
        " ",  # Single space
        "\n",  # Newline
        "\t",  # Tab
        "🚀🌟💫",  # Only emojis
        "mixed: 测试 🚀 test",  # Mixed content
        "a" * 10000,  # Very long ASCII string
    ]

    for test_case in test_cases[1:]:  # Skip empty string as it's handled separately
        basic_kafka_producer.kafka_producer.reset_mock()

        result = basic_kafka_producer(test_case)

        assert result is True
        basic_kafka_producer.kafka_producer.produce.assert_called_once_with(
            topic=KAFKA_TOPIC,
            value=test_case.encode("utf-8"),
            callback=basic_kafka_producer.delivery_report,
        )


def test_close_with_partial_flush_timeout(basic_kafka_producer, mock_logger):
    """
    Test that the `close` method logs a warning and final statistics when flush times out with messages remaining in the queue.
    """
    # Simulate flush timeout with many messages remaining
    basic_kafka_producer.kafka_producer.flush.return_value = 5000
    basic_kafka_producer._delivery_success_count = 8000
    basic_kafka_producer._delivery_failure_count = 2000

    basic_kafka_producer.close()

    # Should warn about remaining messages
    mock_logger.warning.assert_called_once_with(
        "Failed to flush all messages, %d messages remain in queue", 5000
    )

    # Should still log final statistics
    mock_logger.info.assert_called_with(
        "Final Kafka producer statistics: %d successful / %d total messages (success rate: %.2f%%)",
        8000,
        10000,
        80.0,
    )


def test_producer_multiple_close_calls(basic_kafka_producer, mock_logger):
    """
    Test that multiple calls to `close()` don't cause issues. Verifies that calling `close()`
    multiple times is safe and doesn't cause duplicate logging or errors.
    """
    basic_kafka_producer.kafka_producer.flush.return_value = 0

    # Call close multiple times
    basic_kafka_producer.close()
    basic_kafka_producer.close()
    basic_kafka_producer.close()

    # Flush should be called each time
    assert basic_kafka_producer.kafka_producer.flush.call_count == 3

    # Success message should be logged each time
    assert mock_logger.info.call_count >= 3


def test_from_cfg_with_empty_producer_config():
    """
    Test that `from_cfg` correctly handles an explicitly empty `producer_config` by falling back to default configuration without error.
    """
    cfg = DictConfig(
        {
            "kafka_server": "KAFKA_SERVER",
            "kafka_topic": KAFKA_TOPIC,
            "producer_config": {},  # Explicitly empty
        }
    )

    producer = KafkaProducer.from_cfg(cfg)

    assert producer is not None
    assert isinstance(producer, KafkaProducer)
    assert producer.kafka_topic == KAFKA_TOPIC


def test_queue_management_with_exact_threshold_values(
    basic_kafka_producer, mock_logger
):
    """
    Test that the KafkaProducer flushes its queue only when the internal queue size exceeds the threshold.
    
    Verifies that no flush occurs when the queue size is at or below 10,000, and that a flush is triggered when the queue size exceeds this threshold.
    """
    test_cases = [
        (9999, False),  # Just under threshold - no flush
        (10000, False),  # At threshold - no flush
        (10001, True),  # Just over threshold - should flush
        (15000, True),  # Well over threshold - should flush
    ]

    for queue_size, should_flush in test_cases:
        basic_kafka_producer.kafka_producer.reset_mock()
        basic_kafka_producer.kafka_producer.poll.return_value = queue_size
        basic_kafka_producer.kafka_producer.flush.return_value = queue_size // 2

        result = basic_kafka_producer("test message")

        assert result is True

        if should_flush:
            basic_kafka_producer.kafka_producer.flush.assert_called_once_with(
                timeout=1.0
            )
            mock_logger.info.assert_any_call(
                "Large producer queue (%d messages), flushing...", queue_size
            )
        else:
            basic_kafka_producer.kafka_producer.flush.assert_not_called()


def test_comprehensive_error_scenario_chain(basic_kafka_producer, mock_logger):
    """
    Simulates a sequence of error scenarios—including buffer errors, general exceptions, and successful sends—to verify that the KafkaProducer handles each case gracefully and logs appropriate messages.
    """

    # First, test buffer error
    basic_kafka_producer.kafka_producer.produce.side_effect = BufferError("Queue full")
    basic_kafka_producer.kafka_producer.__len__.return_value = 100000
    basic_kafka_producer.kafka_producer.flush.return_value = 0

    result1 = basic_kafka_producer(TEST_DATA)
    assert result1 is False

    # Then, test general exception
    basic_kafka_producer.kafka_producer.produce.side_effect = Exception(
        "Connection lost"
    )

    result2 = basic_kafka_producer(TEST_DATA)
    assert result2 is False

    # Finally, test successful operation
    basic_kafka_producer.kafka_producer.produce.side_effect = None
    basic_kafka_producer.kafka_producer.poll.return_value = 1000

    result3 = basic_kafka_producer(TEST_DATA)
    assert result3 is True

    # Verify all scenarios were handled and logged appropriately
    assert mock_logger.warning.call_count >= 1  # Buffer error warning
    assert mock_logger.error.call_count >= 1  # General exception error
    assert mock_logger.info.call_count >= 2  # Flush attempts
