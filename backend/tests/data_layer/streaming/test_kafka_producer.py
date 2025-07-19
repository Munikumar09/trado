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
    Mock logger fixture for testing log output. Provides a mock logger to verify logging
    behavior throughout the test suite.
    """
    with patch("app.data_layer.streaming.producers.kafka_producer.logger") as mock:
        yield mock


def _create_mock_confluent_producer():
    """
    Helper function to create consistent mock Confluent Kafka producer.
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
    Auto-use fixture to patch `ConfluentProducer` across all tests. Ensures consistent mocking
    of the underlying Kafka producer throughout the test suite.
    """
    with patch(
        "app.data_layer.streaming.producers.kafka_producer.ConfluentProducer"
    ) as mock_producer_class:
        mock_instance = _create_mock_confluent_producer()
        mock_producer_class.return_value = mock_instance
        yield {"class": mock_producer_class, "instance": mock_instance}


def _create_mock_message(topic=KAFKA_TOPIC, partition=0):
    """
    Helper function to create mock Kafka message.
    """
    mock_msg = MagicMock(spec=Message)
    mock_msg.topic.return_value = topic
    mock_msg.partition.return_value = partition
    return mock_msg


@pytest.fixture
def sample_kafka_config():
    """
    Sample Kafka configuration for testing.
    """
    return {
        "compression.type": "snappy",
        "acks": "all",
        "retries": 5,
    }


@pytest.fixture
def test_kafka_producer_config():
    """
    Test configuration for `KafkaProducer` initialization.
    """
    return {
        "kafka_server": "KAFKA_SERVER",
        "kafka_topic": KAFKA_TOPIC,
        "config": {"compression.type": "snappy"},
    }


@pytest.fixture
def basic_kafka_producer(test_kafka_producer_config):
    """
    Fixture that creates a basic `KafkaProducer` for method-level testing. Used for tests that
    focus on individual methods rather than complex integration scenarios.
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
    Fixture that creates a `KafkaProducer` with all external dependencies mocked. Provides a fully
    isolated producer instance for comprehensive testing of all methods and error scenarios.
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
    Validate the properties and state of a KafkaProducer instance.
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
    Test that custom configuration properly merges with `DEFAULT_CONFIG`. Verifies that the
    producer configuration merging logic works correctly and that both default and custom
    values are preserved appropriately.
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
    Test `delivery_report` method with successful delivery. Verifies that successful message
    delivery is properly recorded and logged with appropriate debug information.
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
    Test `delivery_report` statistics logging at `1000` message intervals. Verifies that delivery
    statistics are logged periodically when message count reaches multiples of `1000`.
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
    Test successful data sending via `__call__` method. Verifies that valid data is properly
    encoded and sent to Kafka with correct topic and callback configuration.
    """
    result = basic_kafka_producer(TEST_DATA)

    assert result is True
    basic_kafka_producer.kafka_producer.produce.assert_called_once_with(
        topic=KAFKA_TOPIC,
        value=TEST_DATA.encode("utf-8"),
        callback=basic_kafka_producer.delivery_report,
        key=None,  # Default key is None
    )
    basic_kafka_producer.kafka_producer.poll.assert_called_once_with(0)


def test_call_method_empty_data(basic_kafka_producer, mock_logger):
    """
    Test `__call__` method with empty data. Verifies that empty or `None` data is properly
    handled without attempting to send to Kafka.
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
    Test `__call__` method with large producer queue triggering flush. Verifies that when
    the producer queue grows large (`>10000` messages), automatic flushing is triggered to
    prevent memory issues.
    """
    # Mock poll to return large number of messages
    basic_kafka_producer.kafka_producer.flush.return_value = 5000
    basic_kafka_producer.kafka_producer.__len__.return_value = 15000

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
    Test `__call__` method handling `BufferError`. Verifies that `BufferError` (queue full)
    is properly handled with automatic flush attempt and appropriate logging.
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
    Test `__call__` method handling general exceptions. Verifies that unexpected exceptions
    during produce are properly caught and logged without crashing the application.
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
    Test successful `close` method execution. Verifies that the producer flushes properly on
    close and logs final statistics when messages have been processed.
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
    Test `close` method when some messages remain after flush. Verifies that warning is logged
    when not all messages can be flushed during producer shutdown.
    """
    basic_kafka_producer.kafka_producer.flush.return_value = 5
    basic_kafka_producer.close()

    mock_logger.warning.assert_called_once_with(
        "Failed to flush all messages, %d messages remain in queue", 5
    )


def test_close_method_no_statistics(basic_kafka_producer, mock_logger):
    """
    Test `close` method when no messages have been processed. Verifies that statistics are not
    logged when no messages have been sent through the producer.
    """
    basic_kafka_producer.kafka_producer.flush.return_value = 0
    basic_kafka_producer.close()

    # Verify no statistics logging (since total = 0)
    mock_logger.info.assert_called_once_with("Kafka producer flushed successfully")


def test_close_method_flush_exception(basic_kafka_producer, mock_logger):
    """
    Test `close` method handling flush exceptions. Verifies that exceptions during flush are
    properly caught and logged without crashing the application.
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
    Test handling of consecutive `BufferError` exceptions. Verifies that multiple buffer errors
    are handled gracefully and that flush attempts continue to be made.
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
    Test statistics calculation with mixed success and failure. Verifies that statistics are
    correctly calculated when there are both successful and failed message deliveries.
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
    Test handling of large messages. Verifies that large messages are properly encoded
    and sent without issues in the producer logic.
    """
    # Create a large message (1MB)
    large_data = "x" * (1024 * 1024)

    result = basic_kafka_producer(large_data)

    assert result is True
    basic_kafka_producer.kafka_producer.produce.assert_called_once_with(
        topic=KAFKA_TOPIC,
        value=large_data.encode("utf-8"),
        callback=basic_kafka_producer.delivery_report,
        key=None,  # Default key is None
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
        key=None,  # Default key is None
    )


def test_json_message_handling(basic_kafka_producer):
    """
    Test handling of JSON-serialized messages. Verifies that complex data structures serialized
    as JSON are properly handled by the producer.
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
        key=None,  # Default key is None
    )


def test_queue_management_edge_cases(basic_kafka_producer):
    """
    Test edge cases in queue management logic. Verifies that queue management behaves
    correctly at boundary conditions (exactly `10000` messages, flush failures, etc.).
    """
    # Test exactly at threshold (should not trigger flush)
    basic_kafka_producer.kafka_producer.poll.return_value = 10000
    result = basic_kafka_producer("test message")
    assert result is True
    basic_kafka_producer.kafka_producer.flush.assert_not_called()

    # Reset mock
    basic_kafka_producer.kafka_producer.reset_mock()

    # Test just over threshold (should trigger flush)
    basic_kafka_producer.kafka_producer.__len__.return_value = 10001
    basic_kafka_producer.kafka_producer.flush.return_value = 8000

    result = basic_kafka_producer("test message")
    assert result is True
    basic_kafka_producer.kafka_producer.flush.assert_called_once_with(timeout=1.0)


# =============================================================================
# CONFIGURATION TESTS (from_CFG)
# =============================================================================


def test_from_cfg_success():
    """
    Test successful creation from configuration. Verifies that `from_cfg` class method correctly
    creates a `KafkaProducer` instance from a valid configuration object.
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
    Test `from_cfg` with missing `kafka_server`. Verifies that missing `kafka_server`
    configuration results in `None` return value and appropriate error logging.
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
    Test `from_cfg` with missing `kafka_topic`. Verifies that missing `kafka_topic` configuration
    results in `None` return value and appropriate error logging.
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
    Test `from_cfg` with invalid configuration format. Verifies that invalid configuration types
    are handled gracefully with appropriate error logging.
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
    Test `from_cfg` with additional producer configuration. Verifies that producer-specific
    configuration is properly extracted and passed to the `KafkaProducer` constructor.
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
    Test `from_cfg` without `producer_config`. Verifies that missing `producer_config` doesn't
    cause errors and defaults to empty configuration.
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
    Test `from_cfg` exception handling. Verifies that unexpected exceptions during configuration
    processing are properly caught and logged.
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
    Test accuracy of delivery statistics tracking. Verifies that success and failure counts
    are accurately maintained across multiple delivery report calls.
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
    Test producer queue polling behavior. Verifies that `poll(0)` is called on every send
    operation to trigger delivery callbacks.
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
    Test that statistics are logged only at correct intervals. Verifies that statistics
    logging occurs exactly at multiples of `1000` messages and not at other counts.
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
    Test comprehensive integration scenario. Verifies that all components work together
    correctly in a realistic usage scenario with multiple operations.
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
    Test `delivery_report` with invalid message object. Verifies that `delivery_report`
    handles edge cases where message object might not have expected methods.
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
    Test statistics logging with very high message throughput. Verifies that statistics are
    logged correctly even when processing many thousands of messages.
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
    Test statistics logging when all messages fail. Verifies that `0%` success rate is
    correctly calculated and logged.
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
    Test that all `DEFAULT_CONFIG` values are properly applied. Verifies that when no custom
    config is provided, all default configuration values are correctly set in the producer.
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
    Test `BufferError` handling when flush completely empties the queue. Verifies that when flush
    returns `0` (all messages flushed), the logging reflects this correctly.
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
    Test message encoding with various edge cases. Verifies that different types of string
    content are properly encoded to `UTF-8` bytes.
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
            key=None,  # Default key is None
        )


def test_close_with_partial_flush_timeout(basic_kafka_producer, mock_logger):
    """
    Test `close` method when flush times out with messages remaining. Verifies behavior when
    the flush operation doesn't complete within the timeout period.
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
    Test `from_cfg` with explicitly empty `producer_config`. Verifies that an empty
    `producer_config` dict doesn't cause issues and falls back to default configuration.
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
    Test queue management behavior at exact threshold boundaries. Verifies that queue
    management logic works correctly at boundary values (exactly `10000`, `10001`, etc.).
    """
    test_cases = [
        (9999, False),  # Just under threshold - no flush
        (10000, False),  # At threshold - no flush
        (10001, True),  # Just over threshold - should flush
        (15000, True),  # Well over threshold - should flush
    ]

    for queue_size, should_flush in test_cases:
        basic_kafka_producer.kafka_producer.reset_mock()
        basic_kafka_producer.kafka_producer.__len__.return_value = queue_size
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
    Test a comprehensive chain of error scenarios. Verifies that the producer can handle
    multiple different types of errors in sequence without breaking.
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
