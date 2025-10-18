# pylint: disable=protected-access
"""
Comprehensive tests for the KafkaConsumer class.
"""

import asyncio
import json
from typing import Any
from omegaconf import DictConfig
from app.cache.instrument_cache import CacheUpdateError
from pytest_mock import MockerFixture
from unittest.mock import AsyncMock, MagicMock
from app.core.config import settings
from app.utils.common import init_from_cfg
from redis.exceptions import ConnectionError as RedisConnectionError
import pytest
from confluent_kafka import KafkaException
from app.data_layer.streaming.consumers.kafka_consumer import (
    KafkaConsumer,
    DEFAULT_CONFIG,
)

# =============================================================================
# SHARED FIXTURES
# =============================================================================


@pytest.fixture
def mock_logger(mocker: MockerFixture):
    """
    Unified fixture that provides a mock logger for testing log output.
    This fixture is shared across all tests and ensures consistent logging
    verification throughout the test suite.
    """
    return mocker.patch("app.data_layer.streaming.consumers.kafka_consumer.logger")


@pytest.fixture(autouse=True)
def mock_redis_connection(mocker: MockerFixture):
    """
    Mock Redis connection fixture for testing Redis integration. Provides a mock Redis
    connection and client for isolated testing.
    """
    fake_connection = mocker.patch(
        "app.data_layer.streaming.consumers.kafka_consumer.RedisAsyncConnection",
    )
    mock_redis_client = AsyncMock()
    fake_build = AsyncMock()
    fake_build.get_connection.return_value = mock_redis_client
    fake_connection.build.return_value = fake_build

    return {"client": mock_redis_client, "connection": fake_connection}


@pytest.fixture
def mock_update_stock_cache(mocker: MockerFixture):
    """
    Fixture that mocks the update_stock_cache function. Isolates cache updating logic
    from message consumption logic.
    """
    return mocker.patch(
        "app.data_layer.streaming.consumers.kafka_consumer.update_stock_cache",
        new_callable=AsyncMock,
    )


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
def mock_confluent_consumer(mocker):
    """
    Auto-used fixture that patches the Confluent Kafka Consumer. This ensures all tests
    use mocked Kafka consumers instead of real ones, preventing external dependencies
    and making tests deterministic.
    """
    return mocker.patch(
        "app.data_layer.streaming.consumers.kafka_consumer.ConfluentConsumer",
        autospec=True,
        return_value=_create_mock_confluent_consumer(),
    )


@pytest.fixture(autouse=True)
def patch_redis_utils(mocker):
    """
    Auto-used fixture that patches Redis utilities. Ensures Redis operations are mocked
    consistently across all tests.
    """
    redis_instance = AsyncMock()
    mock_redis = mocker.patch("app.utils.redis_utils.redis", new=MagicMock())
    mock_async_redis = mocker.patch(
        "app.utils.redis_utils.async_redis", new=MagicMock()
    )
    mock_async_redis.ConnectionPool = MagicMock()
    mock_async_redis.Redis = MagicMock(return_value=redis_instance)

    return {"sync": mock_redis, "async": mock_async_redis}


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
def basic_kafka_consumer():
    """
    Fixture that creates a basic KafkaConsumer for method-level testing. Used for tests
    that focus on individual methods rather than the full consume_messages integration.
    """
    kafka_settings = settings.kafka_config
    consumer = KafkaConsumer(
        kafka_settings.topic, kafka_settings.group_id, kafka_settings.brokers
    )

    yield consumer

    consumer.stop()
    KafkaConsumer.clear_instance(KafkaConsumer)


@pytest.fixture
def message_instance(sample_kafka_message):
    message_instance = MagicMock()
    message_instance.value.return_value = json.dumps(sample_kafka_message).encode()
    message_instance.error.return_value = None
    return message_instance


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _cleanup_kafka_consumer():
    """
    Helper function to clean up KafkaConsumer instances.
    """
    KafkaConsumer.clear_instance(KafkaConsumer)


def get_expected_kafka_config(kafka_settings):
    """
    Helper function to get the expected Kafka configuration.

    Parameters
    ----------
    kafka_settings: ``dict``
        The Kafka settings to use for building the expected config.

    Returns
    -------
    ``dict``
        The expected Kafka configuration.
    """
    return {
        "bootstrap.servers": kafka_settings.brokers,
        "group.id": kafka_settings.group_id,
        **DEFAULT_CONFIG,
    }


# =============================================================================
# INITIALIZATION AND CONFIGURATION TESTS
# =============================================================================


class TestKafkaConsumerInitialization:
    valid_config = DictConfig({"setting_type": "kafka_settings"})

    def validate_consumer(self, consumer, kafka_settings, mock_confluent_consumer):
        """
        Validate the KafkaConsumer instance against the expected Kafka settings.

        Parameters
        ----------
        consumer: ``KafkaConsumer``
            The KafkaConsumer instance to validate
        kafka_settings: ``dict``
            The expected Kafka settings for validation
        """
        assert isinstance(consumer, KafkaConsumer)
        assert consumer.topic == kafka_settings.topic
        assert consumer.group_id == kafka_settings.group_id
        assert consumer.brokers == kafka_settings.brokers
        assert consumer._should_run is False

        mock_confluent_consumer.assert_called_once_with(
            get_expected_kafka_config(kafka_settings)
        )
        consumer_instance = mock_confluent_consumer.return_value
        consumer_instance.subscribe.assert_called_once_with([kafka_settings.topic])

        _cleanup_kafka_consumer()

    def test_kafka_consumer_initialization(self, mock_confluent_consumer):
        """
        Initialize from the constructor.
        """
        kafka_settings = settings.kafka_config
        consumer = KafkaConsumer(
            kafka_settings.topic, kafka_settings.group_id, kafka_settings.brokers
        )
        self.validate_consumer(consumer, kafka_settings, mock_confluent_consumer)

    def test_kafka_consumer_from_cfg(self, mock_confluent_consumer):
        """
        Initialize from the from_cfg method.
        """
        # Test initialization from cfg method
        consumer = KafkaConsumer.from_cfg(self.valid_config)
        self.validate_consumer(consumer, settings.kafka_config, mock_confluent_consumer)

        with pytest.raises(ValueError) as exe:
            KafkaConsumer.from_cfg(DictConfig({"setting_type": None}))
        assert str(exe.value) == "setting_type is required"

    def test_kafka_init_from_cfg(self, mock_confluent_consumer):
        """
        Initialize from the init_from_cfg function.
        """
        consumer = init_from_cfg(self.valid_config, KafkaConsumer)
        self.validate_consumer(consumer, settings.kafka_config, mock_confluent_consumer)

    def test_kafka_consumer_build_init(self, mock_confluent_consumer):
        """
        Initialize from the build method.
        """
        consumer = KafkaConsumer.build(settings.kafka_config)
        self.validate_consumer(consumer, settings.kafka_config, mock_confluent_consumer)


# # =============================================================================
# # BASIC METHOD TESTS
# # =============================================================================
class TestKafkaConsumerCreateConsumer:
    def test_kafka_consumer_create_success(
        self, basic_kafka_consumer, mock_logger, mock_confluent_consumer
    ):
        """
        Test KafkaConsumer creation.
        """
        # Reset mocks as the _create_consumer method is called in the initialization
        mock_confluent_consumer.reset_mock()
        basic_kafka_consumer._create_consumer()

        mock_confluent_consumer.assert_called_once_with(
            get_expected_kafka_config(settings.kafka_config)
        )
        consumer_instance = mock_confluent_consumer.return_value
        consumer_instance.subscribe.assert_called_once_with(
            [settings.kafka_config.topic]
        )

        mock_logger.info.assert_called_once_with(
            "Subscribed to Kafka topic '%s' with config: %s",
            settings.kafka_config.topic,
            get_expected_kafka_config(settings.kafka_config),
        )

    def test_kafka_consumer_create_with_error(
        self, basic_kafka_consumer, mock_confluent_consumer, mock_logger
    ):
        """
        Test KafkaConsumer creation with runtime error.
        """
        mock_confluent_consumer.side_effect = RuntimeError("Kafka error")
        with pytest.raises(RuntimeError) as e:
            basic_kafka_consumer._create_consumer()
        assert str(e.value) == "Kafka error"

        mock_confluent_consumer.side_effect = KafkaException("Kafka exception")
        with pytest.raises(KafkaException) as e:
            basic_kafka_consumer._create_consumer()
        assert str(e.value) == "Kafka exception"

        mock_logger.info.assert_not_called()


class TestKafkaConsumerPollMessage:

    def validate_msg(self, msg, expected, basic_kafka_consumer):
        assert msg == expected
        basic_kafka_consumer.consumer.poll.assert_called_once_with(timeout=1.0)

    def test_poll_message_success(
        self, basic_kafka_consumer, sample_kafka_message, message_instance
    ):
        confluent_instance = basic_kafka_consumer.consumer
        confluent_instance.poll.return_value = message_instance

        msg = basic_kafka_consumer._poll_message()
        self.validate_msg(msg, json.dumps(sample_kafka_message), basic_kafka_consumer)

    def test_poll_message_none(self, basic_kafka_consumer):
        msg = basic_kafka_consumer._poll_message()
        self.validate_msg(msg, None, basic_kafka_consumer)

    def test_poll_message_with_error(self, basic_kafka_consumer, message_instance):
        message_instance.error.side_effect = KafkaException("Poll error")
        confluent_instance = basic_kafka_consumer.consumer
        confluent_instance.poll.return_value = message_instance

        with pytest.raises(KafkaException) as e:
            basic_kafka_consumer._poll_message()
        assert str(e.value) == "Poll error"


class TestKafkaConsumerTransformMessage:
    def test_transform_message(
        self, basic_kafka_consumer, sample_kafka_message, sample_transformed_message
    ):
        transformed = basic_kafka_consumer._transform_message(sample_kafka_message)
        assert transformed == sample_transformed_message


class TestKafkaConsumerProcessMessage:
    @pytest.mark.asyncio
    async def test_process_message_success(
        self, basic_kafka_consumer, mock_update_stock_cache, sample_transformed_message
    ):
        mock_redis_client = AsyncMock()
        await basic_kafka_consumer.process_message(
            sample_transformed_message, mock_redis_client
        )

        mock_redis_client.publish.assert_called_once_with(
            f"stock:{sample_transformed_message['symbol']}_NSE",
            json.dumps(sample_transformed_message),
        )
        mock_update_stock_cache.assert_called_once_with(
            f"stock:{sample_transformed_message['symbol']}_NSE",
            sample_transformed_message,
            mock_redis_client,
        )

    @pytest.mark.parametrize(
        "missing_keys", [["symbol"], ["exchange"], ["symbol", "exchange"]]
    )
    @pytest.mark.asyncio
    async def test_process_message_missing_keys(
        self,
        basic_kafka_consumer,
        sample_transformed_message,
        mock_logger,
        missing_keys,
    ):
        mock_redis_client = AsyncMock()
        incomplete_message: dict[str, Any] = sample_transformed_message.copy()

        for key in missing_keys:
            incomplete_message.pop(key, None)

        await basic_kafka_consumer.process_message(
            incomplete_message, mock_redis_client
        )
        mock_logger.warning.assert_called_once_with(
            "Skipping message: missing required field(s). Expected both 'symbol' and 'exchange', got data=%s",
            incomplete_message,
        )

    @pytest.mark.parametrize(
        "error, error_type",
        [
            (RedisConnectionError("Publish error"), RedisConnectionError),
            (CacheUpdateError("Cache update error"), CacheUpdateError),
        ],
    )
    @pytest.mark.asyncio
    async def test_process_message_errors(
        self,
        basic_kafka_consumer,
        sample_transformed_message,
        mock_logger,
        error,
        error_type,
    ):
        mock_redis_client = AsyncMock()
        mock_redis_client.publish.side_effect = error

        with pytest.raises(error_type) as e:
            await basic_kafka_consumer.process_message(
                sample_transformed_message, mock_redis_client
            )
        assert str(e.value) == str(error)
        mock_logger.exception.assert_called_once_with(
            "Failed to process message: %s, error: %s",
            sample_transformed_message,
            str(error),
        )


class TestKafkaConsumerConsumeMessage:

    def validate_confluent_init_and_cleanup(self, kafka_consumer):
        confluent_instance = kafka_consumer.consumer
        confluent_instance.close.assert_called_once()
        confluent_instance.subscribe.assert_called_once_with([kafka_consumer.topic])

    def validate_unused_mock(self, mock_redis, mock_stock_update):
        mock_redis["client"].publish.assert_not_called()
        mock_stock_update.assert_not_called()

    def validate_mock_calls(
        self, mock_redis, mock_stock_update, sample_transformed_message
    ):

        redis_client = mock_redis["client"]
        redis_client.publish.assert_called_once_with(
            f"stock:{sample_transformed_message['symbol']}_NSE",
            json.dumps(sample_transformed_message),
        )
        mock_stock_update.assert_called_once_with(
            f"stock:{sample_transformed_message['symbol']}_NSE",
            sample_transformed_message,
            redis_client,
        )

    def validate_basic_logs(
        self,
        mock_logger,
        kafka_consumer,
        raw_message,
        log_level: int,
        is_clean: bool = True,
    ):
        if log_level >= 2:
            mock_logger.debug.assert_any_call(
                "Raw message: %s", raw_message[:200] + "..."
            )

        if log_level >= 1:
            if is_clean:
                mock_logger.info.assert_any_call(
                    "Max messages reached, stopping consumer."
                )
            mock_logger.info.assert_any_call("Closing Kafka consumer and resources...")
            mock_logger.info.assert_any_call("Kafka consumer resources cleaned up")
            mock_logger.info.assert_any_call(
                "Starting Kafka consumer (topic=%s, group=%s)",
                kafka_consumer.topic,
                kafka_consumer.group_id,
            )

    def setup_message(self, kafka_consumer, message):
        confluent_instance = kafka_consumer.consumer
        confluent_instance.poll.return_value = message

    @pytest.fixture
    def validator(self, mock_redis_connection, mock_logger, mock_update_stock_cache):

        def combine_validation(
            basic_kafka_consumer,
            raw_message=None,
            log_level=1,
            sample_transformed_message=None,
            is_clean=False,
            is_unused=True,
        ):
            self.validate_confluent_init_and_cleanup(basic_kafka_consumer)
            self.validate_basic_logs(
                mock_logger, basic_kafka_consumer, raw_message, log_level, is_clean
            )
            if is_unused:
                self.validate_unused_mock(
                    mock_redis_connection, mock_update_stock_cache
                )
            else:
                self.validate_mock_calls(
                    mock_redis_connection,
                    mock_update_stock_cache,
                    sample_transformed_message,
                )

        return combine_validation

    @pytest.mark.asyncio
    async def test_consume_message_success(
        self,
        message_instance,
        basic_kafka_consumer,
        sample_transformed_message,
        sample_kafka_message,
        validator,
    ):
        self.setup_message(basic_kafka_consumer, message_instance)
        await basic_kafka_consumer.consume_messages(max_messages=1)

        raw_message = json.dumps(sample_kafka_message)
        validator(
            basic_kafka_consumer,
            raw_message=raw_message,
            log_level=2,
            sample_transformed_message=sample_transformed_message,
            is_clean=True,
            is_unused=False,
        )

    @pytest.mark.asyncio
    async def test_consume_message_no_messages(self, basic_kafka_consumer, validator):
        self.setup_message(basic_kafka_consumer, None)

        await basic_kafka_consumer.consume_messages(max_messages=3)

        validator(basic_kafka_consumer)

    @pytest.mark.asyncio
    async def test_consume_message_process_error(
        self, basic_kafka_consumer, validator, mock_logger, mock_redis_connection
    ):
        redis_connection = mock_redis_connection["connection"].build.return_value
        redis_connection_error = RedisConnectionError("Redis connection error")
        redis_connection.get_connection.side_effect = redis_connection_error

        self.setup_message(basic_kafka_consumer, None)

        with pytest.raises(RedisConnectionError) as e:
            await basic_kafka_consumer.consume_messages(max_messages=3)
        assert str(e.value) == "Redis connection error"

        validator(basic_kafka_consumer)
        mock_logger.exception.assert_any_call(
            "Fatal error in Kafka consumer loop: %s", redis_connection_error
        )

    @pytest.mark.asyncio
    async def test_consume_message_process_error_asyncio(
        self, basic_kafka_consumer, validator, mocker, mock_logger
    ):
        loop = asyncio.get_running_loop()
        mocker.patch.object(
            loop,
            "run_in_executor",
            side_effect=asyncio.CancelledError("Error while getting event loop"),
        )
        self.setup_message(basic_kafka_consumer, None)
        await basic_kafka_consumer.consume_messages(max_messages=3)

        validator(basic_kafka_consumer)
        mock_logger.info.assert_any_call(
            "Cancellation requested, stopping consumer loop."
        )

    @pytest.mark.asyncio
    async def test_consume_message_process_error_json_decoder(
        self, basic_kafka_consumer, validator, message_instance, mock_logger
    ):

        message_instance.value.return_value = "h".encode("utf-8")
        self.setup_message(basic_kafka_consumer, message_instance)
        await basic_kafka_consumer.consume_messages(max_messages=1)
        validator(basic_kafka_consumer)
        assert (
            "JSON decode error: %s | Raw message: %s..."
            == mock_logger.error.call_args_list[0][0][0]
        )

    @pytest.mark.asyncio
    async def test_consume_message_process_error_kafka(
        self, basic_kafka_consumer, validator, message_instance, mock_logger
    ):
        kafka_error = KafkaException("Kafka error")
        message_instance.error.side_effect = kafka_error
        self.setup_message(basic_kafka_consumer, message_instance)
        await basic_kafka_consumer.consume_messages(max_messages=1)

        validator(basic_kafka_consumer)

        mock_logger.error.assert_any_call("Kafka error: %s", kafka_error)

    @pytest.mark.asyncio
    async def test_consume_message_process_error_backoff(
        self,
        basic_kafka_consumer,
        message_instance,
        mock_logger,
        mocker,
    ):
        mocker.patch(
            "app.data_layer.streaming.consumers.kafka_consumer.MAX_BACKOFF_TIME", 1
        )
        kafka_error = KafkaException("Kafka error")
        message_instance.error.side_effect = kafka_error
        self.setup_message(basic_kafka_consumer, message_instance)
        await basic_kafka_consumer.consume_messages(max_messages=5)

        assert mock_logger.error.call_count == 5
        assert mock_logger.warning.call_count == 2
        mock_logger.warning.assert_any_call(
            "Multiple errors (%d). Backing off for %.2f seconds.", 5, 1
        )

    @pytest.mark.asyncio
    async def test_consume_message_process_error_backoff_critic(
        self,
        basic_kafka_consumer,
        message_instance,
        mock_logger,
        mocker,
    ):
        mocker.patch(
            "app.data_layer.streaming.consumers.kafka_consumer.MAX_BACKOFF_TIME", 1
        )
        kafka_error = KafkaException("Kafka error")
        message_instance.error.side_effect = kafka_error
        self.setup_message(basic_kafka_consumer, message_instance)
        await basic_kafka_consumer.consume_messages(max_messages=12)

        assert mock_logger.error.call_count == 12
        assert mock_logger.warning.call_count == 7
        mock_logger.warning.assert_any_call(
            "Multiple errors (%d). Backing off for %.2f seconds.", 7, 1
        )

        mock_logger.critical.assert_called_once_with(
            "Reached maximum retry attempts (%d). Backing off for %.2f seconds.", 10, 1
        )
        mock_logger.info.assert_any_call(
            "Subscribed to Kafka topic '%s' with config: %s",
            "instrument",
            get_expected_kafka_config(settings.kafka_config),
        )
        mock_logger.warning.assert_any_call(
            "Restarting Kafka consumer due to persistent errors"
        )

        assert basic_kafka_consumer.consumer.close.call_count == 2
        assert basic_kafka_consumer._should_run is False


class TestKafkaConsumerStop:

    def test_stop_consumer(self, basic_kafka_consumer, mock_logger):
        basic_kafka_consumer._should_run = True
        basic_kafka_consumer.stop()
        mock_logger.info.assert_any_call("Received stop signal for Kafka consumer")
        assert basic_kafka_consumer._should_run is False
