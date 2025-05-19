import asyncio
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, cast

from confluent_kafka import Consumer as ConfluentConsumer
from confluent_kafka import KafkaException
from omegaconf import DictConfig
from redis.asyncio import Redis

from app.cache.instrument_cache import update_stock_cache
from app.data_layer.streaming.consumer import Consumer
from app.utils.common.logger import get_logger
from app.utils.common.types.financial_types import DataProviderType, ExchangeType
from app.utils.constants import (
    CHANNEL_PREFIX,
    KAFKA_BROKER_URL,
    KAFKA_CONSUMER_GROUP_ID,
    KAFKA_TOPIC_INSTRUMENT,
)
from app.utils.fetch_data import get_env_var
from app.utils.redis_utils import RedisAsyncConnection

logger = get_logger(Path(__file__).name)

# Default configuration values
DEFAULT_CONFIG = {
    "auto.offset.reset": "earliest",
    "enable.auto.commit": True,
    "broker.address.family": "v4",
    "session.timeout.ms": 30000,  # 30 seconds
    "fetch.min.bytes": 1,
    "fetch.wait.max.ms": 500,
}

# Backoff parameters
MIN_BACKOFF_TIME = 1  # seconds
MAX_BACKOFF_TIME = 60  # seconds
MAX_BACKOFF_ATTEMPTS = 10


class KafkaConsumer(Consumer):
    """
    Kafka-based consumer implementation for market data streaming.

    Consumes messages from a Kafka topic, transforms the data into a standardized format,
    and publishes to Redis channels while also updating the cache. Implements robust
    error handling with exponential backoff for resilience against transient failures.

    Attributes:
    -----------
    topic: ``str``
        The Kafka topic to subscribe to for market data
    group_id: ``str``
        The consumer group ID for tracking consumption progress
    brokers: ``str``
        The comma-separated list of Kafka broker URLs
    config: ``Dict[str, Any]``
        Additional Kafka consumer configuration parameters
    """

    def __init__(
        self,
        topic: str | None = None,
        group_id: str | None = None,
        brokers: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize a Kafka consumer for market data.

        Creates a new Kafka consumer with the specified parameters or defaults
        from environment variables. Sets up a thread pool for handling blocking
        Kafka operations without blocking the asyncio event loop.

        Parameters
        ----------
        topic: ``str | None``, ( default = None)
            The Kafka topic to subscribe to. If None, uses the KAFKA_TOPIC_INSTRUMENT
            environment variable.
        group_id: ``str | None``, ( default = None))
            The consumer group ID. If None, uses the KAFKA_CONSUMER_GROUP_ID
            environment variable.
        brokers: ``str | None``, ( default = None))
            The Kafka broker URL(s). If None, uses the KAFKA_BROKER_URL
            environment variable.
        config: ``dict[str, Any] | None``, ( default = None))
            Additional Kafka consumer configuration. If provided, these settings
            will override the default configuration values.
        """
        self.topic = topic or get_env_var(KAFKA_TOPIC_INSTRUMENT)
        self.group_id = group_id or get_env_var(KAFKA_CONSUMER_GROUP_ID)
        self.brokers = brokers or get_env_var(KAFKA_BROKER_URL)
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        # Create thread pool for blocking Kafka operations
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="kafka_consumer"
        )
        self.consumer = self._create_consumer()
        self._should_run = False

    def _create_consumer(self) -> ConfluentConsumer:
        """
        Create a configured Confluent Kafka consumer instance.
        Sets up a Kafka consumer with the appropriate configuration and
        subscribes it to the specified topic.

        Returns
        -------
        consumer: ``ConfluentConsumer``
            A configured and subscribed Confluent Kafka consumer instance
        """
        config: Dict[str, Any] = {
            "bootstrap.servers": self.brokers,
            "group.id": self.group_id,
            **self.config,
        }
        consumer = ConfluentConsumer(config)
        consumer.subscribe([self.topic])
        logger.info(
            "Subscribed to Kafka topic '%s' with config: %s",
            self.topic,
            {k: v for k, v in config.items() if k != "bootstrap.servers"},
        )

        return consumer

    def _poll_message(self) -> str | None:
        """
        Poll Kafka for a single message in a blocking manner.

        This method is meant to be executed within a ThreadPoolExecutor
        to avoid blocking the main asyncio event loop, as Confluent Kafka's
        consumer.poll() is a synchronous/blocking operation.

        Returns
        -------
        message: ``str | None``
            The message value as a UTF-8 decoded string if a message was received,
            or None if no message was available within the poll timeout (1 second)

        Raises
        ------
        ``KafkaException``
            If the polled message contains an error or there's a Kafka-related issue
        """
        msg = self.consumer.poll(timeout=1.0)
        if msg is None:
            return None

        if msg.error():
            raise KafkaException(msg.error())

        return msg.value().decode("utf-8")

    async def process_message(self, data: dict[str, Any], redis_client: Redis) -> None:
        """
        Process a market data message from Kafka and publish to Redis.

        Takes transformed market data, publishes it to the appropriate Redis
        channel for real-time updates, and updates the cache with the latest data.

        Parameters
        ----------
        data: ``dict[str, Any]``
            The transformed market data message containing symbol, price, and other
            trading information
        redis_client: ``Redis``
            The Redis client instance to use for publishing and cache updates

        Raises
        ------
        ``Exception``
            Any errors during Redis publishing or cache updating will propagate
            and should be handled by the caller
        """
        try:
            symbol = data.get("symbol")
            exchange = data.get("exchange")

            if not symbol or not exchange:
                logger.warning(
                    "Message missing required fields (symbol or exchange): %s", data
                )
                return

            channel = f"{CHANNEL_PREFIX}{symbol}_{exchange}"

            # Publish to Redis channel
            await redis_client.publish(channel, json.dumps(data))

            # Update cache with latest data
            await update_stock_cache(channel, data, redis_client)

        except Exception as e:
            logger.exception("Failed to process message: %s, error: %s", data, str(e))
            raise

    async def consume_messages(self) -> None:
        """
        Continuously consume and process messages from Kafka.

        Main consumer loop that polls Kafka for messages, transforms them into
        the proper format, and processes them by publishing to Redis and updating
        the cache. Implements robust error handling with exponential backoff for
        transient failures and automatic consumer restart for persistent issues.

        Raises
        ------
        ``Exception``
            Fatal errors that cannot be handled within the retry mechanism
            will be propagated after resource cleanup
        """
        logger.info(
            "Starting Kafka consumer (topic=%s, group=%s)",
            self.topic,
            self.group_id,
        )

        self._should_run = True
        redis_async_connection = RedisAsyncConnection()
        redis_client = await redis_async_connection.get_connection()

        consecutive_errors = 0
        last_success_time = time.time()

        try:
            loop = asyncio.get_running_loop()
            while self._should_run:
                try:
                    raw = await loop.run_in_executor(self._executor, self._poll_message)
                    if not raw:
                        await asyncio.sleep(0.01)
                        continue

                    logger.debug(
                        "Raw message: %s", raw[:200] + "..." if len(raw) > 200 else raw
                    )
                    payload = json.loads(raw)

                    msg = self._transform_message(payload)
                    await self.process_message(msg, redis_client)

                    consecutive_errors = 0
                    last_success_time = time.time()

                except asyncio.CancelledError:
                    logger.info("Cancellation requested, stopping consumer loop.")
                    break

                except (json.JSONDecodeError, KafkaException, Exception) as e:
                    self._handle_consume_error(e, raw if "raw" in locals() else None)
                    consecutive_errors += 1

                if consecutive_errors > 0:
                    await self._apply_backoff(consecutive_errors)

                    if (
                        consecutive_errors >= MAX_BACKOFF_ATTEMPTS
                        or time.time() - last_success_time > 300
                    ):
                        logger.warning(
                            "Restarting Kafka consumer due to persistent errors"
                        )
                        self.consumer.close()
                        self.consumer = self._create_consumer()
                        consecutive_errors = 0

        except Exception as e:
            logger.exception("Fatal error in Kafka consumer loop: %s", e)
            raise

        finally:
            logger.info("Closing Kafka consumer and resources...")
            self._should_run = False
            self.consumer.close()
            await redis_async_connection.close_connection()
            self._executor.shutdown(wait=True)
            logger.info("Kafka consumer resources cleaned up")

    def _transform_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Transform the raw payload into the required message format.
        """
        return {
            "retrieval_timestamp": payload["retrieval_timestamp"],
            "last_traded_timestamp": payload["last_traded_timestamp"],
            "symbol": payload["symbol"],
            "exchange": cast(
                ExchangeType,
                ExchangeType.get_exchange(payload["exchange_id"]),
            ).name,
            "data_provider": cast(
                DataProviderType,
                DataProviderType.get_data_provider(payload["data_provider_id"]),
            ).name,
            "last_traded_price": payload["last_traded_price"],
            "last_traded_quantity": payload.get("last_traded_quantity"),
            "average_traded_price": payload.get("average_traded_price"),
            "volume_trade_for_the_day": payload.get("volume_trade_for_the_day"),
            "total_buy_quantity": payload.get("total_buy_quantity"),
            "total_sell_quantity": payload.get("total_sell_quantity"),
        }

    def _handle_consume_error(self, error: Exception, raw: str | None) -> None:
        """
        Handle errors during message consumption and log appropriately.
        """
        if isinstance(error, json.JSONDecodeError):
            logger.error(
                "JSON decode error: %s | Raw message: %s...",
                error,
                raw[:100] if raw else "None",
            )
        elif isinstance(error, KafkaException):
            logger.error("Kafka error: %s", error)
        else:
            logger.exception("Unexpected error processing message: %s", error)

    async def _apply_backoff(self, consecutive_errors: int) -> None:
        """
        Apply exponential backoff with jitter based on consecutive errors.
        """
        backoff_time = min(
            MIN_BACKOFF_TIME * (2 ** (consecutive_errors - 1)) + random.uniform(0, 1),
            MAX_BACKOFF_TIME,
        )

        if consecutive_errors >= MAX_BACKOFF_ATTEMPTS:
            logger.critical(
                "Reached maximum retry attempts (%d). Backing off for %.2f seconds.",
                MAX_BACKOFF_ATTEMPTS,
                backoff_time,
            )
        elif consecutive_errors > 3:
            logger.warning(
                "Multiple errors (%d). Backing off for %.2f seconds.",
                consecutive_errors,
                backoff_time,
            )
        else:
            logger.info(
                "Error encountered. Backing off for %.2f seconds.",
                backoff_time,
            )

        await asyncio.sleep(backoff_time)

    def stop(self):
        """
        Signal the consumer to gracefully stop processing messages.

        Sets the internal flag to indicate that the consumer should stop
        running after completing the current message processing.
        """
        logger.info("Received stop signal for Kafka consumer")
        self._should_run = False

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Optional["KafkaConsumer"]:
        """
        Create a KafkaConsumer instance from a configuration object.

        Factory method to instantiate and configure a Kafka consumer
        using parameters defined in a configuration object.

        Parameters
        ----------
        cfg: ``DictConfig``
            Configuration object containing Kafka settings including:
            - topic: The Kafka topic to subscribe to
            - group_id: The consumer group ID
            - brokers: The Kafka broker URL(s)
            - consumer_config: Additional configuration parameters

        Returns
        -------
        consumer: ``KafkaConsumer | None``
            A configured KafkaConsumer instance if the configuration is valid,
            or None if the configuration is invalid or an error occurs
        """
        try:
            config_dict = {}

            if "consumer_config" in cfg:
                consumer_config = cfg.get("consumer_config", {})
                if hasattr(consumer_config, "to_dict"):
                    config_dict = consumer_config.to_dict()
                else:
                    config_dict = dict(consumer_config)

            return cls(
                topic=cfg.get("topic"),
                group_id=cfg.get("group_id"),
                brokers=cfg.get("brokers"),
                config=config_dict,
            )
        except Exception as e:
            logger.error("Failed to create KafkaConsumer from config: %s", e)
            return None
