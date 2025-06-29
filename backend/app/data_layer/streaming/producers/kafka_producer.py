from pathlib import Path
from typing import Any, Dict, Optional

from confluent_kafka import KafkaError, Message
from confluent_kafka import Producer as ConfluentProducer
from omegaconf import DictConfig, OmegaConf

from app.data_layer.streaming.producer import Producer
from app.utils.common.logger import get_logger

logger = get_logger(Path(__file__).name)

# Default configuration for Kafka producer
DEFAULT_CONFIG = {
    "compression.type": "lz4",
    "queue.buffering.max.messages": 100000,
    "broker.address.family": "v4",  # Force IPv4
    "linger.ms": 5,  # allow short delay for batching
    "acks": "1",
    "request.timeout.ms": 5000,
    "message.timeout.ms": 60000,
}


@Producer.register("kafka_producer")
class KafkaProducer(Producer):
    """
    Kafka producer implementation for sending data to Kafka topics.

    Provides functionality to reliably send messages to Kafka topics with
    configurable delivery guarantees, compression, and batching options.
    Tracks delivery statistics and implements automatic queue management.

    Attributes:
    -----------
    kafka_topic: ``str``
        The Kafka topic to publish messages to
    kafka_producer: ``ConfluentProducer``
        The underlying Confluent Kafka producer instance
    config: ``Dict[str, Any]``
        Configuration settings for the Kafka producer
    """

    def __init__(
        self,
        kafka_server: str,
        kafka_topic: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a KafkaProducer instance with the specified Kafka server, topic, and optional configuration.
        
        Parameters:
            kafka_server (str): Address of the Kafka broker(s) to connect to.
            kafka_topic (str): Name of the Kafka topic to publish messages to.
            config (Optional[Dict[str, Any]]): Optional dictionary of additional Kafka producer configuration settings.
        """
        self.kafka_topic = kafka_topic

        # Merge default config with user-provided config
        producer_config = {**DEFAULT_CONFIG, **(config or {})}
        producer_config["bootstrap.servers"] = kafka_server

        logger.info(
            "Initializing Kafka producer for topic '%s' with config: %s",
            kafka_topic,
            producer_config,
        )

        self.kafka_producer = ConfluentProducer(producer_config)
        self._delivery_success_count = 0
        self._delivery_failure_count = 0

    def delivery_report(self, err: KafkaError | None, msg: Message) -> None:
        """
        Handles Kafka message delivery reports, updating internal success and failure counters and logging delivery outcomes.
        
        Called by the Kafka producer when a message is delivered or fails delivery. Periodically logs aggregate delivery statistics.
        """
        if err is not None:
            self._delivery_failure_count += 1
            logger.error(
                "Message delivery failed: %s for message to %s [%s]",
                err,
                msg.topic(),
                msg.partition(),
            )
        else:
            self._delivery_success_count += 1
            logger.debug("Message delivered to %s [%s]", msg.topic(), msg.partition())

        # Log statistics periodically
        total = self._delivery_success_count + self._delivery_failure_count
        if total % 1000 == 0 and total > 0:
            success_rate = (self._delivery_success_count / total) * 100
            logger.info(
                "Kafka producer statistics: %d successful / %d total messages (success rate: %.2f%%)",
                self._delivery_success_count,
                total,
                success_rate,
            )

    def __call__(self, data: str) -> bool:
        """
        Attempts to send a UTF-8 encoded string message to the configured Kafka topic.
        
        If the producer's internal queue is full, attempts to flush pending messages before retrying. Returns True if the message was successfully queued for delivery, or False if the input is empty, the queue remains full, or an error occurs.
        
        Parameters:
            data (str): The message to send, typically a JSON-encoded string.
        
        Returns:
            bool: True if the message was queued for delivery, False otherwise.
        """
        if not data:
            logger.warning("Attempted to send empty data to Kafka")
            return False

        try:
            self.kafka_producer.produce(
                topic=self.kafka_topic,
                value=data.encode("utf-8"),
                callback=self.delivery_report,
            )
            # Triggers callbacks for previously sent messages
            messages_remaining = self.kafka_producer.poll(0)

            # If queue is starting to fill up, flush some messages
            if messages_remaining > 10000:
                logger.info(
                    "Large producer queue (%d messages), flushing...",
                    messages_remaining,
                )
                flushed = self.kafka_producer.flush(timeout=1.0)
                logger.info("Flushed producer queue, %d messages remaining", flushed)

            return True

        except BufferError as e:
            logger.warning(
                "Local producer queue is full (%d messages awaiting delivery): %s",
                len(self.kafka_producer),
                e,
            )
            # Try to flush on buffer error
            logger.info("Attempting to flush producer queue...")
            remaining = self.kafka_producer.flush(timeout=5.0)
            logger.info(
                "Producer queue flush completed, %d messages remaining", remaining
            )
            return False

        except Exception as e:
            logger.error("Error sending data to Kafka: %s", e)
            return False

    def close(self) -> None:
        """
        Flushes the Kafka producer queue to ensure all pending messages are delivered before shutdown.
        
        Logs any undelivered messages and final delivery statistics. Intended to be called during application shutdown for clean resource release.
        """
        if self.kafka_producer:
            try:
                remaining = self.kafka_producer.flush(timeout=10.0)
                if remaining > 0:
                    logger.warning(
                        "Failed to flush all messages, %d messages remain in queue",
                        remaining,
                    )
                else:
                    logger.info("Kafka producer flushed successfully")

                # Log final statistics
                total = self._delivery_success_count + self._delivery_failure_count
                if total > 0:
                    success_rate = (self._delivery_success_count / total) * 100
                    logger.info(
                        "Final Kafka producer statistics: %d successful / %d total messages (success rate: %.2f%%)",
                        self._delivery_success_count,
                        total,
                        success_rate,
                    )
            except Exception as e:
                logger.error("Error flushing Kafka producer: %s", e)

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Optional["KafkaProducer"]:
        """
        Instantiate a KafkaProducer from an OmegaConf configuration object.
        
        Creates and returns a KafkaProducer instance using configuration values for Kafka server, topic, and optional producer settings. Returns None if required configuration fields are missing or invalid.
        
        Parameters:
            cfg (DictConfig): Configuration object containing 'kafka_server', 'kafka_topic', and optionally 'producer_config'.
        
        Returns:
            Optional[KafkaProducer]: Configured KafkaProducer instance, or None if configuration is invalid.
        """
        try:
            # Convert OmegaConf to dict
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)

            if not isinstance(cfg_dict, dict):
                logger.error("Invalid configuration format")
                return None

            kafka_server = cfg_dict.get("kafka_server")
            kafka_topic = cfg_dict.get("kafka_topic")

            if not kafka_server:
                logger.error("Missing kafka_server in configuration")
                return None

            if not kafka_topic:
                logger.error("Missing kafka_topic in configuration")
                return None

            # Extract producer-specific configuration if provided
            producer_config = cfg_dict.get("producer_config", {})

            return cls(kafka_server, kafka_topic, producer_config)

        except Exception as e:
            logger.error("Error creating KafkaProducer: %s", e)
            return None
