from typing import Any

from confluent_kafka import Consumer
from confluent_kafka.error import KafkaException


def get_kafka_consumer(
    config: dict[str, Any],
    topic: str | None = None,
) -> Consumer:
    """
    Creates and returns a Kafka consumer instance using the provided configuration.
    
    If a topic is specified, the consumer is subscribed to that topic. Raises a `KafkaException` if consumer creation or subscription fails.
    
    Parameters:
        config (dict[str, Any]): Configuration settings for the Kafka consumer.
        topic (str | None): Optional Kafka topic to subscribe to.
    
    Returns:
        Consumer: Configured Kafka consumer instance.
    
    Raises:
        KafkaException: If consumer creation or subscription fails.
    """
    try:
        consumer = Consumer(config)

        if topic:
            consumer.subscribe([topic])

        return consumer
    except KafkaException as e:
        raise KafkaException(f"Failed to create Kafka consumer: {e}") from e
    except Exception as e:
        raise KafkaException(
            f"An unexpected error occurred while creating Kafka consumer: {e}"
        ) from e
