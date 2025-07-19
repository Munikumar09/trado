from typing import Any

from confluent_kafka import Consumer
from confluent_kafka.error import KafkaException


def get_kafka_consumer(
    config: dict[str, Any],
    topic: str | None = None,
) -> Consumer:
    """
    Create and return a Kafka consumer with the provided configuration.

    Parameters
    ----------
    config: ``dict[str, Any]``
        Configuration settings for the Kafka consumer
    topic: ``str | None``
        The Kafka topic to subscribe to

    Returns
    -------
    consumer: ``Consumer``
        A configured Kafka consumer instance
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
