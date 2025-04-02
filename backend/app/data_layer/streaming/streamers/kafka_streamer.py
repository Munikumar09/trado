from pathlib import Path
from typing import Optional

from kafka import KafkaProducer
from omegaconf import DictConfig

from app.data_layer.streaming.streamer import Streamer
from app.utils.common.logger import get_logger

logger = get_logger(Path(__file__).name)


@Streamer.register("kafka")
class KafkaStreamer(Streamer):
    """
    Kafka streaming class to send data to Kafka server. This can be used as a callback
    function to send data to Kafka.

    Attributes:
    -----------
    kafka_server: ``str``
        The ip address and port of the Kafka server in the format "ip_address:port"
    kafka_topic: ``str``
        The topic to which the data should be sent to in Kafka
    """

    def __init__(self, kafka_server: str, kafka_topic: str):
        self.kafka_topic = kafka_topic
        self.kafka_producer = KafkaProducer(bootstrap_servers=kafka_server)

    def __call__(self, data: str):
        """
        This function sends the received data to the Kafka server. It receives the data
        as a string and encodes it to utf-8 before sending it to the Kafka server.

        Parameters:
        -----------
        data: ``str``
            The data to be sent to the Kafka server as a string
        """
        try:
            bytes_data = data.encode("utf-8")
            self.kafka_producer.send(self.kafka_topic, bytes_data)
            self.kafka_producer.flush()
        except Exception as e:
            logger.error("Error sending data to Kafka: %s", e)

    def close(self):
        """
        Close the Kafka producer connection.
        """
        if self.kafka_producer:
            try:
                self.kafka_producer.close()
            except Exception as e:
                logger.error("Error closing Kafka producer: %s", e)

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Optional["KafkaStreamer"]:
        try:
            return cls(cfg["kafka_server"], cfg["kafka_topic"])
        except Exception as e:
            logger.error("Error creating KafkaStreaming object: %s", e)
            return None
