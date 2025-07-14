import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from confluent_kafka import Consumer
from confluent_kafka.error import KafkaException
from omegaconf import DictConfig

from app.data_layer.data_saver.data_saver import DataSaver
from app.utils.common.logger import get_logger
from app.utils.constants import (
    KAFKA_CONSUMER_DEFAULT_CONFIG,
    KAFKA_CONSUMER_GROUP_ID_ENV,
)
from app.utils.kafka_utils import get_kafka_consumer

logger = get_logger(Path(__file__).name)


@DataSaver.register("jsonl_saver")
class JSONLDataSaver(DataSaver):
    """
    JSONLDataSaver retrieve the data from kafka consumer and save it
    to a jsonl file.

    Attributes
    ----------
    consumer: ``KafkaConsumer``
        Kafka consumer object to consume the data from the specified topic
    jsonl_file_path: ``str | Path``
        Path to save the jsonl file. The file name will be the given name
        appended with the current date.
        For example: `jsonl_file_path` = "data.jsonl", then the file name will
        be `data_2021_09_01.jsonl`
    """

    def __init__(self, consumer: Consumer, jsonl_file_path: str | Path) -> None:
        self.consumer = consumer

        if isinstance(jsonl_file_path, str):
            jsonl_file_path = Path(jsonl_file_path)

        if not jsonl_file_path.parent.exists():
            jsonl_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_name = (
            jsonl_file_path.stem + f"_{datetime.now().strftime('%Y_%m_%d')}.jsonl"
        )
        self.jsonl_file_path = jsonl_file_path.with_name(file_name)

    def retrieve_and_save(self):
        """
        Retrieve the data from the kafka consumer and save it to the jsonl file.
        """
        try:
            idx = 0
            with open(self.jsonl_file_path, "a+", encoding="utf-8", newline="") as file:
                while True:
                    data = self.consumer.poll(timeout=1.0)
                    if not data:
                        continue

                    if data.error():
                        raise KafkaException(data.error())

                    message = data.value().decode("utf-8")
                    decoded_data = json.loads(message)
                    file.write(json.dumps(decoded_data, ensure_ascii=False) + "\n")
                    file.flush()
                    idx += 1
        except Exception as e:
            logger.error("Error while saving data to jsonl: %s", e)
        finally:
            self.consumer.close()
            logger.info("%s messages saved to jsonl", idx)

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Optional["JSONLDataSaver"]:
        """
        Create an instance of the JSONLDataSaver class from the given configuration.
        """
        try:
            consumer = get_kafka_consumer(
                {
                    "bootstrap.servers": cfg.streaming.kafka_server,
                    "group.id": KAFKA_CONSUMER_GROUP_ID_ENV,
                    **KAFKA_CONSUMER_DEFAULT_CONFIG,
                },
                cfg.streaming.kafka_topic,
            )

            if not cfg.get("jsonl_file_path"):
                logger.error(
                    "No jsonl_file_path provided in the configuration. No data will be saved."
                )
                return None

            return cls(
                consumer,
                cfg.get("jsonl_file_path"),
            )
        except KafkaException as e:
            logger.error("Error while creating JSONLDataSaver: %s", e)
            return None
