from datetime import datetime
from pathlib import Path
from typing import Optional

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
from omegaconf import DictConfig

from app.data_layer.data_saver.data_saver import DataSaver
from app.utils.common.logger import get_logger

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

    def __init__(self, consumer: KafkaConsumer, jsonl_file_path: str | Path) -> None:
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
        idx = 0
        try:
            with open(self.jsonl_file_path, "a", encoding="utf-8", newline="") as file:
                for idx, message in enumerate(self.consumer):
                    decoded_data = message.value.decode("utf-8")
                    file.write(decoded_data + "\n")
                    file.flush()
        except Exception as e:
            logger.error("Error while saving data to jsonl: %s", e)
        finally:
            logger.info("%s messages saved to jsonl", idx)

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Optional["JSONLDataSaver"]:
        """
        Create an instance of the JSONLDataSaver class from the given configuration.
        """
        try:
            return cls(
                KafkaConsumer(
                    cfg.streaming.kafka_topic,
                    bootstrap_servers=cfg.streaming.kafka_server,
                    auto_offset_reset="earliest",
                ),
                cfg.get("jsonl_file_path"),
            )
        except NoBrokersAvailable:
            logger.error(
                "No Broker is available at the address: %s. No data will be saved.",
                cfg.streaming.kafka_server,
            )
            return None
