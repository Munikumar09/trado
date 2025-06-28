import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from confluent_kafka import Consumer
from confluent_kafka.error import KafkaException
from omegaconf import DictConfig

from app.data_layer.data_saver.data_saver import DataSaver
from app.utils.common.logger import get_logger
from app.utils.constants import KAFKA_CONSUMER_DEFAULT_CONFIG, KAFKA_CONSUMER_GROUP_ID
from app.utils.kafka_utils import get_kafka_consumer

logger = get_logger(Path(__file__).name)


@DataSaver.register("csv_saver")
class CSVDataSaver(DataSaver):
    """
    This CSVDataSaver retrieve the data from kafka consumer and save it
    to a csv file.

    Attributes
    ----------
    consumer: ``Consumer``
        Kafka consumer object to consume the data from the specified topic
    csv_file_path: ``str | Path``
        Path to save the csv file. The file name will be the given name
        appended with the current date.
        For example: `csv_file_path` = "data.csv", then the file name will
        be `data_2021_09_01.csv`
    """

    def __init__(self, consumer: Consumer, csv_file_path: str | Path) -> None:
        self.consumer = consumer

        if isinstance(csv_file_path, str):
            csv_file_path = Path(csv_file_path)

        if not csv_file_path.parent.exists():
            csv_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_name = csv_file_path.stem + f"_{datetime.now().strftime('%Y_%m_%d')}.csv"
        self.csv_file_path = csv_file_path.with_name(file_name)

    def retrieve_and_save(self):
        """
        Retrieve the data from the kafka consumer and save it to the csv file.
        """
        idx = 0
        try:
            with open(self.csv_file_path, "a", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                while True:
                    data = self.consumer.poll(timeout=1.0)
                    if not data:
                        continue

                    if data.error():
                        raise KafkaException(data.error())

                    message = data.value().decode("utf-8")
                    decoded_data = json.loads(message)

                    if idx == 0:
                        writer.writerow(list(decoded_data.keys()))
                    writer.writerow(list(decoded_data.values()))
                    idx += 1

                    # Save the data as soon as it is received
                    file.flush()
        except Exception as e:
            logger.error("Error while saving data to csv: %s", e)
        finally:
            self.consumer.close()
            logger.info("%s messages saved to csv", idx)

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Optional["CSVDataSaver"]:
        """
        Initialize the CSVDataSaver object from the given configuration.

        Parameters
        ----------
        cfg: ``DictConfig``
            Configuration object containing the necessary information to
            initialize the CSVDataSaver object
        """
        consumer = get_kafka_consumer(
            {
                "bootstrap.servers": cfg.streaming.kafka_server,
                "group.id": KAFKA_CONSUMER_GROUP_ID,
                **KAFKA_CONSUMER_DEFAULT_CONFIG,
            },
            cfg.streaming.kafka_topic,
        )

        if not consumer:
            return None

        csv_file_path = cfg.get("csv_file_path")
        if not csv_file_path:
            logger.error("csv_file_path not provided in configuration")
            return None

        return cls(
            consumer,
            csv_file_path,
        )
