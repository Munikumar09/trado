import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
from omegaconf import DictConfig

from app.data_layer.data_saver.data_saver import DataSaver
from app.utils.common.logger import get_logger

logger = get_logger(Path(__file__).name)


@DataSaver.register("csv_saver")
class CSVDataSaver(DataSaver):
    """
    This CSVDataSaver retrieve the data from kafka consumer and save it
    to a csv file.

    Attributes
    ----------
    consumer: ``KafkaConsumer``
        Kafka consumer object to consume the data from the specified topic
    csv_file_path: ``str | Path``
        Path to save the csv file. The file name will be the given name
        appended with the current date.
        For example: `csv_file_path` = "data.csv", then the file name will
        be `data_2021_09_01.csv`
    """

    def __init__(self, consumer: KafkaConsumer, csv_file_path: str | Path) -> None:
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
        try:
            idx = 0
            with open(self.csv_file_path, "a", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)

                for idx, message in enumerate(self.consumer):
                    decoded_data = message.value.decode("utf-8")
                    decoded_data = json.loads(decoded_data)

                    if idx == 0:
                        writer.writerow(list(decoded_data.keys()))
                    writer.writerow(list(decoded_data.values()))

                    # Save the data as soon as it is received
                    file.flush()
        except Exception as e:
            logger.error("Error while saving data to csv: %s", e)
        finally:
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
        try:
            return cls(
                KafkaConsumer(
                    cfg.streaming.kafka_topic,
                    bootstrap_servers=cfg.streaming.kafka_server,
                    auto_offset_reset="earliest",
                ),
                cfg.get("csv_file_path"),
            )
        except NoBrokersAvailable:
            logger.error(
                "No Broker is available at the address: %s. No data will be saved.",
                cfg.streaming.kafka_server,
            )
            return None
