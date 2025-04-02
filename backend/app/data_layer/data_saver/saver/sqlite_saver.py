import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
from omegaconf import DictConfig
from sqlalchemy import create_engine

from app.data_layer.data_saver.data_saver import DataSaver
from app.data_layer.database.crud.crud_utils import insert_data
from app.data_layer.database.db_connections.sqlite import (
    create_db_and_tables,
    get_session,
)
from app.data_layer.database.models import InstrumentPrice
from app.utils.common.logger import get_logger

logger = get_logger(Path(__file__).name)


@DataSaver.register("sqlite_saver")
class SqliteDataSaver(DataSaver):
    """
    This SqliteDataSaver retrieve the data from kafka consumer and save it
    to sqilte database.

    Attributes
    ----------
    consumer: ``KafkaConsumer``
        Kafka consumer object to consume the data from the specified topic.
    sqlite_db: ``str``
        Sqlite database path to save the data. The database will be created
        in the specified path. The name of the database will be the given name
        appended with the current date.
        For example: `sqlite_db` = "data.sqlite3", then the database name will
        be `data_2021_09_01.sqlite3`
    """

    def __init__(self, consumer: KafkaConsumer, sqlite_db: str | Path) -> None:
        self.consumer = consumer

        if isinstance(sqlite_db, str):
            sqlite_db = Path(sqlite_db)

        if sqlite_db.suffix in [".sqlite3", ".db", ".sqlite"]:
            sqlite_db = sqlite_db.with_suffix("")

        if not sqlite_db.parent.exists():
            sqlite_db.parent.mkdir(parents=True, exist_ok=True)

        current_date = datetime.now().strftime("%Y_%m_%d")
        self.sqlite_db = f"sqlite:///{sqlite_db}_{current_date}.sqlite3"

        self.engine = create_engine(self.sqlite_db)
        create_db_and_tables(self.engine)

    def save_stock_data(self, data: dict[str, str | None]) -> None:
        """
        Create a InstrumentPrice object from the given data and save it
        to the sqlite database.

        Parameters
        ----------
        data: ``dict[str, str | None]``
            The data to be saved in the database. The data should contain all
            the required fields to create a InstrumentPrice object. While
            saving the data, if the data is already present in the database, it
            will be ignored. Means, the data will be saved only if it is not
            present in the database. The presence of the data is checked based
            on the primary key of the InstrumentPrice object.
        """
        instrument_price = InstrumentPrice(
            retrieval_timestamp=data["retrieval_timestamp"],
            last_traded_timestamp=data["last_traded_timestamp"],
            symbol=data["symbol"],
            exchange_id=data["exchange_id"],
            data_provider_id=data["data_provider_id"],
            last_traded_price=data["last_traded_price"],
            last_traded_quantity=data.get("last_traded_quantity"),
            average_traded_price=data.get("average_traded_price"),
            volume_trade_for_the_day=data.get("volume_trade_for_the_day"),
            total_buy_quantity=data.get("total_buy_quantity"),
            total_sell_quantity=data.get("total_sell_quantity"),
        )
        with get_session(self.engine) as session:
            insert_data(InstrumentPrice, instrument_price, session=session)

    def save(self, data: bytes) -> None:
        """
        Decode the given data and save it to the sqlite database.

        Parameters
        ----------
        data: ``bytes``
            The data to be saved in the database. The data should be in bytes
            format and should be decoded to utf-8 format. The decoded data
            should be in json format.
        """
        decoded_data = data.decode("utf-8")
        data_to_insert = json.loads(decoded_data)
        self.save_stock_data(data_to_insert)

    def retrieve_and_save(self):
        """
        Retrieve the data from the kafka consumer and save it to the sqlite
        database.
        """
        for message in self.consumer:
            self.save(message.value)

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Optional["SqliteDataSaver"]:
        try:
            return cls(
                KafkaConsumer(
                    cfg.streaming.kafka_topic,
                    bootstrap_servers=cfg.streaming.kafka_server,
                    auto_offset_reset="earliest",
                ),
                cfg.get("sqlite_db"),
            )
        except NoBrokersAvailable:
            logger.error(
                "No Broker is availble at the address: %s. No data will be saved.",
                cfg.streaming.kafka_server,
            )
            return None
