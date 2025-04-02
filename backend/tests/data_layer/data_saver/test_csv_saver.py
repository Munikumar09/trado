import json
from collections import namedtuple
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import cast

import pandas as pd
import pytest
from kafka.errors import NoBrokersAvailable
from omegaconf import DictConfig, OmegaConf
from pytest_mock import MockerFixture, MockType

from app.data_layer.data_saver import CSVDataSaver, DataSaver
from app.utils.common import init_from_cfg

Message = namedtuple("Message", ["value"])


####################################### FIXTURES #######################################
@pytest.fixture
def csv_config() -> DictConfig:
    """
    Configuration for the CSVDataSaver.

    Returns:
    --------
    ``DictConfig``
        Configuration for the CSVDataSaver
    """
    return OmegaConf.create(
        {
            "name": "csv_saver",
            "csv_file_path": TemporaryDirectory().name + "/test.csv",
            "streaming": {
                "kafka_topic": "test_topic",
                "kafka_server": "localhost:9092",
            },
        }
    )


@pytest.fixture
def mock_consumer(mocker: MockerFixture) -> MockType:
    """
    Mock the KafkaConsumer object.
    """
    return mocker.patch("app.data_layer.data_saver.csv_saver.KafkaConsumer")


@pytest.fixture
def mock_logger(mocker: MockerFixture) -> MockType:
    """
    Mock the logger object in the CSVDataSaver.
    """
    return mocker.patch("app.data_layer.data_saver.csv_saver.logger")


@pytest.fixture
def csv_saver(
    mock_consumer: MockType, csv_config: DictConfig, mocker: MockerFixture
) -> CSVDataSaver:
    """
    Fixture to return the CSVDataSaver object.
    """
    mock_consumer.return_value = mocker.MagicMock()

    return cast(CSVDataSaver, CSVDataSaver.from_cfg(csv_config))


####################################### TESTS #######################################


def validate_init(
    csv_saver: CSVDataSaver | None, mock_consumer: MockType, csv_config: DictConfig
):
    """
    Validate the initialization of the CSVDataSaver object.
    """
    assert csv_saver is not None
    assert csv_saver.consumer is not None

    current_date = datetime.now().strftime("%Y_%m_%d")
    assert (
        str(csv_saver.csv_file_path)
        == f"{csv_config.csv_file_path[:-4]}_{current_date}.csv"
    )

    mock_consumer.assert_called_once_with(
        csv_config.streaming.kafka_topic,
        bootstrap_servers=csv_config.streaming.kafka_server,
        auto_offset_reset="earliest",
    )


# Test: 1
def test_init(
    mock_consumer: MockType,
    mocker: MockerFixture,
    csv_config: DictConfig,
    mock_logger: MockType,
):
    """
    Test the initialization of the CSVDataSaver object with all possible ways.
    """
    mock_consumer.return_value = mocker.MagicMock()
    # Test: 1.1 ( valid initialization from configuration )
    csv_saver = CSVDataSaver.from_cfg(csv_config)
    validate_init(csv_saver, mock_consumer, csv_config)
    mock_consumer.reset_mock()

    # Test: 1.2 ( valid initialization using init_from_cfg )
    csv_saver = cast(CSVDataSaver, init_from_cfg(csv_config, DataSaver))
    validate_init(csv_saver, mock_consumer, csv_config)
    mock_consumer.reset_mock()

    consumer = mock_consumer(
        csv_config.streaming.kafka_topic,
        bootstrap_servers=csv_config.streaming.kafka_server,
        auto_offset_reset="earliest",
    )

    # Test: 1.3 ( valid initialization from constructor )
    csv_saver = CSVDataSaver(consumer, csv_config.csv_file_path)
    validate_init(csv_saver, mock_consumer, csv_config)
    mock_consumer.reset_mock()

    # Test: 1.4 ( Test NoBrokersAvailable exception )
    mock_consumer.side_effect = NoBrokersAvailable()
    csv_saver = CSVDataSaver.from_cfg(csv_config)
    assert csv_saver is None
    mock_logger.error.assert_called_once_with(
        "No Broker is available at the address: %s. No data will be saved.",
        "localhost:9092",
    )
    mock_consumer.assert_called_once_with(
        csv_config.streaming.kafka_topic,
        bootstrap_servers=csv_config.streaming.kafka_server,
        auto_offset_reset="earliest",
    )


# Test: 2
def test_retrieve_and_save(csv_saver: CSVDataSaver, kafka_data: list[dict]):
    """
    Test the `retrieve_and_save` method of the CSVDataSaver object.
    """
    encoded_data = [
        Message(value=json.dumps(data).encode("utf-8")) for data in kafka_data
    ]

    # Setting the return value of the consumer to the encoded data
    csv_saver.consumer.__iter__.return_value = encoded_data
    csv_saver.retrieve_and_save()

    assert csv_saver.consumer.__iter__.call_count == 1

    stored_data = pd.read_csv(csv_saver.csv_file_path)
    stored_data = stored_data.to_dict(orient="records")

    # Converting the data to string to compare
    stored_data = [{k: str(v) for k, v in record.items()} for record in stored_data]
    kafka_data = [{k: str(v) for k, v in record.items()} for record in kafka_data]

    assert stored_data == kafka_data


# Test: 3
def test_retrieve_and_save_error(csv_saver: CSVDataSaver, mock_logger: MockType):
    """
    Test the `retrieve_and_save` method of the CSVDataSaver object when an error occurs.
    """
    csv_saver.consumer.__iter__.side_effect = PermissionError("Permission denied")
    csv_saver.retrieve_and_save()

    mock_logger.error.assert_called_once_with(
        "Error while saving data to csv: %s", csv_saver.consumer.__iter__.side_effect
    )
    mock_logger.info.assert_called_once_with("%s messages saved to csv", 0)
    assert csv_saver.consumer.__iter__.call_count == 1
