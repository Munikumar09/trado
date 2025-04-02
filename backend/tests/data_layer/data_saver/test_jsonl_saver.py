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

from app.data_layer.data_saver import DataSaver, JSONLDataSaver
from app.utils.common import init_from_cfg

Message = namedtuple("Message", ["value"])


####################################### FIXTURES #######################################
@pytest.fixture
def jsonl_config() -> DictConfig:
    """
    Configuration for the JSONLDataSaver.

    Returns:
    --------
    ``DictConfig``
        Configuration for the JSONLDataSaver
    """
    return OmegaConf.create(
        {
            "name": "jsonl_saver",
            "jsonl_file_path": TemporaryDirectory().name + "/test.jsonl",
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
    return mocker.patch("app.data_layer.data_saver.jsonl_saver.KafkaConsumer")


@pytest.fixture
def mock_logger(mocker: MockerFixture) -> MockType:
    """
    Mock the logger object in the JSONLDataSaver.
    """
    return mocker.patch("app.data_layer.data_saver.jsonl_saver.logger")


@pytest.fixture
def jsonl_saver(
    mock_consumer: MockType, jsonl_config: DictConfig, mocker: MockerFixture
) -> JSONLDataSaver:
    """
    Fixture to return the JSONLDataSaver object.
    """
    mock_consumer.return_value = mocker.MagicMock()

    return cast(JSONLDataSaver, JSONLDataSaver.from_cfg(jsonl_config))


####################################### TESTS #######################################


def validate_init(
    jsonl_saver: JSONLDataSaver | None,
    mock_consumer: MockType,
    jsonl_config: DictConfig,
):
    """
    Validate the initialization of the JSONLDataSaver object.
    """
    assert jsonl_saver is not None
    assert jsonl_saver.consumer is not None

    current_date = datetime.now().strftime("%Y_%m_%d")
    assert (
        str(jsonl_saver.jsonl_file_path)
        == f"{jsonl_config.jsonl_file_path[:-6]}_{current_date}.jsonl"
    )

    mock_consumer.assert_called_once_with(
        jsonl_config.streaming.kafka_topic,
        bootstrap_servers=jsonl_config.streaming.kafka_server,
        auto_offset_reset="earliest",
    )


# Test: 1
def test_init(
    mock_consumer: MockType,
    mocker: MockerFixture,
    jsonl_config: DictConfig,
    mock_logger: MockType,
):
    """
    Test the initialization of the JSONLDataSaver object with all possible ways.
    """
    mock_consumer.return_value = mocker.MagicMock()
    # Test: 1.1 ( valid initialization from configuration )
    jsonl_saver = JSONLDataSaver.from_cfg(jsonl_config)
    validate_init(jsonl_saver, mock_consumer, jsonl_config)
    mock_consumer.reset_mock()

    # Test: 1.2 ( valid initialization using init_from_cfg )
    jsonl_saver = cast(JSONLDataSaver, init_from_cfg(jsonl_config, DataSaver))
    validate_init(jsonl_saver, mock_consumer, jsonl_config)
    mock_consumer.reset_mock()

    consumer = mock_consumer(
        jsonl_config.streaming.kafka_topic,
        bootstrap_servers=jsonl_config.streaming.kafka_server,
        auto_offset_reset="earliest",
    )

    # Test: 1.3 ( valid initialization from constructor )
    jsonl_saver = JSONLDataSaver(consumer, jsonl_config.jsonl_file_path)
    validate_init(jsonl_saver, mock_consumer, jsonl_config)
    mock_consumer.reset_mock()

    # Test: 1.4 ( Test NoBrokersAvailable exception )
    mock_consumer.side_effect = NoBrokersAvailable()
    jsonl_saver = JSONLDataSaver.from_cfg(jsonl_config)
    assert jsonl_saver is None
    mock_logger.error.assert_called_once_with(
        "No Broker is available at the address: %s. No data will be saved.",
        "localhost:9092",
    )
    mock_consumer.assert_called_once_with(
        jsonl_config.streaming.kafka_topic,
        bootstrap_servers=jsonl_config.streaming.kafka_server,
        auto_offset_reset="earliest",
    )


# Test: 2
def test_retrieve_and_save(jsonl_saver: JSONLDataSaver, kafka_data: list[dict]):
    """
    Test the `retrieve_and_save` method of the JSONLDataSaver object.
    """
    encoded_data = [
        Message(value=json.dumps(data).encode("utf-8")) for data in kafka_data
    ]

    # Setting the return value of the consumer to the encoded data
    jsonl_saver.consumer.__iter__.return_value = encoded_data
    jsonl_saver.retrieve_and_save()

    assert jsonl_saver.consumer.__iter__.call_count == 1

    stored_data = pd.read_json(
        jsonl_saver.jsonl_file_path, lines=True, orient="records"
    )
    stored_data = stored_data.to_dict(orient="records")

    # Converting the data to string to compare
    stored_data = [{k: str(v) for k, v in record.items()} for record in stored_data]
    kafka_data = [{k: str(v) for k, v in record.items()} for record in kafka_data]

    assert stored_data == kafka_data


# Test: 3
def test_retrieve_and_save_error(jsonl_saver: JSONLDataSaver, mock_logger: MockType):
    """
    Test the `retrieve_and_save` method of the JSONLDataSaver object when an error occurs.
    """
    jsonl_saver.consumer.__iter__.side_effect = PermissionError("Permission denied")
    jsonl_saver.retrieve_and_save()

    mock_logger.error.assert_called_once_with(
        "Error while saving data to jsonl: %s",
        jsonl_saver.consumer.__iter__.side_effect,
    )
    mock_logger.info.assert_called_once_with("%s messages saved to jsonl", 0)
    assert jsonl_saver.consumer.__iter__.call_count == 1
