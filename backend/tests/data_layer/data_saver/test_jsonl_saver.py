"""
Comprehensive tests for JSONLDataSaver module using method-based approach.
Tests cover initialization, configuration, core functionality, edge cases, and integration scenarios.
"""

import json
from pathlib import Path
from unittest.mock import ANY, MagicMock, mock_open, patch

import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf
from pytest_mock import MockerFixture, MockType

from app.data_layer.data_saver import DataSaver, JSONLDataSaver
from app.utils.common import init_from_cfg
from app.utils.file_utils import read_jsonl, write_jsonl

# ======================================================================================
# FIXTURES
# ======================================================================================


@pytest.fixture
def basic_config(temp_dir) -> DictConfig:
    """
    Create a basic OmegaConf configuration for JSONLDataSaver with a specified temporary directory as the file path.
    
    Parameters:
        temp_dir: Path to the temporary directory where the JSONL file will be saved.
    
    Returns:
        DictConfig: OmegaConf configuration containing file path and Kafka streaming settings.
    """
    return OmegaConf.create(
        {
            "name": "jsonl_saver",
            "jsonl_file_path": str(Path(temp_dir) / "data.jsonl"),
            "streaming": {
                "kafka_topic": "test_topic",
                "kafka_server": "localhost:9092",
            },
        }
    )


@pytest.fixture
def mock_get_kafka_consumer(mocker: MockerFixture) -> MockType:
    """
    Fixture that patches the get_kafka_consumer function, allowing tests to control Kafka consumer creation.
    """
    return mocker.patch(
        "app.data_layer.data_saver.saver.jsonl_saver.get_kafka_consumer"
    )


@pytest.fixture
def mock_logger(mocker: MockerFixture) -> MockType:
    """
    Creates a mock for the logger used in the JSONLDataSaver module for testing purposes.
    
    Returns:
        MockType: The mocked logger object.
    """
    return mocker.patch("app.data_layer.data_saver.saver.jsonl_saver.logger")


@pytest.fixture(autouse=True)
def mock_datetime(mocker: MockerFixture) -> MockType:
    """
    Fixture that mocks `datetime.now` to always return a fixed date string for consistent test results.
    
    Returns:
        MockType: The mocked datetime object with `now().strftime()` returning "2023_12_01".
    """
    mock_dt = mocker.patch("app.data_layer.data_saver.saver.jsonl_saver.datetime")
    mock_dt.now.return_value.strftime.return_value = "2023_12_01"

    return mock_dt


@pytest.fixture
def mock_consumer():
    """
    Return a MagicMock instance simulating a Kafka consumer for testing purposes.
    """
    consumer = MagicMock()
    return consumer


@pytest.fixture
def sample_json_data():
    """
    Provides a list of sample JSON objects representing user data for use in tests.
    
    Returns:
        list[dict]: A list of dictionaries with keys 'id', 'name', and 'value'.
    """
    return [
        {"id": 1, "name": "Alice", "value": 100.5},
        {"id": 2, "name": "Bob", "value": 200.75},
        {"id": 3, "name": "Charlie", "value": 300.25},
    ]


@pytest.fixture
def unicode_json_data():
    """
    Provides a list of JSON objects containing Unicode characters for use in tests.
    """
    return [
        {"id": 1, "name": "José", "city": "São Paulo", "emoji": "🎉"},
        {"id": 2, "name": "北京", "description": "Test with 中文"},
        {"id": 3, "special": "åäö", "symbols": "€£¥"},
    ]


@pytest.fixture
def edge_case_json_data():
    """
    Return a list of JSON objects containing edge case values for testing, including nulls, booleans, large numbers, nested structures, arrays, and special characters.
    """
    return [
        {"null_value": None, "empty_string": "", "zero": 0},
        {"boolean_true": True, "boolean_false": False},
        {"large_number": 123456789012345, "float": 3.14159},
        {"nested": {"level1": {"level2": {"value": "deep"}}}},
        {"array": [1, 2, 3, {"nested_in_array": "value"}]},
        {"special_chars": "!@#$%^&*()[]{}|\\:;\"'<>,.?/~`"},
    ]


# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================


def validate_jsonl_data_saver_instance(saver, mock_consumer, jsonl_path):
    """
    Assert that a JSONLDataSaver instance is correctly initialized with the expected consumer and file path.
    
    Parameters:
        saver: The JSONLDataSaver instance to validate.
        mock_consumer: The expected Kafka consumer mock.
        jsonl_path: The base path expected for the JSONL file.
    """
    assert isinstance(saver, JSONLDataSaver)
    assert saver.consumer == mock_consumer
    assert saver.jsonl_file_path == Path(jsonl_path).with_name("data_2023_12_01.jsonl")
    assert saver.jsonl_file_path.suffix == ".jsonl"
    assert saver.jsonl_file_path.parent.exists()


def get_message(message_data):
    """
    Create a mock Kafka message with its value set to the JSON-encoded representation of the provided data.
    
    Parameters:
        message_data (dict): The data to encode as the message's value.
    
    Returns:
        MagicMock: A mock Kafka message object with the specified value.
    """
    message = MagicMock()
    message.error.return_value = None
    message.value.return_value.decode.return_value = json.dumps(message_data)
    return message


def get_messages(message_data):
    """
    Create a list of mock Kafka message objects from a list of message data.
    
    Parameters:
        message_data (list): A list of data items to be encoded in mock Kafka messages.
    
    Returns:
        list: A list of mock Kafka message objects corresponding to the input data.
    """
    messages = []
    for data in message_data:
        message = get_message(data)
        messages.append(message)
    return messages


# ======================================================================================
# INITIALIZATION TESTS
# ======================================================================================


def test_init_with_string_path(mock_consumer, basic_config):
    """
    Test initialization with string file path.
    """
    saver = JSONLDataSaver(mock_consumer, basic_config.jsonl_file_path)
    validate_jsonl_data_saver_instance(
        saver, mock_consumer, basic_config.jsonl_file_path
    )


def test_init_with_path_object(mock_consumer, basic_config):
    """
    Test initialization with Path object.
    """
    saver = JSONLDataSaver(mock_consumer, Path(basic_config.jsonl_file_path))
    validate_jsonl_data_saver_instance(
        saver, mock_consumer, basic_config.jsonl_file_path
    )


def test_init_preserves_existing_directories(mock_consumer, temp_dir):
    """
    Test that existing directories are preserved during initialization.
    """
    # Create directory first
    nested_dir = Path(temp_dir) / "existing"
    nested_dir.mkdir()

    file_path = nested_dir / "test.jsonl"
    saver = JSONLDataSaver(mock_consumer, file_path)

    assert nested_dir.exists()
    assert saver.jsonl_file_path.parent == nested_dir


# ======================================================================================
# FROM_CFG CLASS METHOD TESTS
# ======================================================================================


def test_init_from_config(basic_config):
    """
    Test that the `init_from_cfg` function correctly initializes a `JSONLDataSaver` instance using the provided configuration.
    """
    saver = init_from_cfg(basic_config, DataSaver)
    validate_jsonl_data_saver_instance(
        saver, saver.consumer, basic_config.jsonl_file_path
    )


def test_from_cfg_success(basic_config, mock_get_kafka_consumer):
    """
    Test that JSONLDataSaver is successfully created from configuration when a Kafka consumer is provided.
    
    Verifies that the instance is initialized with the correct consumer and file path, and that the Kafka consumer creation function is called once.
    """
    mock_consumer = MagicMock()
    mock_get_kafka_consumer.return_value = mock_consumer

    saver = JSONLDataSaver.from_cfg(basic_config)

    validate_jsonl_data_saver_instance(
        saver, mock_consumer, basic_config.jsonl_file_path
    )
    mock_get_kafka_consumer.assert_called_once()


def test_from_cfg_no_consumer(basic_config, mock_get_kafka_consumer):
    """
    Test that `from_cfg` returns None when Kafka consumer creation fails.
    """
    mock_get_kafka_consumer.return_value = None

    saver = JSONLDataSaver.from_cfg(basic_config)
    mock_get_kafka_consumer.assert_called_once()

    assert saver is None


def test_from_cfg_missing_jsonl_file_path(
    basic_config, mock_get_kafka_consumer, mock_logger
):
    """
    Test that `from_cfg` returns None and logs an error when `jsonl_file_path` is missing from the configuration.
    """
    mock_consumer = MagicMock()
    mock_get_kafka_consumer.return_value = mock_consumer
    del basic_config.jsonl_file_path

    saver = JSONLDataSaver.from_cfg(basic_config)

    assert saver is None
    mock_logger.error.assert_called_once_with(
        "No jsonl_file_path provided in the configuration. No data will be saved."
    )


def test_from_cfg_empty_jsonl_file_path(
    basic_config, mock_get_kafka_consumer, mock_logger
):
    """
    Test that `from_cfg` returns None and logs an error when `jsonl_file_path` is empty in the configuration.
    """
    mock_consumer = MagicMock()
    mock_get_kafka_consumer.return_value = mock_consumer
    basic_config.jsonl_file_path = ""

    saver = JSONLDataSaver.from_cfg(basic_config)

    assert saver is None
    mock_logger.error.assert_called_once_with(
        "No jsonl_file_path provided in the configuration. No data will be saved."
    )


def test_from_cfg_kafka_config_passed_correctly(basic_config, mock_get_kafka_consumer):
    """
    Verify that the Kafka configuration and topic from the provided config are correctly passed to the Kafka consumer creation function during initialization.
    """
    mock_consumer = MagicMock()
    mock_get_kafka_consumer.return_value = mock_consumer

    JSONLDataSaver.from_cfg(basic_config)

    # Verify the call arguments
    args, _ = mock_get_kafka_consumer.call_args
    config, topic = args

    assert config["bootstrap.servers"] == "localhost:9092"
    assert "group.id" in config
    assert topic == "test_topic"


# ======================================================================================
# RETRIEVE_AND_SAVE CORE FUNCTIONALITY TESTS
# ======================================================================================


def test_retrieve_and_save_single_message(
    mock_kafka_message,
    mock_consumer,
    mock_logger,
    basic_config,
):
    """
    Test that a single Kafka message is retrieved and saved to a JSONL file.
    
    Verifies that the file is created, contains the correct message, the consumer is closed, and the appropriate log message is emitted.
    """
    mock_consumer.poll.side_effect = [
        mock_kafka_message,
        None,
        None,
        None,
        KeyboardInterrupt(),
    ]

    saver = JSONLDataSaver(mock_consumer, basic_config.jsonl_file_path)
    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify file was created and has correct content
    assert saver.jsonl_file_path.exists()

    records = read_jsonl(saver.jsonl_file_path)

    assert len(records) == 1
    message = json.loads(mock_kafka_message.value().decode("utf-8"))
    assert records[0] == message

    mock_consumer.close.assert_called_once()
    mock_logger.info.assert_called_once_with("%s messages saved to jsonl", 1)


def test_retrieve_and_save_multiple_messages(
    mock_consumer, basic_config, sample_data_list, mock_logger
):
    """
    Test that multiple Kafka messages are correctly saved to a JSONL file.
    
    Verifies that all provided messages are written to the file in order, the file contains the expected number of records, and an informational log is emitted upon completion.
    """
    messages = get_messages(sample_data_list)

    mock_consumer.poll.side_effect = messages + [None, None, KeyboardInterrupt()]

    saver = JSONLDataSaver(mock_consumer, basic_config.jsonl_file_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify file content
    records = read_jsonl(saver.jsonl_file_path)

    assert len(records) == 3
    assert all(records[i] == sample_data_list[i] for i in range(len(sample_data_list)))
    mock_logger.info.assert_called_once_with("%s messages saved to jsonl", 3)


def test_retrieve_and_save_file_append_mode(
    mock_consumer, temp_dir, sample_data_list, mock_logger
):
    """
    Test that the JSONLDataSaver appends new data to an existing JSONL file without overwriting previous content.
    
    Verifies that after saving additional messages, the file contains both the original and newly appended records in order.
    """
    file_path = f"{temp_dir}/append_test.jsonl"

    # Create the saver first to get the actual file path with date
    saver = JSONLDataSaver(mock_consumer, file_path)

    messages = get_messages(sample_data_list[1:])

    mock_consumer.poll.side_effect = messages + [None, None, KeyboardInterrupt()]

    write_jsonl(saver.jsonl_file_path, [sample_data_list[0]])

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify file content
    records = read_jsonl(saver.jsonl_file_path)

    assert len(records) == 3
    assert all(records[i] == sample_data_list[i] for i in range(len(sample_data_list)))
    mock_logger.info.assert_called_once_with("%s messages saved to jsonl", 2)


def test_retrieve_and_save_consumer_closed(mock_consumer, temp_dir):
    """
    Test that consumer is properly closed after operation.
    """
    file_path = f"{temp_dir}/close_test.jsonl"
    saver = JSONLDataSaver(mock_consumer, file_path)

    # Mock to immediately raise exception to exit loop
    mock_consumer.poll.side_effect = KeyboardInterrupt()

    with patch("builtins.open", mock_open()):
        try:
            saver.retrieve_and_save()
        except KeyboardInterrupt:
            pass

    mock_consumer.close.assert_called_once()


def test_retrieve_and_save_no_messages(mock_consumer, basic_config, mock_logger):
    """
    Test that `retrieve_and_save` creates an empty JSONL file and closes the consumer when no messages are received.
    
    Verifies that the file is created but remains empty, the consumer is closed, and an info log is generated indicating zero messages saved.
    """
    mock_consumer.poll.side_effect = [None, None, None, KeyboardInterrupt()]

    saver = JSONLDataSaver(mock_consumer, basic_config.jsonl_file_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify file was created but is empty (no header because no messages)
    assert saver.jsonl_file_path.exists()
    assert saver.jsonl_file_path.stat().st_size == 0

    mock_consumer.close.assert_called_once()
    mock_logger.info.assert_called_once_with("%s messages saved to jsonl", 0)


def test_retrieve_and_save_kafka_error(
    mock_consumer,
    basic_config,
    mock_logger,
):
    """
    Test that `JSONLDataSaver.retrieve_and_save` handles Kafka errors by logging the error and closing the consumer.
    """
    message = get_message({"error": {"code": 1}})
    mock_consumer.poll.return_value = [message]

    saver = JSONLDataSaver(mock_consumer, basic_config.jsonl_file_path)
    saver.retrieve_and_save()

    mock_consumer.close.assert_called_once()
    mock_logger.error.assert_called_once()


def test_retrieve_and_save_json_decode_error(mock_consumer, basic_config, mock_logger):
    """
    Test handling of JSON decode errors.
    """
    message = get_message("invalid json {")
    mock_consumer.poll.return_value = [message]

    saver = JSONLDataSaver(mock_consumer, basic_config.jsonl_file_path)
    saver.retrieve_and_save()  # Should handle error gracefully without raising

    mock_consumer.close.assert_called_once()
    mock_logger.error.assert_called_once()


def test_retrieve_and_save_file_permission_error(
    mock_consumer, temp_dir, mocker, mock_logger
):
    """
    Test that JSONLDataSaver logs an error and closes the consumer when a file permission error occurs during saving.
    """
    # Mock open to raise PermissionError
    mocker.patch("builtins.open", side_effect=PermissionError("Permission denied"))

    # Use temp_dir to avoid permission issues with directory creation
    restricted_file = Path(temp_dir) / "restricted.jsonl"
    saver = JSONLDataSaver(mock_consumer, restricted_file)
    saver.retrieve_and_save()

    error_calls = mock_logger.error.call_args_list
    assert (
        "call('Error while saving data to jsonl: %s', PermissionError('Permission denied'))"
        == str(error_calls[0])
    )
    mock_consumer.close.assert_called_once()


def test_retrieve_and_save_ensures_consumer_closed_on_exception(
    mock_consumer, basic_config, mocker, mock_logger
):
    """
    Test that consumer is always closed even when exceptions occur.
    """
    # Mock open to raise an exception
    open_mocker = mocker.patch("builtins.open", side_effect=IOError("Disk full"))

    saver = JSONLDataSaver(mock_consumer, basic_config.jsonl_file_path)
    saver.retrieve_and_save()

    mock_consumer.close.assert_called_once()
    mock_logger.error.assert_called_once_with(
        "Error while saving data to jsonl: %s", open_mocker.side_effect
    )


def test_retrieve_and_save_file_flushing(
    mock_consumer, basic_config, mock_kafka_message, mocker
):
    """
    Test that the JSONLDataSaver flushes the file after writing each message during retrieval and saving.
    """
    mock_file = mocker.MagicMock()
    mocker.patch("builtins.open", return_value=mock_file)
    mock_consumer.poll.side_effect = [mock_kafka_message, KeyboardInterrupt()]

    saver = JSONLDataSaver(mock_consumer, basic_config.jsonl_file_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify flush was called
    mock_file.__enter__.return_value.flush.assert_called()


def test_retrieve_and_save_empty_json_object(mock_consumer, basic_config):
    """
    Test that `JSONLDataSaver` correctly saves an empty JSON object from a Kafka message to the JSONL file.
    """
    message = get_message({})
    mock_consumer.poll.side_effect = [message, KeyboardInterrupt()]

    saver = JSONLDataSaver(mock_consumer, basic_config.jsonl_file_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    records = read_jsonl(saver.jsonl_file_path)

    assert len(records) == 1
    assert records[0] == {}


def test_retrieve_and_save_different_message_structures(mock_consumer, basic_config):
    """
    Tests that `JSONLDataSaver` correctly saves messages with varying JSON structures to a JSONL file.
    
    Verifies that multiple messages with different key sets are processed and written as separate records.
    """
    message_data = [{"a": 1, "b": 2}, {"c": 3, "d": 4}]
    messages = get_messages(message_data)

    mock_consumer.poll.side_effect = [*messages, KeyboardInterrupt()]

    saver = JSONLDataSaver(mock_consumer, basic_config.jsonl_file_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify both messages were processed (header from first message)
    records = read_jsonl(saver.jsonl_file_path)

    assert len(records) == 2
    assert records == message_data


def test_retrieve_and_save_unicode_content(mock_consumer, basic_config):
    """
    Tests that `JSONLDataSaver` correctly saves and preserves Unicode and special character content from Kafka messages to a JSONL file.
    """

    message_data = {
        "symbol": "测试",
        "description": "Test with émojis 🚀",
        "price": 100.50,
        "special": 'quotes"backslash\\newline\ntab\t',
    }
    message = get_message(message_data)
    mock_consumer.poll.side_effect = [message, KeyboardInterrupt()]

    saver = JSONLDataSaver(mock_consumer, basic_config.jsonl_file_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify Unicode content is properly handled
    records = read_jsonl(saver.jsonl_file_path)
    print(records)

    assert records[0] == message_data


# ======================================================================================
# ERROR HANDLING TESTS
# ======================================================================================


def test_retrieve_and_save_general_exception(mock_consumer, basic_config, mock_logger):
    """
    Test that `JSONLDataSaver.retrieve_and_save` logs an error and closes the consumer when a general exception occurs during message polling.
    """
    saver = JSONLDataSaver(mock_consumer, basic_config.jsonl_file_path)

    # Mock consumer to raise exception
    mock_consumer.poll.side_effect = RuntimeError("Unexpected error")

    saver.retrieve_and_save()

    mock_logger.error.assert_called_once_with(
        "Error while saving data to jsonl: %s", ANY
    )
    mock_consumer.close.assert_called_once()


# ======================================================================================
# EDGE CASE TESTS
# ======================================================================================


def test_retrieve_and_save_null_values(
    mock_consumer, basic_config, edge_case_json_data
):
    """
    Test that JSONLDataSaver correctly saves messages containing null values and edge case data.
    
    Verifies that all records with nulls and edge cases are written to the JSONL file and match the expected input.
    """
    saver = JSONLDataSaver(mock_consumer, basic_config.jsonl_file_path)

    # Create mock messages for edge case data
    messages = get_messages(edge_case_json_data)
    mock_consumer.poll.side_effect = [*messages, KeyboardInterrupt()]

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify file content
    records = read_jsonl(saver.jsonl_file_path)
    assert len(records) == len(edge_case_json_data)

    for record, expected in zip(records, edge_case_json_data):
        assert record == expected


def test_retrieve_and_save_large_json_object(mock_consumer, basic_config):
    """
    Test that the JSONLDataSaver correctly saves large JSON objects, including large arrays, long strings, and deeply nested structures, and that the saved file contains the expected data.
    """
    saver = JSONLDataSaver(mock_consumer, basic_config.jsonl_file_path)

    # Create a large JSON object
    large_data = [
        {
            "large_array": list(range(1000)),
            "large_string": "x" * 10000,
            "nested": {
                "level_" + str(i): {"data": "value_" + str(i)} for i in range(100)
            },
        }
    ]
    message = get_message(large_data)
    mock_consumer.poll.side_effect = [message, KeyboardInterrupt()]

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify file content
    records = read_jsonl(saver.jsonl_file_path)
    assert len(records) == 1
    assert records[0] == large_data


# ======================================================================================
# INTEGRATION TESTS
# ======================================================================================


def test_pandas_compatibility(mock_consumer, basic_config, sample_json_data):
    """
    Verifies that JSONL files saved by JSONLDataSaver can be read into pandas DataFrames with correct structure and data integrity.
    """
    saver = JSONLDataSaver(mock_consumer, basic_config.jsonl_file_path)
    messages = get_messages(sample_json_data)
    mock_consumer.poll.side_effect = [*messages, KeyboardInterrupt()]

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Test pandas can read the file
    df = pd.read_json(saver.jsonl_file_path, lines=True)

    assert len(df) == len(sample_json_data)
    assert list(df.columns) == list(sample_json_data[0].keys())

    # Verify data integrity
    df_dict = df.to_dict("records")
    for i, row in enumerate(df_dict):
        for key, value in sample_json_data[i].items():
            assert row[key] == value


# ======================================================================================
# RESOURCE MANAGEMENT TESTS
# ======================================================================================


def test_concurrent_file_access(mock_consumer, basic_config):
    """
    Tests that JSONLDataSaver correctly appends new data to a JSONL file when the file is already populated, simulating concurrent file access.
    """
    saver = JSONLDataSaver(mock_consumer, basic_config.jsonl_file_path)
    test_data = [{"initial": "data"}, {"new": "data"}]

    write_jsonl(saver.jsonl_file_path, test_data[0:1])
    message = get_message(test_data[1])
    mock_consumer.poll.side_effect = [message, KeyboardInterrupt()]

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    records = read_jsonl(saver.jsonl_file_path)
    assert len(records) == 2
    assert records == test_data
