# pylint: disable=missing-function-docstring

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from confluent_kafka import KafkaException
from omegaconf import OmegaConf

from app.data_layer.data_saver.saver.csv_saver import CSVDataSaver, DataSaver
from app.utils.common import init_from_cfg
from app.utils.constants import (
    KAFKA_CONSUMER_DEFAULT_CONFIG,
    KAFKA_CONSUMER_GROUP_ID_ENV,
)
from app.utils.file_utils import read_csv

#################### FIXTURES ####################


@pytest.fixture
def sample_csv_path(temp_dir):
    """
    Sample CSV file path in temp directory.
    """
    return Path(temp_dir) / "test_data.csv"


@pytest.fixture
def mock_consumer():
    """
    Mock Kafka consumer.
    """
    consumer = MagicMock()
    consumer.poll.return_value = None
    consumer.close.return_value = None
    return consumer


@pytest.fixture
def valid_config(temp_dir):
    """
    Valid configuration for CSVDataSaver.
    """
    return OmegaConf.create(
        {
            "name": "csv_saver",
            "csv_file_path": str(Path(temp_dir) / "data.csv"),
            "streaming": {
                "kafka_server": "localhost:9092",
                "kafka_topic": "test_topic",
            },
        }
    )


@pytest.fixture
def mock_get_kafka_consumer(mocker):
    """
    Mock get_kafka_consumer function.
    """
    return mocker.patch("app.data_layer.data_saver.saver.csv_saver.get_kafka_consumer")


@pytest.fixture
def mock_logger(mocker):
    """
    Mock logger.
    """
    return mocker.patch("app.data_layer.data_saver.saver.csv_saver.logger")


@pytest.fixture(autouse=True)
def fixed_datetime(mocker):
    """
    Mock datetime.now() for consistent testing.
    """
    mock_dt = mocker.patch("app.data_layer.data_saver.saver.csv_saver.datetime")
    mock_dt.now.return_value.strftime.return_value = "2024_01_01"
    return mock_dt


def validate_csv_data_saver_instance(saver, mock_consumer, csv_file_path):
    """
    Helper function to validate CSVDataSaver instance.
    """
    assert isinstance(saver, CSVDataSaver)
    assert saver.consumer == mock_consumer
    assert saver.csv_file_path == Path(csv_file_path).with_name("data_2024_01_01.csv")
    assert saver.csv_file_path.suffix == ".csv"
    assert saver.csv_file_path.parent.exists()


#################### INITIALIZATION TESTS ####################


def test_init_with_string_path(mock_consumer, valid_config):
    """
    Test initialization with string path.
    """

    saver = CSVDataSaver(mock_consumer, valid_config.csv_file_path)
    validate_csv_data_saver_instance(saver, mock_consumer, valid_config.csv_file_path)


def test_init_with_path_object(mock_consumer, valid_config):
    """
    Test initialization with Path object.
    """
    saver = CSVDataSaver(mock_consumer, Path(valid_config.csv_file_path))
    validate_csv_data_saver_instance(saver, mock_consumer, valid_config.csv_file_path)


def test_init_with_existing_directory(mock_consumer, temp_dir):
    """
    Test initialization when directory already exists.
    """
    existing_dir = Path(temp_dir) / "existing"
    existing_dir.mkdir()
    csv_path = existing_dir / "data.csv"

    saver = CSVDataSaver(mock_consumer, csv_path)

    assert saver.csv_file_path.parent.exists()
    assert saver.csv_file_path.name == "data_2024_01_01.csv"


#################### FROM_CFG TESTS ####################


def test_init_from_config(valid_config):
    """
    Test initialization using `init_from_cfg` function.
    """
    saver = init_from_cfg(valid_config, DataSaver)
    validate_csv_data_saver_instance(saver, saver.consumer, valid_config.csv_file_path)


def test_from_cfg_success(valid_config, mock_get_kafka_consumer, mock_consumer):
    """
    Test successful creation from configuration.
    """
    mock_get_kafka_consumer.return_value = mock_consumer

    saver = CSVDataSaver.from_cfg(valid_config)

    assert saver is not None
    validate_csv_data_saver_instance(saver, mock_consumer, valid_config.csv_file_path)
    mock_get_kafka_consumer.assert_called_once()


def test_from_cfg_kafka_consumer_creation_fails(
    valid_config, mock_get_kafka_consumer, mock_logger
):
    """
    Test when Kafka consumer creation fails.
    """
    mock_get_kafka_consumer.side_effect = KafkaException(
        "Kafka consumer creation failed"
    )

    saver = CSVDataSaver.from_cfg(valid_config)

    mock_logger.error.assert_called_once_with(
        "Error while creating CSVDataSaver: %s", mock_get_kafka_consumer.side_effect
    )

    assert saver is None


def test_from_cfg_missing_csv_file_path(
    mock_get_kafka_consumer, mock_consumer, mock_logger
):
    """
    Test when csv_file_path is missing from config.
    """
    config = OmegaConf.create(
        {"streaming": {"kafka_server": "localhost:9092", "kafka_topic": "test_topic"}}
    )
    mock_get_kafka_consumer.return_value = mock_consumer

    saver = CSVDataSaver.from_cfg(config)

    assert saver is None
    mock_logger.error.assert_called_once_with(
        "csv_file_path not provided in configuration"
    )


def test_from_cfg_empty_csv_file_path(
    mock_get_kafka_consumer, mock_consumer, mock_logger
):
    """
    Test when csv_file_path is empty.
    """
    config = OmegaConf.create(
        {
            "csv_file_path": "",
            "streaming": {
                "kafka_server": "localhost:9092",
                "kafka_topic": "test_topic",
            },
        }
    )
    mock_get_kafka_consumer.return_value = mock_consumer

    saver = CSVDataSaver.from_cfg(config)

    assert saver is None
    mock_logger.error.assert_called_once_with(
        "csv_file_path not provided in configuration"
    )


def test_from_cfg_none_csv_file_path(
    mock_get_kafka_consumer, mock_consumer, mock_logger
):
    """
    Test when csv_file_path is None.
    """
    config = OmegaConf.create(
        {
            "csv_file_path": None,
            "streaming": {
                "kafka_server": "localhost:9092",
                "kafka_topic": "test_topic",
            },
        }
    )
    mock_get_kafka_consumer.return_value = mock_consumer

    saver = CSVDataSaver.from_cfg(config)

    assert saver is None
    mock_logger.error.assert_called_once_with(
        "csv_file_path not provided in configuration"
    )


def test_from_cfg_kafka_config_construction(
    valid_config, mock_get_kafka_consumer, mock_consumer
):
    """
    Test that Kafka configuration is constructed correctly.
    """
    mock_get_kafka_consumer.return_value = mock_consumer

    CSVDataSaver.from_cfg(valid_config)

    expected_config = {
        "bootstrap.servers": "localhost:9092",
        "group.id": KAFKA_CONSUMER_GROUP_ID_ENV,
        **KAFKA_CONSUMER_DEFAULT_CONFIG,
    }
    mock_get_kafka_consumer.assert_called_once_with(expected_config, "test_topic")


#################### RETRIEVE_AND_SAVE TESTS ####################


def test_retrieve_and_save_single_message(
    mock_consumer,
    sample_csv_path,
    mock_kafka_message,
    mock_logger,
):
    """
    Test saving a single message to CSV.
    """
    mock_consumer.poll.side_effect = [
        mock_kafka_message,
        None,
        None,
        None,
        KeyboardInterrupt(),
    ]

    saver = CSVDataSaver(mock_consumer, sample_csv_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify file was created and has correct content
    assert saver.csv_file_path.exists()

    rows = read_csv(saver.csv_file_path)

    assert len(rows) == 2  # Header + 1 data row
    message = json.loads(mock_kafka_message.value().decode("utf-8"))
    assert rows[0] == list(message.keys())  # Header
    assert rows[1] == list(map(str, message.values()))

    mock_consumer.close.assert_called_once()
    mock_logger.info.assert_called_once_with("%s messages saved to csv", 1)


def test_retrieve_and_save_multiple_messages(
    mock_consumer, sample_csv_path, sample_data_list, mock_logger
):
    """
    Test saving multiple messages to CSV.
    """
    messages = []
    for data in sample_data_list:
        message = MagicMock()
        message.error.return_value = None
        message.value.return_value.decode.return_value = json.dumps(data)
        messages.append(message)

    mock_consumer.poll.side_effect = messages + [None, None, KeyboardInterrupt()]

    saver = CSVDataSaver(mock_consumer, sample_csv_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify file content
    rows = read_csv(saver.csv_file_path)

    assert len(rows) == 4  # Header + 3 data rows
    assert rows[0] == list(sample_data_list[0].keys())  # Header from first message
    assert all(
        rows[i] == list(map(str, sample_data_list[i - 1].values())) for i in range(1, 4)
    )

    mock_logger.info.assert_called_once_with("%s messages saved to csv", 3)


def test_retrieve_and_save_no_messages(mock_consumer, sample_csv_path, mock_logger):
    """
    Test behavior when no messages are received.
    """
    mock_consumer.poll.side_effect = [None, None, None, KeyboardInterrupt()]

    saver = CSVDataSaver(mock_consumer, sample_csv_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify file was created but is empty (no header because no messages)
    assert saver.csv_file_path.exists()
    assert saver.csv_file_path.stat().st_size == 0

    mock_consumer.close.assert_called_once()
    mock_logger.info.assert_called_once_with("%s messages saved to csv", 0)


def test_retrieve_and_save_kafka_error(
    mock_consumer,
    sample_csv_path,
    mock_logger,
):
    """
    Test handling of Kafka errors.
    """
    message = MagicMock()
    kafka_error = MagicMock()
    kafka_error.code.return_value = 1
    message.error.return_value = kafka_error
    mock_consumer.poll.return_value = message

    saver = CSVDataSaver(mock_consumer, sample_csv_path)
    saver.retrieve_and_save()

    mock_consumer.close.assert_called_once()
    mock_logger.error.assert_called_once()


def test_retrieve_and_save_json_decode_error(
    mock_consumer, sample_csv_path, mock_logger
):
    """
    Test handling of JSON decode errors.
    """
    message = MagicMock()
    message.error.return_value = None
    message.value.return_value.decode.return_value = "invalid json {"
    mock_consumer.poll.side_effect = [message]

    saver = CSVDataSaver(mock_consumer, sample_csv_path)
    saver.retrieve_and_save()  # Should handle error gracefully without raising

    mock_consumer.close.assert_called_once()
    mock_logger.error.assert_called_once()


def test_retrieve_and_save_file_permission_error(
    mock_consumer, temp_dir, mocker, mock_logger
):
    """
    Test handling of file permission errors.
    """
    # Mock open to raise PermissionError
    mocker.patch("builtins.open", side_effect=PermissionError("Permission denied"))

    # Use temp_dir to avoid permission issues with directory creation
    restricted_file = Path(temp_dir) / "restricted.csv"
    saver = CSVDataSaver(mock_consumer, restricted_file)
    saver.retrieve_and_save()

    mock_consumer.close.assert_called_once()
    mock_logger.error.assert_called_once()


def test_retrieve_and_save_ensures_consumer_closed_on_exception(
    mock_consumer, sample_csv_path, mocker, mock_logger
):
    """
    Test that consumer is always closed even when exceptions occur.
    """
    # Mock open to raise an exception
    open_mocker = mocker.patch("builtins.open", side_effect=IOError("Disk full"))

    saver = CSVDataSaver(mock_consumer, sample_csv_path)
    saver.retrieve_and_save()

    mock_consumer.close.assert_called_once()
    mock_logger.error.assert_called_once_with(
        "File I/O error while saving to csv: %s", open_mocker.side_effect
    )


def test_retrieve_and_save_file_flushing(
    mock_consumer, sample_csv_path, mock_kafka_message, mocker
):
    """
    Test that file is flushed after each message.
    """
    mock_file = mocker.MagicMock()
    mocker.patch("builtins.open", return_value=mock_file)
    mock_consumer.poll.side_effect = [mock_kafka_message, KeyboardInterrupt()]

    saver = CSVDataSaver(mock_consumer, sample_csv_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify flush was called
    mock_file.__enter__.return_value.flush.assert_called()


def test_retrieve_and_save_empty_json_object(mock_consumer, sample_csv_path):
    """
    Test handling of empty JSON objects.
    """
    message = MagicMock()
    message.error.return_value = None
    message.value.return_value.decode.return_value = json.dumps({})
    mock_consumer.poll.side_effect = [message, KeyboardInterrupt()]

    saver = CSVDataSaver(mock_consumer, sample_csv_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify file was created with empty header
    rows = read_csv(saver.csv_file_path)

    assert len(rows) == 2  # Empty header + empty values
    assert rows[0] == []
    assert rows[1] == []


def test_retrieve_and_save_different_message_structures(mock_consumer, sample_csv_path):
    """
    Test handling of messages with different structures.
    """
    message_data = [{"a": 1, "b": 2}, {"c": 3, "d": 4}]
    messages = []
    for message in message_data:
        message1 = MagicMock()
        message1.error.return_value = None
        message1.value.return_value.decode.return_value = json.dumps(message)
        messages.append(message1)

    mock_consumer.poll.side_effect = [*messages, KeyboardInterrupt()]

    saver = CSVDataSaver(mock_consumer, sample_csv_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify both messages were processed (header from first message)
    rows = read_csv(saver.csv_file_path)

    assert len(rows) == 3  # Header + 2 data rows
    assert rows[0] == list(message_data[0].keys())
    assert rows[1] == list(map(str, message_data[0].values()))
    assert rows[2] == list(map(str, message_data[1].values()))
    assert rows[0] != list(message_data[1].keys())


def test_retrieve_and_save_unicode_content(mock_consumer, sample_csv_path):
    """
    Test handling of Unicode content in messages.
    """
    message = MagicMock()
    message.error.return_value = None
    message.value.return_value.decode.return_value = json.dumps(
        {"symbol": "测试", "description": "Test with émojis 🚀", "price": 100.50}
    )
    mock_consumer.poll.side_effect = [message, KeyboardInterrupt()]

    saver = CSVDataSaver(mock_consumer, sample_csv_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify Unicode content is properly handled
    rows = read_csv(saver.csv_file_path)

    assert rows[1] == [
        "测试",
        "Test with émojis 🚀",
        "100.5",
    ]  # JSON doesn't preserve trailing zeros


#################### EDGE CASES AND ERROR HANDLING ####################


def test_consumer_poll_timeout_handling(mock_consumer, sample_csv_path):
    """
    Test that polling timeouts are handled correctly.
    """
    mock_consumer.poll.side_effect = [None, None, None, KeyboardInterrupt()]

    saver = CSVDataSaver(mock_consumer, sample_csv_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify poll was called with correct timeout
    assert all(call[1]["timeout"] == 1.0 for call in mock_consumer.poll.call_args_list)


def test_very_large_json_message(mock_consumer, sample_csv_path):
    """
    Test handling of very large JSON messages.
    """
    large_data = {f"field_{i}": f"value_{i}" for i in range(1000)}
    message = MagicMock()
    message.error.return_value = None
    message.value.return_value.decode.return_value = json.dumps(large_data)
    mock_consumer.poll.side_effect = [message, KeyboardInterrupt()]

    saver = CSVDataSaver(mock_consumer, sample_csv_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify large message was processed
    rows = read_csv(saver.csv_file_path)

    assert len(rows) == 2  # Header + 1 data row
    assert len(rows[0]) == 1000  # 1000 fields


def test_nested_json_structures(mock_consumer, sample_csv_path):
    """
    Test handling of nested JSON structures (should be flattened to string).
    """
    nested_data = {
        "simple_field": "value",
        "nested_object": {"inner": "data"},
        "array_field": [1, 2, 3],
    }
    message = MagicMock()
    message.error.return_value = None
    message.value.return_value.decode.return_value = json.dumps(nested_data)
    mock_consumer.poll.side_effect = [message, KeyboardInterrupt()]

    saver = CSVDataSaver(mock_consumer, sample_csv_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify nested structures are converted to strings
    rows = read_csv(saver.csv_file_path)

    assert rows[0] == ["simple_field", "nested_object", "array_field"]
    assert rows[1][0] == "value"

    # Nested structures become string representations
    assert "inner" in rows[1][1] or "data" in rows[1][1]


def test_null_values_in_json(mock_consumer, sample_csv_path):
    """
    Test handling of null values in JSON.
    """
    data_with_nulls = {"symbol": "TEST", "price": None, "volume": 100, "metadata": None}
    message = MagicMock()
    message.error.return_value = None
    message.value.return_value.decode.return_value = json.dumps(data_with_nulls)
    mock_consumer.poll.side_effect = [message, KeyboardInterrupt()]

    saver = CSVDataSaver(mock_consumer, sample_csv_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify null values are handled
    rows = read_csv(saver.csv_file_path)

    assert rows[1] == ["TEST", "", "100", ""]  # None values become empty strings in CSV


def test_special_characters_in_csv(mock_consumer, sample_csv_path):
    """
    Test handling of special CSV characters (commas, quotes, newlines).
    """
    special_data = {
        "field_with_comma": "value, with comma",
        "field_with_quote": 'value "with" quotes',
        "field_with_newline": "value\nwith\nnewline",
    }
    message = MagicMock()
    message.error.return_value = None
    message.value.return_value.decode.return_value = json.dumps(special_data)
    mock_consumer.poll.side_effect = [message, KeyboardInterrupt()]

    saver = CSVDataSaver(mock_consumer, sample_csv_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify special characters are properly escaped in CSV
    with open(saver.csv_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # CSV module should properly escape these
    assert "value, with comma" in content
    assert 'value ""with"" quotes' in content  # CSV doubles quotes for escaping
    assert "value\nwith\nnewline" in content


#################### INTEGRATION TESTS ####################


def test_end_to_end_workflow(valid_config, mock_get_kafka_consumer, sample_data_list):
    """
    Test complete end-to-end workflow from config to file output.
    """
    # Setup
    mock_consumer = MagicMock()
    mock_get_kafka_consumer.return_value = mock_consumer

    messages = []
    for data in sample_data_list:
        message = MagicMock()
        message.error.return_value = None
        message.value.return_value.decode.return_value = json.dumps(data)
        messages.append(message)

    mock_consumer.poll.side_effect = messages + [KeyboardInterrupt()]

    # Execute
    saver = CSVDataSaver.from_cfg(valid_config)
    assert saver is not None

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify
    assert saver.csv_file_path.exists()

    rows = read_csv(saver.csv_file_path)

    assert len(rows) == len(sample_data_list) + 1  # Header + data rows
    assert rows[0] == ["symbol", "price", "volume", "timestamp"]

    # Verify all data was written correctly
    for i, expected_data in enumerate(sample_data_list, 1):
        assert rows[i] == [str(v) for v in expected_data.values()]


def test_real_file_operations(mock_consumer, temp_dir, sample_data_list):
    """
    Test with real file operations (no mocking of file I/O).
    """
    csv_path = Path(temp_dir) / "real_test.csv"

    messages = []
    for data in sample_data_list:
        message = MagicMock()
        message.error.return_value = None
        message.value.return_value.decode.return_value = json.dumps(data)
        messages.append(message)

    mock_consumer.poll.side_effect = messages + [KeyboardInterrupt()]

    saver = CSVDataSaver(mock_consumer, csv_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify file exists and has correct content
    assert saver.csv_file_path.exists()
    assert saver.csv_file_path.stat().st_size > 0

    # Read and verify content
    with open(saver.csv_file_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "symbol,price,volume,timestamp" in content
        assert "AAPL" in content
        assert "GOOGL" in content
        assert "MSFT" in content


def test_concurrent_access_simulation(mock_consumer, temp_dir):
    """
    Test behavior that might occur with concurrent access.
    """
    csv_path = Path(temp_dir) / "concurrent_test.csv"

    # Create file first
    csv_path.touch()

    message = MagicMock()
    message.error.return_value = None
    message.value.return_value.decode.return_value = json.dumps({"test": "data"})
    mock_consumer.poll.side_effect = [message, KeyboardInterrupt()]

    saver = CSVDataSaver(mock_consumer, csv_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # File should be appended to, not overwritten
    assert saver.csv_file_path.exists()


#################### PERFORMANCE AND RESOURCE TESTS ####################


def test_message_counter_accuracy(mock_consumer, sample_csv_path, mock_logger):
    """
    Test that message counter is accurate.
    """
    num_messages = 5
    messages = []
    for i in range(num_messages):
        message = MagicMock()
        message.error.return_value = None
        message.value.return_value.decode.return_value = json.dumps({"id": i})
        messages.append(message)

    mock_consumer.poll.side_effect = messages + [KeyboardInterrupt()]

    saver = CSVDataSaver(mock_consumer, sample_csv_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    mock_logger.info.assert_called_once_with("%s messages saved to csv", num_messages)


def test_resource_cleanup_on_normal_exit(mock_consumer, sample_csv_path):
    """
    Test resource cleanup on normal operation.
    """
    mock_consumer.poll.side_effect = [KeyboardInterrupt()]
    saver = CSVDataSaver(mock_consumer, sample_csv_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    mock_consumer.close.assert_called_once()


def test_file_handle_management(mock_consumer, sample_csv_path, mocker):
    """
    Test that file handles are properly managed.
    """
    mock_file = mocker.MagicMock()
    mock_open_context = mocker.patch("builtins.open", return_value=mock_file)

    message = MagicMock()
    message.error.return_value = None
    message.value.return_value.decode.return_value = json.dumps({"test": "data"})
    mock_consumer.poll.side_effect = [message, KeyboardInterrupt()]

    saver = CSVDataSaver(mock_consumer, sample_csv_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify file was opened with correct parameters
    mock_open_context.assert_called_once_with(
        saver.csv_file_path, "a", encoding="utf-8", newline=""
    )

    # Verify context manager was used (file properly closed)
    mock_file.__enter__.assert_called_once()
    mock_file.__exit__.assert_called_once()
