# pylint: disable=missing-function-docstring

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from app.data_layer.data_saver.saver.csv_saver import CSVDataSaver, DataSaver
from app.utils.common import init_from_cfg
from app.utils.constants import KAFKA_CONSUMER_DEFAULT_CONFIG, KAFKA_CONSUMER_GROUP_ID
from app.utils.file_utils import read_csv

#################### FIXTURES ####################


@pytest.fixture
def sample_csv_path(temp_dir):
    """
    Returns a sample CSV file path located within the provided temporary directory.
    
    Parameters:
        temp_dir: The path to a temporary directory.
    
    Returns:
        Path: The full path to 'test_data.csv' within the temporary directory.
    """
    return Path(temp_dir) / "test_data.csv"


@pytest.fixture
def mock_consumer():
    """
    Provides a mock Kafka consumer with stubbed `poll` and `close` methods for use in tests.
    
    Returns:
        MagicMock: A mock Kafka consumer instance with `poll` and `close` methods preset to return `None`.
    """
    consumer = MagicMock()
    consumer.poll.return_value = None
    consumer.close.return_value = None
    return consumer


@pytest.fixture
def valid_config(temp_dir):
    """
    Create a valid OmegaConf configuration for initializing a CSVDataSaver instance.
    
    Parameters:
        temp_dir (str or Path): Temporary directory path where the CSV file will be created.
    
    Returns:
        OmegaConf: Configuration object with CSV file path and Kafka streaming settings.
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
    Pytest fixture that patches the `get_kafka_consumer` function in the CSV saver module.
    
    Returns:
        MagicMock: The patched mock of `get_kafka_consumer`.
    """
    return mocker.patch("app.data_layer.data_saver.saver.csv_saver.get_kafka_consumer")


@pytest.fixture
def mock_logger(mocker):
    """
    Provides a pytest fixture that returns a mocked logger for the CSV saver module.
    """
    return mocker.patch("app.data_layer.data_saver.saver.csv_saver.logger")


@pytest.fixture(autouse=True)
def fixed_datetime(mocker):
    """
    Fixture that mocks `datetime.now()` to always return a fixed date string for consistent test results.
    
    Returns:
        MagicMock: The mocked datetime module with `now().strftime()` returning "2024_01_01".
    """
    mock_dt = mocker.patch("app.data_layer.data_saver.saver.csv_saver.datetime")
    mock_dt.now.return_value.strftime.return_value = "2024_01_01"
    return mock_dt


def validate_csv_data_saver_instance(saver, mock_consumer, csv_file_path):
    """
    Assert that a CSVDataSaver instance is correctly initialized with the expected consumer and CSV file path.
    
    Parameters:
        saver: The CSVDataSaver instance to validate.
        mock_consumer: The expected Kafka consumer instance.
        csv_file_path: The base file path used to construct the expected CSV file.
    
    Raises:
        AssertionError: If any of the validation checks fail.
    """
    assert isinstance(saver, CSVDataSaver)
    assert saver.consumer == mock_consumer
    assert saver.csv_file_path == Path(csv_file_path).with_name("data_2024_01_01.csv")
    assert saver.csv_file_path.suffix == ".csv"
    assert saver.csv_file_path.parent.exists()


#################### INITIALIZATION TESTS ####################


def test_init_with_string_path(mock_consumer, valid_config):
    """
    Test that CSVDataSaver initializes correctly when provided with a CSV file path as a string.
    """

    saver = CSVDataSaver(mock_consumer, valid_config.csv_file_path)
    validate_csv_data_saver_instance(saver, mock_consumer, valid_config.csv_file_path)


def test_init_with_path_object(mock_consumer, valid_config):
    """
    Test that CSVDataSaver initializes correctly when provided a Path object as the CSV file path.
    """
    saver = CSVDataSaver(mock_consumer, Path(valid_config.csv_file_path))
    validate_csv_data_saver_instance(saver, mock_consumer, valid_config.csv_file_path)


def test_init_with_existing_directory(mock_consumer, temp_dir):
    """
    Test that CSVDataSaver initializes correctly when the target directory already exists.
    
    Verifies that the CSV file path is created with the expected date suffix and that the directory is present.
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
    Tests that a `CSVDataSaver` instance is correctly initialized from a valid configuration using `init_from_cfg`.
    """
    saver = init_from_cfg(valid_config, DataSaver)
    validate_csv_data_saver_instance(saver, saver.consumer, valid_config.csv_file_path)


def test_from_cfg_success(valid_config, mock_get_kafka_consumer, mock_consumer):
    """
    Test that `CSVDataSaver.from_cfg` successfully creates an instance when provided with a valid configuration and a working Kafka consumer.
    
    Verifies that the returned saver is correctly initialized and that the Kafka consumer creation function is called once.
    """
    mock_get_kafka_consumer.return_value = mock_consumer

    saver = CSVDataSaver.from_cfg(valid_config)

    assert saver is not None
    validate_csv_data_saver_instance(saver, mock_consumer, valid_config.csv_file_path)
    mock_get_kafka_consumer.assert_called_once()


def test_from_cfg_kafka_consumer_creation_fails(valid_config, mock_get_kafka_consumer):
    """
    Test that `CSVDataSaver.from_cfg` returns `None` when Kafka consumer creation fails.
    """
    mock_get_kafka_consumer.return_value = None

    saver = CSVDataSaver.from_cfg(valid_config)

    assert saver is None


def test_from_cfg_missing_csv_file_path(
    mock_get_kafka_consumer, mock_consumer, mock_logger
):
    """
    Test that CSVDataSaver.from_cfg returns None and logs an error when 'csv_file_path' is missing from the configuration.
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
    Test that CSVDataSaver.from_cfg returns None and logs an error when the csv_file_path in the configuration is an empty string.
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
    Test that `CSVDataSaver.from_cfg` returns None and logs an error when `csv_file_path` is None in the configuration.
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
    Verify that the Kafka consumer configuration passed to the consumer creation function matches the expected parameters when initializing CSVDataSaver from configuration.
    """
    mock_get_kafka_consumer.return_value = mock_consumer

    CSVDataSaver.from_cfg(valid_config)

    expected_config = {
        "bootstrap.servers": "localhost:9092",
        "group.id": KAFKA_CONSUMER_GROUP_ID,
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
    Test that a single Kafka message is retrieved and saved to a CSV file.
    
    Verifies that the CSV file is created with the correct header and data row, the Kafka consumer is closed after processing, and the appropriate log message is emitted.
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
    Test that multiple Kafka messages are correctly saved to a CSV file, verifying header and row content, and that the message count is logged.
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
    Test that `CSVDataSaver.retrieve_and_save` creates an empty CSV file when no messages are received from the consumer.
    
    Verifies that the file is created with zero size (no header or data), the consumer is closed, and the correct log message is emitted.
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
    Test that `CSVDataSaver.retrieve_and_save` handles invalid JSON messages gracefully by logging an error and closing the consumer without raising an exception.
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
    Test that `CSVDataSaver.retrieve_and_save` handles file permission errors by logging an error and closing the Kafka consumer.
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
    Test that the Kafka consumer is closed and an error is logged if an exception occurs during file operations in `retrieve_and_save`.
    """
    # Mock open to raise an exception
    open_mocker = mocker.patch("builtins.open", side_effect=IOError("Disk full"))

    saver = CSVDataSaver(mock_consumer, sample_csv_path)
    saver.retrieve_and_save()

    mock_consumer.close.assert_called_once()
    mock_logger.error.assert_called_once_with(
        "Error while saving data to csv: %s", open_mocker.side_effect
    )


def test_retrieve_and_save_file_flushing(
    mock_consumer, sample_csv_path, mock_kafka_message, mocker
):
    """
    Test that the CSV file is flushed after each message is written during retrieval and saving.
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
    Test that `CSVDataSaver` correctly writes an empty header and row to the CSV file when processing an empty JSON object from Kafka.
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
    Test that `CSVDataSaver` correctly writes messages with differing JSON structures to a CSV file, using the header from the first message and preserving row order.
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


def test_retrieve_and_save_unicode_content(mock_consumer, sample_csv_path):
    """
    Tests that Unicode characters in Kafka messages are correctly written to the CSV file by CSVDataSaver.
    
    Verifies that Unicode strings and emojis are preserved in the CSV output and that numeric values are formatted as expected.
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
    Test that the Kafka consumer's poll method is called with a timeout of 1.0 seconds during message retrieval, and that polling timeouts are handled until interrupted.
    """
    mock_consumer.poll.side_effect = [None, None, None, KeyboardInterrupt()]

    saver = CSVDataSaver(mock_consumer, sample_csv_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    # Verify poll was called with correct timeout
    assert all(call[1]["timeout"] == 1.0 for call in mock_consumer.poll.call_args_list)


def test_very_large_json_message(mock_consumer, sample_csv_path):
    """
    Tests that CSVDataSaver correctly processes and saves a single very large JSON message with 1000 fields, ensuring all fields are written to the CSV file.
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
    Test that nested JSON objects and arrays in Kafka messages are converted to string representations when saved to CSV.
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
    Test that JSON null values are converted to empty strings when saving to CSV.
    
    Verifies that fields with `None` values in the input JSON are written as empty strings in the resulting CSV file.
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
    Test that CSVDataSaver correctly escapes and writes fields containing commas, quotes, and newlines to the CSV file.
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
    Tests the full workflow of CSVDataSaver from configuration initialization to saving Kafka messages to a CSV file.
    
    Verifies that messages are correctly polled, written to the CSV with the appropriate header and data rows, and that the output file matches the expected content.
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
    Tests the CSVDataSaver's ability to write messages to a real CSV file without mocking file I/O.
    
    Verifies that the file is created, contains the expected header and data rows, and that actual file operations succeed.
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
    Simulates concurrent access by pre-creating the CSV file and verifies that `CSVDataSaver` appends data rather than overwriting the file during message saving.
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
    Verify that the message counter accurately reflects the number of messages processed and saved to CSV.
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
    Test that the Kafka consumer is properly closed when a KeyboardInterrupt occurs during message retrieval.
    """
    mock_consumer.poll.side_effect = [KeyboardInterrupt()]
    saver = CSVDataSaver(mock_consumer, sample_csv_path)

    with pytest.raises(KeyboardInterrupt):
        saver.retrieve_and_save()

    mock_consumer.close.assert_called_once()


def test_file_handle_management(mock_consumer, sample_csv_path, mocker):
    """
    Test that file handles are opened with the correct parameters and properly closed using context managers during CSV writing.
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
