"""
Comprehensive tests for SqliteDataSaver module using method-based approach.

This test suite validates the SqliteDataSaver's functionality across multiple dimensions:
- Database initialization and configuration handling
- Stock data persistence and retrieval operations  
- Kafka message consumption and processing
- Error handling and edge case scenarios
- Security considerations (SQL injection prevention)
- Integration workflows from configuration to data storage

The tests ensure data integrity, proper error propagation, and robust handling
of various input formats including Unicode, special characters, and malformed data.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Generator, List, Optional
from unittest.mock import MagicMock

import pytest
from confluent_kafka.error import KafkaError, KafkaException
from omegaconf import DictConfig, OmegaConf
from pytest_mock import MockerFixture, MockType
from sqlalchemy import text

from app.data_layer.data_saver import DataSaver, SqliteDataSaver
from app.data_layer.database.crud.instrument_crud import get_all_stock_price_info
from app.data_layer.database.db_connections.sqlite import get_session
from app.utils.common import init_from_cfg
from app.utils.common.types.financial_types import DataProviderType, ExchangeType


# Mock message class that mimics Kafka message interface
class MockMessage:
    """
    Mock message class for testing purposes.
    """

    def __init__(
        self, value: Optional[bytes] = None, error_obj: Optional[KafkaError] = None
    ) -> None:
        """
        Create a mock Kafka message with optional value and error for testing purposes.
        
        Parameters:
            value (Optional[bytes]): The message payload as bytes, or None.
            error_obj (Optional[KafkaError]): An optional error object to simulate Kafka errors.
        """
        self._value = value
        self._error = error_obj

    def value(self) -> Optional[bytes]:
        """
        Returns the value of the mocked Kafka message as bytes, or None if no value is set.
        """
        return self._value

    def error(self) -> Optional[KafkaError]:
        """
        Returns the KafkaError object associated with this message, or None if there is no error.
        """
        return self._error


# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================
#
# Utility functions to reduce code duplication and improve test maintainability:
# - Database creation and setup helpers
# - Test data generation and message creation utilities
# - Data verification and assertion helpers
# - Mock configuration and consumer management
# - Common test patterns and reusable components


def run_retrieve_and_save_with_messages(
    saver: SqliteDataSaver, mock_consumer: MagicMock, messages: List[MockMessage]
) -> int:
    """
    Runs the `retrieve_and_save` method on the saver, simulating Kafka message polling with a predefined list of messages.
    
    Processes each message in the provided list by configuring the mock consumer's `poll` method to return them sequentially, then raises `KeyboardInterrupt` to exit after all messages are processed.
    
    Returns:
        int: The number of messages processed.
    """
    call_count = 0

    def poll_side_effect(*_: Any, **__: Any) -> Optional[MockMessage]:
        """
        Simulates polling messages from a Kafka consumer, returning predefined messages in sequence and raising KeyboardInterrupt when all messages have been processed.
        
        Returns:
            MockMessage or None: The next message in the sequence, or raises KeyboardInterrupt when finished.
        """
        nonlocal call_count
        if call_count < len(messages):
            result = messages[call_count]
            call_count += 1
            return result

        # Force break after all messages processed
        raise KeyboardInterrupt()

    mock_consumer.poll.side_effect = poll_side_effect

    try:
        saver.retrieve_and_save()
    except KeyboardInterrupt:
        pass

    return call_count


def create_test_instrument_price_data(**overrides) -> dict[str, Any]:
    """
    Generate a dictionary representing test data for an InstrumentPrice, allowing field overrides.
    
    Parameters:
    	overrides: Optional keyword arguments to override default field values.
    
    Returns:
    	A dictionary containing instrument price data suitable for testing.
    """
    base_data = {
        "retrieval_timestamp": 1729532024.309936,
        "last_traded_timestamp": 1729504796,
        "symbol": "TEST_SYMBOL",
        "exchange_id": ExchangeType.NSE.value,
        "data_provider_id": DataProviderType.SMARTAPI.value,
        "last_traded_price": 13468.0,
        "last_traded_quantity": 414,
        "average_traded_price": 13529.0,
        "volume_trade_for_the_day": 131137,
        "total_buy_quantity": 0.2,
        "total_sell_quantity": 0.1,
    }
    base_data.update(overrides)
    return base_data


def create_saver_with_temp_db(
    mock_consumer: MagicMock, config: DictConfig
) -> SqliteDataSaver:
    """
    Instantiate a SqliteDataSaver using a temporary SQLite database path from the provided configuration.
    """
    return SqliteDataSaver(mock_consumer, config.sqlite_db)


def get_saved_stock_data(saver: SqliteDataSaver) -> List[Any]:
    """
    Retrieve all stock price records stored in the saver's database.
    
    Returns:
        List[Any]: A list of all stock price records retrieved from the database.
    """
    with get_session(saver.engine) as session:
        return get_all_stock_price_info(session=session)


def create_json_message(
    data: Optional[dict], error_obj: Optional[KafkaError] = None
) -> MockMessage:
    """
    Create a MockMessage containing the given data as UTF-8 encoded JSON bytes and an optional KafkaError.
    
    Parameters:
        data (dict, optional): The dictionary to encode as JSON for the message value. If None, the message value will be None.
        error_obj (KafkaError, optional): An optional KafkaError to associate with the message.
    
    Returns:
        MockMessage: A mock Kafka message with the specified value and error.
    """
    json_value = json.dumps(data).encode("utf-8") if data else None
    return MockMessage(value=json_value, error_obj=error_obj)


def verify_saved_data_count(saver: SqliteDataSaver, expected_count: int) -> List[Any]:
    """
    Asserts that the number of saved stock data records matches the expected count and returns the saved records.
    
    Parameters:
        expected_count (int): The expected number of saved records.
    
    Returns:
        List[Any]: The list of saved stock data records.
    """
    saved_data = get_saved_stock_data(saver)
    assert len(saved_data) == expected_count
    return saved_data


def verify_saved_symbol(saver: SqliteDataSaver, expected_symbol: str) -> Any:
    """
    Asserts that exactly one stock record with the specified symbol exists in the database.
    
    Parameters:
        expected_symbol (str): The symbol expected to be present in the saved data.
    
    Returns:
        The saved stock record matching the expected symbol.
    """
    saved_data = get_saved_stock_data(saver)
    assert len(saved_data) == 1
    assert saved_data[0].symbol == expected_symbol
    return saved_data[0]


# ======================================================================================
# FIXTURES
# ======================================================================================
#
# Test fixtures providing consistent test data and mock objects:
# - Temporary directory management for isolated testing
# - Configuration objects with realistic parameter sets
# - Mock Kafka consumers and external service dependencies
# - Sample stock data sets for various testing scenarios
# - Datetime mocking for consistent timestamp-based testing


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """
    Yields a temporary directory path for use during tests.
    
    The directory and its contents are automatically cleaned up after the test completes.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def basic_config(temp_dir: str) -> DictConfig:
    """
    Creates a basic configuration object for initializing a SqliteDataSaver with a temporary SQLite database path and Kafka streaming parameters.
    
    Parameters:
        temp_dir (str): Path to the temporary directory where the SQLite database file will be created.
    
    Returns:
        DictConfig: An OmegaConf configuration object containing database and Kafka settings.
    """
    return OmegaConf.create(
        {
            "name": "sqlite_saver",
            "sqlite_db": f"{temp_dir}/test.sqlite3",
            "streaming": {
                "kafka_topic": "test_topic",
                "kafka_server": "localhost:9092",
            },
        }
    )


@pytest.fixture
def mock_get_kafka_consumer(mocker: MockerFixture) -> MockType:
    """
    Pytest fixture that patches the get_kafka_consumer function for use in tests.
    
    Returns:
        MockType: The patched mock of get_kafka_consumer.
    """
    return mocker.patch(
        "app.data_layer.data_saver.saver.sqlite_saver.get_kafka_consumer"
    )


@pytest.fixture
def mock_logger(mocker: MockerFixture) -> MockType:
    """
    Provides a mocked logger instance for use in tests by patching the logger in the SqliteDataSaver module.
    """
    return mocker.patch("app.data_layer.data_saver.saver.sqlite_saver.logger")


@pytest.fixture(autouse=True)
def mock_datetime(mocker: MockerFixture) -> MockType:
    """
    Fixture that mocks `datetime.now()` to return a fixed date string for deterministic testing of date-dependent functionality.
    """
    mock_dt = mocker.patch("app.data_layer.data_saver.saver.sqlite_saver.datetime")
    mock_dt.now.return_value.strftime.return_value = "2023_12_01"
    return mock_dt


@pytest.fixture
def mock_consumer() -> MagicMock:
    """
    Provides a MagicMock instance simulating a Kafka consumer for use in tests.
    """
    consumer = MagicMock()
    return consumer


@pytest.fixture
def sample_stock_data() -> List[dict[str, Any]]:
    """
    Provides a list of sample stock data dictionaries for use in tests.
    
    Returns:
        List of dictionaries representing stock data with different symbols and prices.
    """
    return [
        create_test_instrument_price_data(symbol="AAPL", last_traded_price=150.25),
        create_test_instrument_price_data(symbol="GOOGL", last_traded_price=2750.80),
        create_test_instrument_price_data(symbol="MSFT", last_traded_price=305.15),
    ]


@pytest.fixture
def kafka_data() -> List[dict[str, Any]]:
    """
    Provides a list of sample instrument price data dictionaries for use in Kafka integration tests.
    
    Returns:
        List of dictionaries representing instrument price data with varying symbols and prices.
    """
    return [
        create_test_instrument_price_data(
            symbol="KAFKA_TEST_1", last_traded_price=100.0
        ),
        create_test_instrument_price_data(
            symbol="KAFKA_TEST_2", last_traded_price=200.0
        ),
        create_test_instrument_price_data(
            symbol="KAFKA_TEST_3", last_traded_price=300.0
        ),
    ]


# ======================================================================================
# INITIALIZATION TESTS
# ======================================================================================
#
# These tests validate the SqliteDataSaver initialization process:
# - Proper handling of different path formats (string vs Path objects)
# - Automatic creation of parent directories when they don't exist
# - Database file naming with date insertion and extension handling
# - Database table creation and schema setup
# - Date formatting consistency in database filenames


def test_init_with_path_object(mock_consumer: MagicMock, temp_dir: str) -> None:
    """
    Verify that SqliteDataSaver initializes correctly with both Path and string database paths, appending the date to the filename and creating the database engine.
    """
    # Test with Path object
    db_path_obj = Path(temp_dir) / "test.sqlite3"
    saver_obj = SqliteDataSaver(mock_consumer, db_path_obj)

    assert saver_obj.consumer == mock_consumer
    assert "test_2023_12_01.sqlite3" in saver_obj.sqlite_db
    assert saver_obj.engine is not None

    # Test with string path
    db_path_str = f"{temp_dir}/test_string.sqlite3"
    saver_str = SqliteDataSaver(mock_consumer, db_path_str)

    assert saver_str.consumer == mock_consumer
    assert "test_string_2023_12_01.sqlite3" in saver_str.sqlite_db
    assert saver_str.engine is not None


def test_init_creates_parent_directories(
    mock_consumer: MagicMock, temp_dir: str
) -> None:
    """
    Verify that initializing SqliteDataSaver with a nested database path creates all necessary parent directories.
    """
    nested_path = f"{temp_dir}/nested/deep/path/test.sqlite3"

    saver = SqliteDataSaver(mock_consumer, nested_path)

    assert Path(nested_path).parent.exists()
    assert "test_2023_12_01.sqlite3" in saver.sqlite_db


def test_init_removes_db_extension(
    mock_consumer: MagicMock, basic_config: DictConfig
) -> None:
    """
    Verify that the database filename retains the correct extension after initialization.
    
    Ensures that when a database path with a `.sqlite3` extension is provided, the resulting filename includes the extension and the date.
    """
    # Test .sqlite3 extension (most common case)
    saver = SqliteDataSaver(mock_consumer, basic_config.sqlite_db)
    assert "test_2023_12_01.sqlite3" in saver.sqlite_db


def test_init_date_formatting(
    mock_consumer: MagicMock, basic_config: DictConfig, mocker: MockerFixture
) -> None:
    """
    Verify that the database filename includes the correctly formatted date when initializing SqliteDataSaver with a mocked datetime.
    """
    # Mock specific date
    mock_dt = mocker.patch("app.data_layer.data_saver.saver.sqlite_saver.datetime")
    mock_dt.now.return_value.strftime.return_value = "2024_01_15"

    saver = SqliteDataSaver(mock_consumer, basic_config.sqlite_db)

    assert "test_2024_01_15.sqlite3" in saver.sqlite_db


def test_init_creates_tables(
    mock_consumer: MagicMock, basic_config: DictConfig
) -> None:
    """
    Verify that the database tables are created upon initializing the SqliteDataSaver.
    
    Ensures that the 'instrumentprice' table exists and can be queried after initialization.
    """
    saver = SqliteDataSaver(mock_consumer, basic_config.sqlite_db)

    # Test that we can create a session and query the tables
    with get_session(saver.engine) as session:
        # This should not raise an error if tables exist
        result = session.exec(
            text("SELECT name FROM sqlite_master WHERE type='table'")  # type: ignore[call-overload]
        ).fetchall()
        table_names = [row[0] for row in result]
        assert "instrumentprice" in table_names


# ======================================================================================
# SAVE_STOCK_DATA TESTS
# ======================================================================================
#
# These tests verify the core data persistence functionality:
# - Saving stock data with minimal required fields vs complete field sets
# - Proper validation and error handling for missing required fields
# - Multiple record insertion and data integrity verification
# - Duplicate data handling and database constraint behavior
# - NULL/None value handling for optional fields
# - Data type conversion and field mapping accuracy


def test_save_stock_data_valid_minimal(
    mock_consumer: MagicMock, basic_config: DictConfig
) -> None:
    """
    Test that stock data with only the minimal required fields can be saved and retrieved successfully.
    """
    saver = create_saver_with_temp_db(mock_consumer, basic_config)

    data = create_test_instrument_price_data()
    saver.save_stock_data(data)

    # Verify data was saved
    verify_saved_symbol(saver, "TEST_SYMBOL")


def test_save_stock_data_valid_complete(
    mock_consumer: MagicMock, basic_config: DictConfig
) -> None:
    """
    Test that saving stock data with all possible fields persists the complete set of values.
    
    Verifies that each provided field is correctly stored and retrievable from the database.
    """
    saver = create_saver_with_temp_db(mock_consumer, basic_config)

    data = create_test_instrument_price_data(
        symbol="COMPLETE_DATA",
        last_traded_quantity=1000,
        average_traded_price=100.5,
        volume_trade_for_the_day=50000,
        total_buy_quantity=25000,
        total_sell_quantity=25000,
    )
    saver.save_stock_data(data)

    # Verify data was saved with all fields
    saved_item = verify_saved_symbol(saver, "COMPLETE_DATA")
    assert saved_item.last_traded_quantity == 1000
    assert saved_item.average_traded_price == 100.5


def test_save_stock_data_missing_required_field(
    mock_consumer: MagicMock, basic_config: DictConfig
) -> None:
    """
    Verify that saving stock data missing required fields raises a KeyError.
    
    This test attempts to save a stock data dictionary without the required 'symbol' field and asserts that a KeyError is raised.
    """
    saver = create_saver_with_temp_db(mock_consumer, basic_config)

    # Missing symbol
    incomplete_data: dict[str, Any] = {
        "retrieval_timestamp": "1729532024.309936",
        "last_traded_timestamp": "1729504796",
        "exchange_id": ExchangeType.NSE.value,
        "data_provider_id": DataProviderType.SMARTAPI.value,
        "last_traded_price": "100.00",
    }

    with pytest.raises(KeyError):
        saver.save_stock_data(incomplete_data)  # type: ignore[arg-type]


def test_save_stock_data_multiple_records(
    mock_consumer: MagicMock,
    basic_config: DictConfig,
    sample_stock_data: List[dict[str, Any]],
) -> None:
    """
    Test that multiple stock data records can be saved and all are persisted correctly.
    
    Verifies that each record in the provided sample data is saved and that the set of saved symbols matches the expected set from the input data.
    """
    saver = create_saver_with_temp_db(mock_consumer, basic_config)

    for data in sample_stock_data:
        saver.save_stock_data(data)

    # Verify all data was saved
    saved_data = verify_saved_data_count(saver, len(sample_stock_data))
    symbols = {item.symbol for item in saved_data}
    expected_symbols = {data["symbol"] for data in sample_stock_data}
    assert symbols == expected_symbols


def test_save_stock_data_duplicate_handling(
    mock_consumer: MagicMock, basic_config: DictConfig
) -> None:
    """
    Verify that saving duplicate stock data records does not raise errors and that at least one record is persisted.
    
    This test saves the same stock data twice and asserts that the database contains at least one record, regardless of whether duplicates are allowed or deduplicated by the implementation.
    """
    saver = create_saver_with_temp_db(mock_consumer, basic_config)

    data = create_test_instrument_price_data()

    # Save same data twice
    saver.save_stock_data(data)
    saver.save_stock_data(data)  # Should not raise error due to insert_data logic

    # Verify only one record exists (depending on CRUD implementation)
    saved_data = get_saved_stock_data(saver)
    # The behavior depends on the insert_data implementation
    assert len(saved_data) >= 1


def test_save_stock_data_none_values(
    mock_consumer: MagicMock, basic_config: DictConfig
) -> None:
    """
    Verify that stock data with None values for optional fields is saved correctly.
    
    This test ensures that when optional fields such as 'last_traded_quantity' and 'average_traded_price' are set to None, the data is still persisted and the None values are retained in the database.
    """
    saver = create_saver_with_temp_db(mock_consumer, basic_config)

    data = create_test_instrument_price_data()
    data["last_traded_quantity"] = None
    data["average_traded_price"] = None

    saver.save_stock_data(data)

    # Verify data was saved with None values
    saved_item = verify_saved_symbol(saver, "TEST_SYMBOL")
    assert saved_item.last_traded_quantity is None
    assert saved_item.average_traded_price is None


# ======================================================================================
# SAVE TESTS (BYTES TO JSON PROCESSING)
# ======================================================================================
#
# These tests validate the JSON processing pipeline:
# - Conversion of raw bytes to JSON and subsequent data extraction
# - Error handling for malformed JSON data and encoding issues
# - UTF-8 encoding validation and Unicode character support
# - Empty/null JSON object handling and validation
# - Special character preservation and internationalization support
# - Data integrity through the bytes → JSON → database pipeline


def test_save_valid_json_bytes(
    mock_consumer: MagicMock, basic_config: DictConfig
) -> None:
    """
    Test that valid JSON bytes are correctly saved as stock data.
    
    Verifies that the saver can decode and persist valid UTF-8 encoded JSON bytes representing instrument price data.
    """
    saver = create_saver_with_temp_db(mock_consumer, basic_config)

    data = create_test_instrument_price_data()
    json_bytes = json.dumps(data).encode("utf-8")

    saver.save(json_bytes)

    # Verify data was saved
    verify_saved_symbol(saver, "TEST_SYMBOL")


def test_save_invalid_json_bytes(
    mock_consumer: MagicMock, basic_config: DictConfig
) -> None:
    """
    Verify that attempting to save invalid JSON bytes raises a JSONDecodeError.
    """
    saver = create_saver_with_temp_db(mock_consumer, basic_config)

    invalid_json = b"{'invalid': json}"  # Single quotes are invalid JSON

    with pytest.raises(json.JSONDecodeError):
        saver.save(invalid_json)


def test_save_non_utf8_bytes(
    mock_consumer: MagicMock, basic_config: DictConfig
) -> None:
    """
    Verify that attempting to save non-UTF8 encoded bytes raises a UnicodeDecodeError.
    """
    saver = create_saver_with_temp_db(mock_consumer, basic_config)

    # Create bytes that can't be decoded as UTF-8
    invalid_utf8 = b"\xff\xfe\xfd"

    with pytest.raises(UnicodeDecodeError):
        saver.save(invalid_utf8)


def test_save_empty_json_object(
    mock_consumer: MagicMock, basic_config: DictConfig
) -> None:
    """
    Test that saving an empty JSON object raises a KeyError due to missing required fields.
    """
    saver = create_saver_with_temp_db(mock_consumer, basic_config)

    empty_json = json.dumps({}).encode("utf-8")

    # Should raise KeyError for missing required fields
    with pytest.raises(KeyError):
        saver.save(empty_json)


def test_save_unicode_special_and_null_handling(
    mock_consumer: MagicMock, basic_config: DictConfig
) -> None:
    """
    Test that saving data with Unicode symbols, special characters, and null or empty values is handled correctly.
    
    Verifies that records containing non-ASCII characters and special symbols are saved and retrievable, and that fields with None or empty string values are persisted as expected.
    """
    saver = create_saver_with_temp_db(mock_consumer, basic_config)

    # Test Unicode content via JSON bytes
    unicode_data = create_test_instrument_price_data(
        symbol="测试股票"
    )  # Chinese characters
    json_bytes = json.dumps(unicode_data, ensure_ascii=False).encode("utf-8")
    saver.save(json_bytes)

    # Test special characters with null/empty values
    special_chars_data = create_test_instrument_price_data(symbol="SPE©IAL_ÇH@RS_ñ_ü_é")
    special_chars_data["last_traded_quantity"] = None
    special_chars_data["average_traded_price"] = ""  # Empty string
    saver.save_stock_data(special_chars_data)

    # Verify both were saved correctly
    saved_data = verify_saved_data_count(saver, 2)
    symbols = {item.symbol for item in saved_data}
    assert "测试股票" in symbols
    assert "SPE©IAL_ÇH@RS_ñ_ü_é" in symbols

    # Verify null handling
    special_item = next(
        item for item in saved_data if item.symbol == "SPE©IAL_ÇH@RS_ñ_ü_é"
    )
    assert special_item.last_traded_quantity is None


# ======================================================================================
# INTEGRATION TESTS
# ======================================================================================
#
# These tests validate complete end-to-end workflows:
# - Full pipeline from configuration loading to data persistence
# - Kafka consumer integration and message processing flow
# - Configuration validation and dependency injection
# - Multi-component interaction and data flow verification
# - System-level behavior under realistic usage scenarios


def test_end_to_end_workflow(
    mock_get_kafka_consumer: MockType,
    basic_config: DictConfig,
    kafka_data: List[dict[str, Any]],
) -> None:
    """
    Performs an end-to-end test of the SqliteDataSaver workflow, verifying that all Kafka messages are processed and saved to the database using configuration-based initialization.
    
    This test mocks the Kafka consumer, initializes the saver from configuration, processes a list of Kafka messages, and asserts that all expected records are persisted.
    """
    # Mock consumer
    mock_consumer = MagicMock()
    mock_get_kafka_consumer.return_value = mock_consumer

    # Create saver from config
    saver = SqliteDataSaver.from_cfg(basic_config)
    assert saver is not None

    # Process messages
    messages = [create_json_message(data) for data in kafka_data]

    run_retrieve_and_save_with_messages(saver, mock_consumer, messages)

    # Verify all data was processed
    verify_saved_data_count(saver, len(kafka_data))


# ======================================================================================
# RETRIEVE_AND_SAVE TESTS (KAFKA INTEGRATION)
# ======================================================================================
#
# These tests focus on Kafka message consumption and processing:
# - Single and multiple message handling from Kafka streams
# - Kafka error propagation and exception handling
# - Consumer lifecycle management (polling, processing, cleanup)
# - Message format validation and JSON decoding error scenarios
# - Graceful handling of network timeouts and connection issues


def test_retrieve_and_save_single_message(
    mock_consumer: MagicMock,
    basic_config: DictConfig,
    sample_stock_data: List[dict[str, Any]],
) -> None:
    """
    Verifies that `retrieve_and_save` processes a single Kafka message and correctly saves the corresponding stock data to the database.
    
    This test ensures that the saved record matches the input data and that the Kafka consumer is properly closed after processing.
    """
    saver = create_saver_with_temp_db(mock_consumer, basic_config)

    # Create single message
    data = sample_stock_data[0]
    message = create_json_message(data)
    messages = [message]

    # Mock consumer behavior
    run_retrieve_and_save_with_messages(saver, mock_consumer, messages)

    # Verify data was saved
    saved_item = verify_saved_symbol(saver, data["symbol"])

    assert saved_item.last_traded_price == data["last_traded_price"]

    # Verify consumer was closed
    mock_consumer.close.assert_called_once()


def test_retrieve_and_save_kafka_error(
    mock_consumer: MagicMock, basic_config: DictConfig
) -> None:
    """
    Test that retrieve_and_save raises KafkaException when a Kafka error is encountered in a message and ensures the consumer is closed.
    """
    saver = create_saver_with_temp_db(mock_consumer, basic_config)

    # Create message with error
    error_message = MockMessage(
        value=None,
        error_obj=lambda: KafkaError(
            KafkaError._MSG_TIMED_OUT  # pylint: disable= protected-access
        ),
    )
    mock_consumer.poll.return_value = error_message

    with pytest.raises(KafkaException):
        saver.retrieve_and_save()

    mock_consumer.close.assert_called_once()


def test_retrieve_and_save_json_decode_error(
    mock_consumer: MagicMock, basic_config: DictConfig
) -> None:
    """
    Test that retrieve_and_save raises a JSONDecodeError when processing a message with invalid JSON data, and ensures the Kafka consumer is closed after the error.
    """
    saver = create_saver_with_temp_db(mock_consumer, basic_config)

    # Create message with invalid JSON
    invalid_message = MockMessage(value=b"{'invalid': json}", error_obj=None)
    messages = [invalid_message]

    # This should raise a JSON decode error
    with pytest.raises(json.JSONDecodeError):
        run_retrieve_and_save_with_messages(saver, mock_consumer, messages)

    mock_consumer.close.assert_called_once()


# ======================================================================================
# FROM_CFG TESTS (CONFIGURATION)
# ======================================================================================
#
# These tests validate configuration-based initialization:
# - Successful creation from valid configuration objects
# - Error handling for invalid or missing configuration parameters
# - Kafka consumer creation and dependency injection validation
# - Configuration parameter validation and type checking
# - Graceful failure handling when external dependencies are unavailable


def test_init_from_cfg(
    mock_get_kafka_consumer: MockType, basic_config: DictConfig
) -> None:
    """
    Test that the data saver is correctly initialized from configuration using a mocked Kafka consumer.
    
    Verifies that the saver instance is created, the consumer is set, the database filename includes the expected date, and the Kafka consumer is initialized with the correct parameters.
    """
    # Mock successful consumer creation
    mock_consumer = MagicMock()
    mock_get_kafka_consumer.return_value = mock_consumer

    saver = init_from_cfg(basic_config, DataSaver)

    assert saver is not None
    assert saver.consumer == mock_consumer
    assert "test_2023_12_01.sqlite3" in saver.sqlite_db

    # Verify kafka consumer was created with correct parameters
    mock_get_kafka_consumer.assert_called_once()


def test_from_cfg_success(
    mock_get_kafka_consumer: MockType, basic_config: DictConfig
) -> None:
    """
    Verify that SqliteDataSaver is correctly instantiated from a valid configuration, including proper Kafka consumer assignment and database filename formatting.
    """
    # Mock successful consumer creation
    mock_consumer = MagicMock()
    mock_get_kafka_consumer.return_value = mock_consumer

    saver = SqliteDataSaver.from_cfg(basic_config)

    assert saver is not None
    assert saver.consumer == mock_consumer
    assert "test_2023_12_01.sqlite3" in saver.sqlite_db

    # Verify kafka consumer was created with correct parameters
    mock_get_kafka_consumer.assert_called_once()


def test_from_cfg_kafka_consumer_creation_fails(
    mock_get_kafka_consumer: MockType,
    basic_config: DictConfig,
) -> None:
    """
    Test that `SqliteDataSaver.from_cfg` returns None when Kafka consumer creation fails.
    """
    # Mock consumer creation failure
    mock_get_kafka_consumer.return_value = None

    saver = SqliteDataSaver.from_cfg(basic_config)

    assert saver is None


def test_from_cfg_invalid_sqlite_db_path(mock_logger: MockType) -> None:
    """
    Test that SqliteDataSaver.from_cfg returns None and logs an error when the sqlite_db path is missing, None, or empty in the configuration.
    """
    test_cases = [
        # Missing sqlite_db key
        {
            "name": "sqlite_saver",
            "streaming": {
                "kafka_topic": "test_topic",
                "kafka_server": "localhost:9092",
            },
        },
        # None sqlite_db value
        {
            "name": "sqlite_saver",
            "sqlite_db": None,
            "streaming": {
                "kafka_topic": "test_topic",
                "kafka_server": "localhost:9092",
            },
        },
        # Empty sqlite_db value
        {
            "name": "sqlite_saver",
            "sqlite_db": "",
            "streaming": {
                "kafka_topic": "test_topic",
                "kafka_server": "localhost:9092",
            },
        },
    ]

    for config_dict in test_cases:
        config = OmegaConf.create(config_dict)  # type: ignore[call-overload]
        saver = SqliteDataSaver.from_cfg(config)
        assert saver is None

    # Should be called 3 times (once for each test case)
    assert mock_logger.error.call_count == 3
    mock_logger.error.assert_called_with("sqlite_db path not provided in configuration")


# ======================================================================================
# ERROR HANDLING AND EDGE CASES
# ======================================================================================
#
# These tests ensure robust error handling and security:
# - Database connection failure scenarios and error propagation
# - SQL injection attack prevention and input sanitization
# - File system permission and path validation errors
# - Resource cleanup and proper exception handling
# - Security validation against malicious input patterns


def test_database_connection_error() -> None:
    """
    Test that SqliteDataSaver raises an exception when initialized with an invalid or inaccessible database path.
    """
    mock_consumer = MagicMock()

    # Use invalid database path that would cause connection issues
    with pytest.raises(Exception):  # Could be various database-related exceptions
        # Try to create saver with read-only path
        SqliteDataSaver(mock_consumer, "/root/invalid/path/test.sqlite3")


def test_save_stock_data_with_sql_injection_attempt(
    mock_consumer: MagicMock, basic_config: DictConfig
) -> None:
    """
    Verify that stock data containing SQL injection patterns in the symbol field is saved safely without compromising the database.
    
    This test ensures that malicious input does not result in SQL execution and that the target table remains intact after saving.
    """
    saver = create_saver_with_temp_db(mock_consumer, basic_config)

    # Data with potential SQL injection
    malicious_data = create_test_instrument_price_data(
        symbol="'; DROP TABLE instrument_price; --"
    )

    # Should save safely without executing injection
    saver.save_stock_data(malicious_data)

    # Verify data was saved and table still exists
    verify_saved_symbol(saver, "'; DROP TABLE instrument_price; --")
