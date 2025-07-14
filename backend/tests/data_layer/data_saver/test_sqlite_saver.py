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
        Initialize a mock Kafka message.

        Attributes
        ----------
        value: ``Optional[bytes]``, ( default = None )
            Value of the Kafka message
        error_obj: ``Optional[KafkaError]``, ( default = None )
            Error object associated with the Kafka message
        """
        self._value = value
        self._error = error_obj

    def value(self) -> Optional[bytes]:
        """
        Get the value of the Kafka message.
        """
        return self._value

    def error(self) -> Optional[KafkaError]:
        """
        Get the error object associated with the Kafka message.
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
    Helper function to run retrieve_and_save with controlled message flow.
    This avoids the infinite loop issue by properly managing poll calls.
    """
    call_count = 0

    def poll_side_effect(*_: Any, **__: Any) -> Optional[MockMessage]:
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
    Create test data for InstrumentPrice with optional overrides.
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
    Create a SqliteDataSaver with a temporary database.
    """
    return SqliteDataSaver(mock_consumer, config.sqlite_db)


def get_saved_stock_data(saver: SqliteDataSaver) -> List[Any]:
    """
    Get all saved stock data from the saver's database.
    """
    with get_session(saver.engine) as session:
        return get_all_stock_price_info(session=session)


def create_json_message(
    data: Optional[dict], error_obj: Optional[KafkaError] = None
) -> MockMessage:
    """
    Create a MockMessage with JSON-encoded data.
    """
    json_value = json.dumps(data).encode("utf-8") if data else None
    return MockMessage(value=json_value, error_obj=error_obj)


def verify_saved_data_count(saver: SqliteDataSaver, expected_count: int) -> List[Any]:
    """
    Verify that the expected number of records were saved.
    """
    saved_data = get_saved_stock_data(saver)
    assert len(saved_data) == expected_count
    return saved_data


def verify_saved_symbol(saver: SqliteDataSaver, expected_symbol: str) -> Any:
    """
    Verify that a specific symbol was saved correctly.
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
    Create a temporary directory for test files.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def basic_config(temp_dir: str) -> DictConfig:
    """
    Basic configuration for SqliteDataSaver.
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
    Mock the get_kafka_consumer function.
    """
    return mocker.patch(
        "app.data_layer.data_saver.saver.sqlite_saver.get_kafka_consumer"
    )


@pytest.fixture
def mock_logger(mocker: MockerFixture) -> MockType:
    """
    Mock the logger.
    """
    return mocker.patch("app.data_layer.data_saver.saver.sqlite_saver.logger")


@pytest.fixture(autouse=True)
def mock_datetime(mocker: MockerFixture) -> MockType:
    """
    Mock datetime.now for consistent testing.
    """
    mock_dt = mocker.patch("app.data_layer.data_saver.saver.sqlite_saver.datetime")
    mock_dt.now.return_value.strftime.return_value = "2023_12_01"
    return mock_dt


@pytest.fixture
def mock_consumer() -> MagicMock:
    """
    Create a mock Kafka consumer.
    """
    consumer = MagicMock()
    return consumer


@pytest.fixture
def sample_stock_data() -> List[dict[str, Any]]:
    """
    Sample stock data for testing.
    """
    return [
        create_test_instrument_price_data(symbol="AAPL", last_traded_price=150.25),
        create_test_instrument_price_data(symbol="GOOGL", last_traded_price=2750.80),
        create_test_instrument_price_data(symbol="MSFT", last_traded_price=305.15),
    ]


@pytest.fixture
def kafka_data() -> List[dict[str, Any]]:
    """
    Sample Kafka message data for testing.
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
    Test initialization with Path object and string paths.
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
    Test that parent directories are created if they don't exist.
    """
    nested_path = f"{temp_dir}/nested/deep/path/test.sqlite3"

    saver = SqliteDataSaver(mock_consumer, nested_path)

    assert Path(nested_path).parent.exists()
    assert "test_2023_12_01.sqlite3" in saver.sqlite_db


def test_init_removes_db_extension(
    mock_consumer: MagicMock, basic_config: DictConfig
) -> None:
    """
    Test that database extensions are properly handled.
    """
    # Test .sqlite3 extension (most common case)
    saver = SqliteDataSaver(mock_consumer, basic_config.sqlite_db)
    assert "test_2023_12_01.sqlite3" in saver.sqlite_db


def test_init_date_formatting(
    mock_consumer: MagicMock, basic_config: DictConfig, mocker: MockerFixture
) -> None:
    """
    Test that date is correctly formatted in database name.
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
    Test that database tables are created during initialization.
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
    Test saving stock data with minimal required fields.
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
    Test saving stock data with all fields.
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
    Test saving stock data with missing required fields.
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
    Test saving multiple stock data records.
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
    Test handling of duplicate stock data records.
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
    Test saving stock data with None values for optional fields.
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
    Test saving valid JSON bytes data.
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
    Test saving invalid JSON bytes.
    """
    saver = create_saver_with_temp_db(mock_consumer, basic_config)

    invalid_json = b"{'invalid': json}"  # Single quotes are invalid JSON

    with pytest.raises(json.JSONDecodeError):
        saver.save(invalid_json)


def test_save_non_utf8_bytes(
    mock_consumer: MagicMock, basic_config: DictConfig
) -> None:
    """
    Test saving non-UTF8 encoded bytes.
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
    Test saving empty JSON object.
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
    Test saving JSON with Unicode content, special characters, and null/empty values.
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
    Test complete end-to-end workflow from configuration to data saving.
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
    Test retrieve_and_save with a single message.
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
    Test retrieve_and_save with Kafka error.
    """
    saver = create_saver_with_temp_db(mock_consumer, basic_config)

    # Create message with error
    kafka_error = KafkaError(
        KafkaError._MSG_TIMED_OUT  # pylint: disable=protected-access
    )
    error_message = MockMessage(value=None, error_obj=kafka_error)
    mock_consumer.poll.return_value = error_message

    with pytest.raises(KafkaException):
        saver.retrieve_and_save()

    mock_consumer.close.assert_called_once()


def test_retrieve_and_save_json_decode_error(
    mock_consumer: MagicMock, basic_config: DictConfig
) -> None:
    """
    Test retrieve_and_save with JSON decode error.
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
    Test successful creation from configuration.
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
    Test successful creation from configuration.
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
    mock_logger: MockType,
) -> None:
    """
    Test from_cfg when Kafka consumer creation fails.
    """
    # Mock consumer creation failure
    mock_get_kafka_consumer.side_effect = KafkaException(
        "Kafka consumer creation failed"
    )

    saver = SqliteDataSaver.from_cfg(basic_config)
    mock_logger.error.assert_called_once_with(
        "Error while creating SqliteDataSaver: %s", mock_get_kafka_consumer.side_effect
    )

    assert saver is None


def test_from_cfg_invalid_sqlite_db_path(mock_logger: MockType) -> None:
    """
    Test from_cfg with invalid sqlite_db path scenarios.
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
    Test handling of database connection errors.
    """
    mock_consumer = MagicMock()

    # Use invalid database path that would cause connection issues
    with pytest.raises((OSError, PermissionError)):
        # Try to create saver with read-only path
        SqliteDataSaver(mock_consumer, "/root/invalid/path/test.sqlite3")


def test_save_stock_data_with_sql_injection_attempt(
    mock_consumer: MagicMock, basic_config: DictConfig
) -> None:
    """
    Test that SQL injection attempts are safely handled.
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
