"""
Comprehensive tests for save_data module using method-based approach.
Tests cover configuration handling, data saver creation, threading, error handling, and integration scenarios.
Follows minimal mocking strategy - only mocks external, computational, or memory-intensive components.
"""

import threading
import time
from contextlib import contextmanager
from typing import Any, Generator
from unittest.mock import MagicMock, call, patch

import pytest
from omegaconf import DictConfig, OmegaConf
from pytest_mock import MockerFixture, MockType

from app.data_layer.data_saver.data_saver import DataSaver
from app.data_layer.data_saver.save_data import main

# ======================================================================================
# MOCK DATA SAVER CLASSES
# ======================================================================================


class MockDataSaver(DataSaver):
    """
    Mock DataSaver for testing purposes.
    """

    def __init__(
        self, name: str, should_fail: bool = False, execution_time: float = 0.1
    ):
        """
        Initialize MockDataSaver instance.

        Parameters
        -----------
        name: ``str``
            Name of the mock data saver
        should_fail: ``bool``
            Whether the saver should fail during execution, defaults to False
        execution_time: ``float``
            Simulated execution time in seconds, defaults to 0.1
        """
        self.name = name
        self.should_fail = should_fail
        self.execution_time = execution_time
        self.retrieve_and_save_called = False
        self.thread_id = None

    def retrieve_and_save(self):
        """
        Mock implementation of retrieve_and_save.

        Simulates work execution and tracks thread information.
        Raises RuntimeError if should_fail is True.
        """
        self.retrieve_and_save_called = True
        self.thread_id = threading.current_thread().ident

        # Simulate some work
        time.sleep(self.execution_time)

        if self.should_fail:
            raise RuntimeError(f"Mock failure in {self.name}")

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "MockDataSaver":
        """
        Mock implementation of from_cfg.

        Parameters
        -----------
        cfg: ``DictConfig``
            Configuration object containing saver settings

        Returns
        --------
        ``MockDataSaver``
            New MockDataSaver instance created from configuration
        """
        return cls(
            name=cfg.get("name", "unnamed"),
            should_fail=cfg.get("should_fail", False),
            execution_time=cfg.get("execution_time", 0.1),
        )


# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================


def create_config_with_savers(*saver_configs) -> DictConfig:
    """
    Create a configuration with specified data savers.

    Parameters
    -----------
    *saver_configs: ``dict[str, Any]``
        Variable number of saver configuration dictionaries

    Returns
    --------
    ``DictConfig``
        OmegaConf configuration object with data_saver list
    """
    return OmegaConf.create({"data_saver": list(saver_configs)})


def create_saver_config(
    name: str, saver_type: str = "mock", **kwargs
) -> dict[str, Any]:
    """
    Create a single data saver configuration.

    Parameters
    -----------
    name: ``str``
        Name of the data saver
    saver_type: ``str``
        Type of the data saver, defaults to "mock"
    **kwargs: ``Any``
        Additional configuration parameters

    Returns
    --------
    ``dict[str, Any]``
        Dictionary containing saver configuration with name as key
    """
    config = {"name": name, "type": saver_type, **kwargs}
    return {name: config}


def setup_multi_saver_mocks(
    mock_init_from_cfg: MockType, mock_thread: MockType, saver_names: list[str]
) -> tuple[list[MockDataSaver], list[MagicMock]]:
    """
    Set up mocks for multiple savers with consistent pattern.

    Parameters
    -----------
    mock_init_from_cfg: ``MockType``
        Mock for init_from_cfg function
    mock_thread: ``MockType``
        Mock for Thread class
    saver_names: ``list[str]``
        List of saver names to create mocks for

    Returns
    --------
    mock_savers: ``list[MockDataSaver]``
        List of mock data saver instances
    mock_thread_instances: ``list[MagicMock]``
        List of mock thread instances
    """
    mock_savers = [MockDataSaver(name) for name in saver_names]
    mock_init_from_cfg.side_effect = mock_savers

    mock_thread_instances = [MagicMock() for _ in range(len(saver_names))]
    mock_thread.side_effect = mock_thread_instances

    return mock_savers, mock_thread_instances


def setup_single_saver_mock(
    mock_init_from_cfg: MockType, mock_thread: MockType, saver_name: str
) -> tuple[MockDataSaver, MagicMock]:
    """
    Set up mocks for a single saver.

    Parameters
    -----------
    mock_init_from_cfg: ``MockType``
        Mock for init_from_cfg function
    mock_thread: ``MockType``
        Mock for Thread class
    saver_name: ``str``
        Name of the saver to create mock for

    Returns
    --------
    mock_saver: ``MockDataSaver``
        Mock data saver instance
    mock_thread_instance: ``MagicMock``
        Mock thread instance
    """
    mock_saver = MockDataSaver(saver_name)
    mock_init_from_cfg.return_value = mock_saver

    mock_thread_instance = MagicMock()
    mock_thread.return_value = mock_thread_instance

    return mock_saver, mock_thread_instance


def verify_thread_operations(mock_thread_instances: list[MagicMock]) -> None:
    """
    Verify that all thread instances were started and joined.

    Parameters
    -----------
    mock_thread_instances: ``list[MagicMock]``
        List of mock thread instances to verify
    """
    for thread_instance in mock_thread_instances:
        thread_instance.start.assert_called_once()
        thread_instance.join.assert_called_once()


def verify_info_logs_for_savers(mock_logger: MockType, saver_names: list[str]) -> None:
    """
    Verify that info logs were created for starting savers.

    Parameters
    -----------
    mock_logger: ``MockType``
        Mock logger instance
    saver_names: ``list[str]``
        List of saver names that should have been logged
    """
    expected_calls = [call("Starting the saver %s", name) for name in saver_names]
    mock_logger.info.assert_has_calls(expected_calls, any_order=True)


def verify_error_logs_for_unregistered_savers(
    mock_logger: MockType, saver_names: list[str]
) -> None:
    """
    Verify that error logs were created for unregistered savers.

    Parameters
    -----------
    mock_logger: ``MockType``
        Mock logger instance
    saver_names: ``list[str]``
        List of saver names that should have error logs
    """
    expected_error_calls = [
        call("Data saver %s is not registered", name) for name in saver_names
    ]
    mock_logger.error.assert_has_calls(expected_error_calls, any_order=True)


def verify_thread_creation_for_savers(
    mock_thread: MockType, mock_savers: list[MockDataSaver]
) -> None:
    """
    Verify that threads were created with correct targets for each saver.

    Parameters
    -----------
    mock_thread: ``MockType``
        Mock Thread class
    mock_savers: ``list[MockDataSaver]``
        List of mock savers to verify thread creation for
    """
    expected_calls = [call(target=saver.retrieve_and_save) for saver in mock_savers]
    mock_thread.assert_has_calls(expected_calls, any_order=True)


@contextmanager
def patch_save_data_dependencies() -> (
    Generator[tuple[MockType, MockType, MockType], None, None]
):
    """
    Context manager to patch common save_data module dependencies.

    Yields
    --------
    tuple[MockType, MockType, MockType]
        Tuple of (mock_init, mock_thread, mock_logger)
    """
    with patch("app.data_layer.data_saver.save_data.init_from_cfg") as mock_init:
        with patch("app.data_layer.data_saver.save_data.Thread") as mock_thread:
            with patch("app.data_layer.data_saver.save_data.logger") as mock_logger:
                yield mock_init, mock_thread, mock_logger


def setup_mixed_success_failure_mocks(
    mock_init_from_cfg: MockType,
    mock_thread: MockType,
    success_names: list[str],
    failure_index: int,
) -> tuple[list[MockDataSaver | None], list[MockDataSaver], list[MagicMock]]:
    """
    Set up mocks for mixed success/failure scenario.

    Parameters
    -----------
    mock_init_from_cfg: ``MockType``
        Mock for init_from_cfg function
    mock_thread: ``MockType``
        Mock for Thread class
    success_names: ``list[str]``
        List of successful saver names
    failure_index: ``int``
        Index where failure should occur

    Returns
    --------
    mock_savers_with_none: ``list[MockDataSaver | None]``
        List of mock savers with None at failure index
    successful_mock_savers: ``list[MockDataSaver]``
        List of successful mock savers only
    mock_thread_instances: ``list[MagicMock]``
        List of mock thread instances for successful savers
    """
    # Create mock savers with None at failure index
    mock_savers_with_none: list[MockDataSaver | None] = []
    successful_mock_savers: list[MockDataSaver] = []

    for i, name in enumerate(success_names):
        if i == failure_index:
            mock_savers_with_none.append(None)
        else:
            mock_saver = MockDataSaver(name)
            mock_savers_with_none.append(mock_saver)
            successful_mock_savers.append(mock_saver)

    mock_init_from_cfg.side_effect = mock_savers_with_none

    # Create thread instances only for successful savers
    mock_thread_instances = [MagicMock() for _ in range(len(successful_mock_savers))]
    mock_thread.side_effect = mock_thread_instances

    return mock_savers_with_none, successful_mock_savers, mock_thread_instances


# ======================================================================================
# FIXTURES
# ======================================================================================


@pytest.fixture
def mock_logger(mocker: MockerFixture) -> MockType:
    """
    Mock the logger.
    """
    return mocker.patch("app.data_layer.data_saver.save_data.logger")


@pytest.fixture
def mock_init_from_cfg(mocker: MockerFixture) -> MockType:
    """
    Mock the init_from_cfg function.
    """
    return mocker.patch("app.data_layer.data_saver.save_data.init_from_cfg")


@pytest.fixture
def mock_thread(mocker: MockerFixture) -> MockType:
    """
    Mock the Thread class.
    """
    return mocker.patch("app.data_layer.data_saver.save_data.Thread")


@pytest.fixture
def basic_single_saver_config() -> DictConfig:
    """
    Basic configuration with a single data saver.
    """
    return create_config_with_savers(create_saver_config("test_saver"))


@pytest.fixture
def multi_saver_config() -> DictConfig:
    """
    Configuration with multiple data savers.
    """
    return create_config_with_savers(
        create_saver_config("csv_saver"),
        create_saver_config("sqlite_saver"),
        create_saver_config("jsonl_saver"),
    )


@pytest.fixture
def empty_config() -> DictConfig:
    """
    Configuration with no data savers.

    Returns
    --------
    config: ``DictConfig``
        Configuration object with empty data_saver list
    """
    return OmegaConf.create({"data_saver": []})


# ======================================================================================
# CONFIGURATION PROCESSING TESTS
# ======================================================================================


def test_main_processes_single_saver_config(
    mock_init_from_cfg: MockType,
    mock_thread: MockType,
    mock_logger: MockType,
    basic_single_saver_config: DictConfig,
) -> None:
    """
    Test that main correctly processes a single data saver configuration.
    """
    mock_saver, mock_thread_instance = setup_single_saver_mock(
        mock_init_from_cfg, mock_thread, "test_saver"
    )

    main(basic_single_saver_config)

    # Verify init_from_cfg was called with correct arguments
    mock_init_from_cfg.assert_called_once()
    call_args = mock_init_from_cfg.call_args
    config, data_saver_class = call_args[0]
    assert config["name"] == "test_saver"
    assert data_saver_class == DataSaver

    # Verify thread creation and management
    mock_thread.assert_called_once_with(target=mock_saver.retrieve_and_save)
    verify_thread_operations([mock_thread_instance])

    mock_logger.info.assert_called_once_with("Starting the saver %s", "test_saver")


def test_main_processes_multiple_saver_config(
    mock_init_from_cfg: MockType,
    mock_thread: MockType,
    mock_logger: MockType,
    multi_saver_config: DictConfig,
) -> None:
    """
    Test that main correctly processes multiple data saver configurations.
    """
    saver_names = ["csv_saver", "sqlite_saver", "jsonl_saver"]
    mock_savers, mock_thread_instances = setup_multi_saver_mocks(
        mock_init_from_cfg, mock_thread, saver_names
    )

    main(multi_saver_config)

    # Verify init_from_cfg was called for each saver
    assert mock_init_from_cfg.call_count == 3

    # Verify thread creation for each saver
    assert mock_thread.call_count == 3
    verify_thread_creation_for_savers(mock_thread, mock_savers)

    # Verify all threads were started and joined
    verify_thread_operations(mock_thread_instances)

    assert mock_logger.info.call_count == 3


def test_main_handles_empty_config(
    mock_init_from_cfg: MockType,
    mock_thread: MockType,
    mock_logger: MockType,
    empty_config: DictConfig,
) -> None:
    """
    Test that main handles empty data saver configuration.
    """
    main(empty_config)

    # Verify no savers were created
    mock_init_from_cfg.assert_not_called()
    mock_thread.assert_not_called()
    mock_logger.info.assert_not_called()


def test_main_processes_config_with_nested_structure(
    mock_init_from_cfg: MockType, mock_thread: MockType
) -> None:
    """
    Test that main correctly extracts saver name and config from nested structure.
    """
    complex_config = create_config_with_savers(
        {
            "csv_saver": {
                "name": "csv_saver",
                "csv_file_path": "/tmp/test.csv",
                "streaming": {"kafka_topic": "test"},
            }
        }
    )

    setup_single_saver_mock(mock_init_from_cfg, mock_thread, "csv_saver")

    main(complex_config)

    # Verify the correct config was extracted and passed
    call_args = mock_init_from_cfg.call_args[0]
    extracted_config = call_args[0]
    assert extracted_config["name"] == "csv_saver"
    assert extracted_config["csv_file_path"] == "/tmp/test.csv"


# ======================================================================================
# DATA SAVER CREATION TESTS
# ======================================================================================


def test_main_handles_saver_creation_failure(
    mock_init_from_cfg: MockType,
    mock_thread: MockType,
    mock_logger: MockType,
    basic_single_saver_config: DictConfig,
) -> None:
    """
    Test handling of data saver creation failure.
    """
    mock_init_from_cfg.return_value = None  # Simulate creation failure

    main(basic_single_saver_config)

    # Verify error was logged
    mock_logger.error.assert_called_once_with(
        "Data saver %s is not registered", "test_saver"
    )

    # Verify no thread was created
    mock_thread.assert_not_called()


def test_main_handles_mixed_creation_success_failure(
    mock_init_from_cfg: MockType,
    mock_thread: MockType,
    mock_logger: MockType,
    multi_saver_config: DictConfig,
) -> None:
    """
    Test handling of mixed success and failure in saver creation.
    """
    saver_names = ["csv_saver", "sqlite_saver", "jsonl_saver"]
    (_, successful_mock_savers, _) = setup_mixed_success_failure_mocks(
        mock_init_from_cfg, mock_thread, saver_names, failure_index=1
    )

    main(multi_saver_config)

    # Verify error was logged for failed saver
    mock_logger.error.assert_called_once_with(
        "Data saver %s is not registered", "sqlite_saver"
    )

    # Verify only successful savers had threads created
    assert mock_thread.call_count == 2
    mock_thread.assert_any_call(target=successful_mock_savers[0].retrieve_and_save)
    mock_thread.assert_any_call(target=successful_mock_savers[1].retrieve_and_save)


def test_main_handles_all_savers_creation_failure(
    mock_init_from_cfg: MockType,
    mock_thread: MockType,
    mock_logger: MockType,
    multi_saver_config: DictConfig,
) -> None:
    """
    Test handling when all data saver creations fail.
    """
    mock_init_from_cfg.return_value = None

    main(multi_saver_config)

    # Verify errors were logged for all savers
    assert mock_logger.error.call_count == 3
    verify_error_logs_for_unregistered_savers(
        mock_logger, ["csv_saver", "sqlite_saver", "jsonl_saver"]
    )

    # Verify no threads were created
    mock_thread.assert_not_called()


# ======================================================================================
# THREADING TESTS
# ======================================================================================


def test_main_creates_separate_threads_for_each_saver(
    mock_init_from_cfg: MockType, mock_thread: MockType, multi_saver_config: DictConfig
) -> None:
    """
    Test that separate threads are created for each data saver.
    """
    saver_names = ["saver_0", "saver_1", "saver_2"]
    mock_savers, _ = setup_multi_saver_mocks(
        mock_init_from_cfg, mock_thread, saver_names
    )

    main(multi_saver_config)

    # Verify correct number of threads created
    assert mock_thread.call_count == 3

    # Verify each thread was created with correct target
    verify_thread_creation_for_savers(mock_thread, mock_savers)


# ======================================================================================
# ERROR HANDLING TESTS
# ======================================================================================


def test_main_handles_init_from_cfg_exception(
    mock_init_from_cfg: MockType,
    mock_thread: MockType,
    mock_logger: MockType,
    basic_single_saver_config: DictConfig,
) -> None:
    """
    Test handling of exceptions during saver initialization.
    """
    mock_init_from_cfg.side_effect = RuntimeError("Configuration error")

    # Should not crash, but propagate the exception
    with pytest.raises(RuntimeError, match="Configuration error"):
        main(basic_single_saver_config)

    # Verify no threads were created
    mock_thread.assert_not_called()
    mock_logger.info.assert_not_called()


def test_main_continues_after_partial_failures(
    mock_init_from_cfg: MockType, mock_thread: MockType, mock_logger: MockType
) -> None:
    """
    Test that main continues processing after some savers fail.
    """
    config = create_config_with_savers(
        create_saver_config("good_saver_1"),
        create_saver_config("bad_saver"),
        create_saver_config("good_saver_2"),
    )

    # Set up mixed success/failure scenario
    saver_names = ["good_saver_1", "bad_saver", "good_saver_2"]
    setup_mixed_success_failure_mocks(
        mock_init_from_cfg, mock_thread, saver_names, failure_index=1
    )

    main(config)

    # Verify error was logged for failed saver
    mock_logger.error.assert_called_once_with(
        "Data saver %s is not registered", "bad_saver"
    )

    # Verify successful savers still processed
    assert mock_thread.call_count == 2


# ======================================================================================
# LOGGING TESTS
# ======================================================================================


def test_main_logs_info_for_started_savers(
    mock_init_from_cfg: MockType,
    mock_thread: MockType,
    mock_logger: MockType,
    multi_saver_config: DictConfig,
) -> None:
    """
    Test that info messages are logged for started savers.
    """
    saver_names = ["csv_saver", "sqlite_saver", "jsonl_saver"]
    setup_multi_saver_mocks(mock_init_from_cfg, mock_thread, saver_names)

    main(multi_saver_config)

    # Verify info messages were logged for each started saver
    verify_info_logs_for_savers(mock_logger, saver_names)


def test_main_logs_errors_for_failed_savers(
    mock_init_from_cfg: MockType,
    mock_thread: MockType,
    mock_logger: MockType,
    multi_saver_config: DictConfig,
) -> None:
    """
    Test that error messages are logged for failed
    """
    mock_init_from_cfg.return_value = None  # All fail

    main(multi_saver_config)
    mock_thread.assert_not_called()  # No threads created

    # Verify error messages were logged for each failed saver
    verify_error_logs_for_unregistered_savers(
        mock_logger, ["csv_saver", "sqlite_saver", "jsonl_saver"]
    )


# ======================================================================================
# INTEGRATION TESTS
# ======================================================================================


def test_main_end_to_end_multiple_savers() -> None:
    """
    Test complete end-to-end execution with real thread behavior (multiple savers).
    """
    config = create_config_with_savers(
        create_saver_config("saver_1"),
        create_saver_config("saver_2"),
        create_saver_config("saver_3"),
    )

    executed_savers = []
    execution_threads = []

    with patch("app.data_layer.data_saver.save_data.init_from_cfg") as mock_init:
        with patch("app.data_layer.data_saver.save_data.logger") as mock_logger:
            # Create real mock savers that track execution
            mock_savers = [MockDataSaver(f"saver_{i}") for i in range(1, 4)]
            mock_init.side_effect = mock_savers

            # Track which savers executed and in which threads
            for saver in mock_savers:

                def make_tracked_retrieve(saver_name: str, orig_method):
                    def tracked_retrieve():
                        executed_savers.append(saver_name)
                        execution_threads.append(threading.current_thread().ident)

                        return orig_method()

                    return tracked_retrieve

                setattr(
                    saver,
                    "retrieve_and_save",
                    make_tracked_retrieve(saver.name, saver.retrieve_and_save),
                )

            main(config)

            # Verify all savers were executed
            assert set(executed_savers) == {"saver_1", "saver_2", "saver_3"}

            # Verify they ran in different threads (concurrency)
            assert (
                len(set(execution_threads)) > 1
            ), "Savers should run in different threads"

            # Verify logging
            assert mock_logger.info.call_count == 3


def test_main_with_realistic_config_structure() -> None:
    """
    Test with a realistic configuration structure.
    """
    realistic_config = OmegaConf.create(
        {
            "data_saver": [
                {
                    "csv_saver": {
                        "name": "csv_saver",
                        "csv_file_path": "/tmp/stock_data.csv",
                        "streaming": {
                            "kafka_topic": "stock_prices",
                            "kafka_server": "localhost:9092",
                        },
                    }
                },
                {
                    "sqlite_saver": {
                        "name": "sqlite_saver",
                        "sqlite_db": "/tmp/stock_data.db",
                        "streaming": {
                            "kafka_topic": "stock_prices",
                            "kafka_server": "localhost:9092",
                        },
                    }
                },
            ]
        }
    )

    with patch_save_data_dependencies() as (mock_init, mock_thread, _):
        saver_names = ["csv_saver", "sqlite_saver"]
        setup_multi_saver_mocks(mock_init, mock_thread, saver_names)

        main(realistic_config)

        # Verify both savers were processed
        assert mock_init.call_count == 2
        assert mock_thread.call_count == 2

        # Verify correct configurations were passed
        call_args_list = mock_init.call_args_list
        csv_config = call_args_list[0][0][0]
        sqlite_config = call_args_list[1][0][0]

        assert csv_config["name"] == "csv_saver"
        assert csv_config["csv_file_path"] == "/tmp/stock_data.csv"
        assert sqlite_config["name"] == "sqlite_saver"
        assert sqlite_config["sqlite_db"] == "/tmp/stock_data.db"
