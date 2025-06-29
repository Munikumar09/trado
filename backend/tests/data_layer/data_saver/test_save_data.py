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
        Create a MockDataSaver instance for testing, with configurable failure behavior and simulated execution time.
        
        Parameters:
            name (str): Identifier for the mock data saver.
            should_fail (bool, optional): If True, the saver will simulate a failure during execution. Defaults to False.
            execution_time (float, optional): Time in seconds to simulate work during execution. Defaults to 0.1.
        """
        self.name = name
        self.should_fail = should_fail
        self.execution_time = execution_time
        self.retrieve_and_save_called = False
        self.thread_id = None

    def retrieve_and_save(self):
        """
        Simulates data retrieval and saving, optionally raising an error to mimic failure.
        
        Sets flags indicating execution, records the current thread ID, and sleeps for the configured execution time. Raises a RuntimeError if the instance is configured to fail.
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
        Create a MockDataSaver instance from a configuration object.
        
        Parameters:
        	cfg (DictConfig): Configuration containing 'name', 'should_fail', and 'execution_time' settings.
        
        Returns:
        	MockDataSaver: A new instance initialized with values from the configuration.
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
    Create an OmegaConf configuration containing a list of data saver configurations.
    
    Parameters:
        *saver_configs: One or more dictionaries representing individual saver configurations.
    
    Returns:
        DictConfig: An OmegaConf configuration object with a 'data_saver' key containing the provided saver configurations.
    """
    return OmegaConf.create({"data_saver": list(saver_configs)})


def create_saver_config(
    name: str, saver_type: str = "mock", **kwargs
) -> dict[str, Any]:
    """
    Generate a configuration dictionary for a single data saver.
    
    Parameters:
        name (str): The unique name of the data saver.
        saver_type (str): The type identifier for the data saver. Defaults to "mock".
        **kwargs: Additional configuration fields to include in the saver config.
    
    Returns:
        dict[str, Any]: A dictionary mapping the saver name to its configuration dictionary.
    """
    config = {"name": name, "type": saver_type, **kwargs}
    return {name: config}


def setup_multi_saver_mocks(
    mock_init_from_cfg: MockType, mock_thread: MockType, saver_names: list[str]
) -> tuple[list[MockDataSaver], list[MagicMock]]:
    """
    Creates mock data saver instances and corresponding mock thread instances for a list of saver names.
    
    Returns:
        A tuple containing a list of `MockDataSaver` instances and a list of mock thread instances, one for each saver name.
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
    Creates and configures mock objects for testing a single data saver and its associated thread.
    
    Returns:
        A tuple containing the mock data saver instance and the mock thread instance.
    """
    mock_saver = MockDataSaver(saver_name)
    mock_init_from_cfg.return_value = mock_saver

    mock_thread_instance = MagicMock()
    mock_thread.return_value = mock_thread_instance

    return mock_saver, mock_thread_instance


def verify_thread_operations(mock_thread_instances: list[MagicMock]) -> None:
    """
    Assert that each mock thread instance was started and joined exactly once.
    
    Parameters:
        mock_thread_instances (list[MagicMock]): List of mock thread instances to verify.
    """
    for thread_instance in mock_thread_instances:
        thread_instance.start.assert_called_once()
        thread_instance.join.assert_called_once()


def verify_info_logs_for_savers(mock_logger: MockType, saver_names: list[str]) -> None:
    """
    Assert that info logs were generated for each saver name indicating the saver was started.
    
    Checks that the mock logger's info method was called with the expected messages for all provided saver names.
    """
    expected_calls = [call("Starting the saver %s", name) for name in saver_names]
    mock_logger.info.assert_has_calls(expected_calls, any_order=True)


def verify_error_logs_for_unregistered_savers(
    mock_logger: MockType, saver_names: list[str]
) -> None:
    """
    Assert that error logs were generated for each unregistered saver name.
    
    Checks that the mock logger's error method was called with the expected messages for all provided saver names.
    """
    expected_error_calls = [
        call("Data saver %s is not registered", name) for name in saver_names
    ]
    mock_logger.error.assert_has_calls(expected_error_calls, any_order=True)


def verify_thread_creation_for_savers(
    mock_thread: MockType, mock_savers: list[MockDataSaver]
) -> None:
    """
    Assert that a thread was created for each mock saver with the correct target function.
    
    Checks that the mock Thread class was called with each saver's `retrieve_and_save` method as the thread target.
    """
    expected_calls = [call(target=saver.retrieve_and_save) for saver in mock_savers]
    mock_thread.assert_has_calls(expected_calls, any_order=True)


@contextmanager
def patch_save_data_dependencies() -> (
    Generator[tuple[MockType, MockType, MockType], None, None]
):
    """
    Context manager that patches the `init_from_cfg`, `Thread`, and `logger` dependencies in the `save_data` module for testing.
    
    Yields:
        tuple[MockType, MockType, MockType]: A tuple containing mocks for `init_from_cfg`, `Thread`, and `logger`.
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
    Configures mocks for a scenario where saver creation partially fails, returning a list of mock savers with one failure, the successful savers, and corresponding mock thread instances.
    
    Parameters:
        success_names (list[str]): Names of savers to simulate as successful.
        failure_index (int): Index in the list where saver creation should fail (returns None).
    
    Returns:
        mock_savers_with_none (list[MockDataSaver | None]): List of mock savers with None at the failure index.
        successful_mock_savers (list[MockDataSaver]): List of successfully created mock savers.
        mock_thread_instances (list[MagicMock]): Mock thread instances for each successful saver.
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
    Fixture that provides a mock for the logger used in the save_data module.
    """
    return mocker.patch("app.data_layer.data_saver.save_data.logger")


@pytest.fixture
def mock_init_from_cfg(mocker: MockerFixture) -> MockType:
    """
    Fixture that patches the `init_from_cfg` function in the `save_data` module for use in tests.
    
    Returns:
        MockType: The mock object replacing `init_from_cfg`.
    """
    return mocker.patch("app.data_layer.data_saver.save_data.init_from_cfg")


@pytest.fixture
def mock_thread(mocker: MockerFixture) -> MockType:
    """
    Fixture that patches the Thread class used in the save_data module for testing purposes.
    
    Returns:
        MockType: The patched Thread class mock.
    """
    return mocker.patch("app.data_layer.data_saver.save_data.Thread")


@pytest.fixture
def basic_single_saver_config() -> DictConfig:
    """
    Provides a DictConfig containing a single data saver configuration for testing purposes.
    
    Returns:
        DictConfig: Configuration object with one mock data saver entry.
    """
    return create_config_with_savers(create_saver_config("test_saver"))


@pytest.fixture
def multi_saver_config() -> DictConfig:
    """
    Provides a configuration containing multiple data saver entries for testing.
    
    Returns:
        DictConfig: An OmegaConf configuration with three savers: 'csv_saver', 'sqlite_saver', and 'jsonl_saver'.
    """
    return create_config_with_savers(
        create_saver_config("csv_saver"),
        create_saver_config("sqlite_saver"),
        create_saver_config("jsonl_saver"),
    )


@pytest.fixture
def empty_config() -> DictConfig:
    """
    Provides a configuration object with an empty list of data savers.
    
    Returns:
        DictConfig: A configuration containing no data savers.
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
    Verify that the main function processes a configuration with a single data saver, correctly initializes the saver, creates and manages its thread, and logs the appropriate info message.
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
    Verify that the main function processes a configuration with multiple data savers by initializing each saver, creating and managing separate threads for each, and logging info for all savers.
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
    Verify that the main function correctly handles an empty data saver configuration by not creating savers, threads, or logging info messages.
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
    Test that the main function correctly processes a configuration with nested saver structures, ensuring the saver name and relevant configuration fields are extracted and passed to the saver initializer.
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
    Verify that `main` logs an error and does not create a thread when data saver creation fails.
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
    Test that `main` logs an error and only creates threads for savers that are successfully created when some saver initializations fail.
    
    Verifies that an error is logged for each failed saver creation and that threads are only started for savers that were successfully instantiated.
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
    Test that `main` logs errors and does not create threads when all data saver initializations fail.
    
    Verifies that an error is logged for each saver and that no threads are started if `init_from_cfg` returns `None` for all savers in the configuration.
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
    Verify that the main function creates a separate thread for each data saver in the configuration.
    
    Ensures that the correct number of threads are created and that each thread is initialized with the appropriate target saver.
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
    Test that `main` propagates exceptions raised during saver initialization and does not create threads or log info messages in such cases.
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
    Verify that the main function continues to process and start threads for successfully created savers even if some savers fail to initialize, and logs an error for each failed saver.
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
    Verify that the main function logs info messages for each saver that is started when processing a configuration with multiple savers.
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
    Verify that error messages are logged for each saver that fails to be created and that no threads are started when all saver initializations fail.
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
    Performs an end-to-end test of the main function with multiple savers, verifying concurrent execution in separate threads and correct logging.
    
    This test creates three mock savers, tracks their execution and thread IDs, runs the main function with a configuration containing all savers, and asserts that each saver is executed, that execution occurs in multiple threads, and that an info log is generated for each saver.
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
                    """
                    Wraps a data saver retrieval method to track its execution and the thread ID.
                    
                    Parameters:
                        saver_name (str): The name of the saver being tracked.
                        orig_method (Callable): The original retrieval method to be wrapped.
                    
                    Returns:
                        Callable: A function that, when called, records the saver name and thread ID, then invokes the original method.
                    """
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
    Verifies that the main function correctly processes a realistic, nested configuration structure with multiple savers, ensuring each saver is initialized with the appropriate configuration and corresponding threads are created.
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
