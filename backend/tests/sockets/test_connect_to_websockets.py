"""
Unit tests for the connect_to_websockets module. This module contains comprehensive
tests for the websocket connection functionality, including connection creation,
thread management, configuration handling, and error scenarios.

Test Coverage:
- Connection creation with varying numbers of connections (0, 1, multiple)
- Thread creation and management
- Error handling (init_from_cfg failures, mixed success/failure scenarios)
- Main function workflow with single and multiple connections
- Integration testing for end-to-end flow
"""

from threading import Thread
from unittest.mock import MagicMock, call

import pytest
from omegaconf import DictConfig, OmegaConf
from pytest_mock import MockerFixture

from app.sockets.connect_to_websockets import create_websocket_connection, main
from app.sockets.connections import WebsocketConnection


@pytest.fixture
def mock_logger(mocker: MockerFixture) -> MagicMock:
    """
    Mock logger for connect_to_websockets module.
    """
    return mocker.patch("app.sockets.connect_to_websockets.logger")


@pytest.fixture
def mock_websocket_connection() -> MagicMock:
    """
    Mock WebsocketConnection fixture.
    """
    mock_connection = MagicMock(spec=WebsocketConnection)
    mock_connection.websocket = MagicMock()
    mock_connection.websocket.connect = MagicMock()

    return mock_connection


@pytest.fixture(autouse=True)
def mock_init_from_cfg(
    mocker: MockerFixture, mock_websocket_connection: MagicMock
) -> MagicMock:
    """
    Mock init_from_cfg fixture.
    """
    return mocker.patch(
        "app.sockets.connect_to_websockets.init_from_cfg",
        return_value=mock_websocket_connection,
    )


@pytest.fixture
def mock_sleep(mocker: MockerFixture) -> MagicMock:
    """
    Mock time.sleep fixture.
    """
    return mocker.patch("app.sockets.connect_to_websockets.time.sleep")


@pytest.fixture(autouse=True)
def mock_thread(mocker: MockerFixture) -> MagicMock:
    """
    Mock Thread fixture.
    """
    return mocker.patch(
        "app.sockets.connect_to_websockets.Thread",
        return_value=MagicMock(),
        spec=Thread,
    )


@pytest.fixture
def mock_create_tokens_db(mocker: MockerFixture) -> MagicMock:
    """
    Mock create_tokens_db function.
    """
    return mocker.patch("app.sockets.connect_to_websockets.create_tokens_db")


@pytest.fixture
def single_connection_cfg() -> DictConfig:
    """
    Configuration fixture with single connection.
    """
    config_dict = {
        "connections": [
            {
                "connection": {
                    "num_connections": 1,
                    "current_connection_number": 0,
                    "use_thread": True,
                    "name": "single_connection",
                }
            }
        ]
    }
    return OmegaConf.create(config_dict)


class TestCreateWebsocketConnection:
    """
    Test cases for the create_websocket_connection function.
    """

    @pytest.fixture
    def base_cfg(self) -> DictConfig:
        """
        Basic configuration fixture for websocket connections.
        """
        config_dict = {
            "connection": {
                "num_connections": 2,
                "current_connection_number": 0,
                "use_thread": True,
                "name": "test_connection",
                "provider": {"test": "value"},
            }
        }
        return OmegaConf.create(config_dict)

    @pytest.fixture
    def minimal_cfg(self) -> DictConfig:
        """
        Minimal configuration fixture for single connection.
        """
        config_dict = {
            "connection": {
                "num_connections": 1,
                "current_connection_number": 0,
                "use_thread": False,
                "name": "minimal_connection",
            }
        }
        return OmegaConf.create(config_dict)

    def test_create_websocket_connection_success(
        self,
        mock_sleep: MagicMock,
        mock_thread: MagicMock,
        mock_init_from_cfg: MagicMock,
        base_cfg: DictConfig,
        mock_websocket_connection: MagicMock,
        mock_logger: MagicMock,
    ):
        """
        Test successful creation of multiple websocket connections with threading.
        """

        result = create_websocket_connection(base_cfg)

        assert len(result) == 2
        assert mock_init_from_cfg.call_count == 2
        assert mock_thread.call_count == 2
        assert mock_thread.return_value.start.call_count == 2
        assert mock_sleep.call_count == 2

        # Verify thread creation with correct parameters
        thread_calls = mock_thread.call_args_list
        assert len(thread_calls) == 2

        # Check first thread call
        first_call = thread_calls[0]
        assert (
            first_call.kwargs["target"] == mock_websocket_connection.websocket.connect
        )
        assert first_call.kwargs["args"] == (True,)
        assert first_call.kwargs["name"] == "WebSocketConnection-0"

        # Check second thread call
        second_call = thread_calls[1]
        assert (
            second_call.kwargs["target"] == mock_websocket_connection.websocket.connect
        )
        assert second_call.kwargs["args"] == (True,)
        assert second_call.kwargs["name"] == "WebSocketConnection-1"

        # Ensure logger was called correctly
        mock_logger.info.assert_any_call("Creating connection instance %s", 0)
        mock_logger.info.assert_any_call("Creating connection instance %s", 1)

    def test_create_websocket_connection_without_copy_method(
        self, mock_init_from_cfg: MagicMock, minimal_cfg: DictConfig
    ):
        """
        Test connection creation when configuration lacks copy method.
        """
        minimal_cfg.connection.current_connection_number = 3
        minimal_cfg.connection.use_thread = True

        result = create_websocket_connection(minimal_cfg)

        called_cfg = mock_init_from_cfg.call_args[0][0]
        assert called_cfg.current_connection_number == 3
        assert id(called_cfg) != id(minimal_cfg)

        assert len(result) == 1

        # Verify that current_connection_number is correctly set
        assert mock_init_from_cfg.call_count == 1

    def test_create_websocket_connection_init_from_cfg_returns_none(
        self,
        mock_sleep: MagicMock,
        mock_init_from_cfg: MagicMock,
        minimal_cfg: DictConfig,
    ):
        """
        Test behavior when init_from_cfg returns None.
        """
        # Setup mock to return None
        mock_init_from_cfg.return_value = None

        result = create_websocket_connection(minimal_cfg)

        assert not result
        assert mock_init_from_cfg.call_count == 1
        assert mock_sleep.call_count == 1

    def test_create_websocket_connection_zero_connections(
        self,
        mock_sleep: MagicMock,
        mock_thread: MagicMock,
        mock_init_from_cfg: MagicMock,
        minimal_cfg: DictConfig,
    ):
        """
        Test behavior when num_connections is zero.
        """
        # Setup configuration with zero connections
        minimal_cfg.connection.num_connections = 0

        result = create_websocket_connection(minimal_cfg)

        assert not result
        assert mock_init_from_cfg.call_count == 0
        assert mock_thread.call_count == 0
        assert mock_sleep.call_count == 0

    def test_create_websocket_connection_exception_in_init_from_cfg(
        self,
        mock_init_from_cfg: MagicMock,
        minimal_cfg: DictConfig,
    ):
        """
        Test behavior when init_from_cfg raises an exception.
        """
        # Setup mock to raise exception
        mock_init_from_cfg.side_effect = Exception("Initialization failed")

        with pytest.raises(Exception, match="Initialization failed"):
            create_websocket_connection(minimal_cfg)

    def test_create_websocket_connection_mixed_success_failure(
        self,
        mock_sleep: MagicMock,
        mock_thread: MagicMock,
        mock_init_from_cfg: MagicMock,
        mock_websocket_connection: MagicMock,
        minimal_cfg: DictConfig,
        mock_logger: MagicMock,
    ):
        """
        Test behavior when some connections succeed and others fail.
        """
        # Setup configuration
        minimal_cfg.connection.num_connections = 3
        minimal_cfg.connection.use_thread = True

        # Setup mock to return None for second connection
        mock_init_from_cfg.side_effect = [
            mock_websocket_connection,  # First succeeds
            None,  # Second fails
            mock_websocket_connection,  # Third succeeds
        ]

        result = create_websocket_connection(minimal_cfg)

        # Verify only successful connections are returned
        assert len(result) == 2
        assert mock_init_from_cfg.call_count == 3
        assert mock_thread.call_count == 2
        assert mock_thread.return_value.start.call_count == 2
        assert mock_sleep.call_count == 3  # Sleep called after each iteration

        mock_logger.error.assert_called_once_with(
            "Failed to create WebsocketConnection instance for connection %s", 1
        )


class TestMain:
    """
    Test cases for the main function.
    """

    @pytest.fixture
    def multi_connection_cfg(self) -> DictConfig:
        """
        Configuration fixture with multiple connections.
        """
        config_dict = {
            "connections": [
                {
                    "connection": {
                        "num_connections": 2,
                        "current_connection_number": 0,
                        "use_thread": True,
                        "name": "connection1",
                    }
                },
                {
                    "connection": {
                        "num_connections": 1,
                        "current_connection_number": 0,
                        "use_thread": False,
                        "name": "connection2",
                    }
                },
            ]
        }
        return OmegaConf.create(config_dict)

    @pytest.fixture
    def mock_create_connection(self, mocker) -> MagicMock:
        """
        Mock create_websocket_connection function.
        """
        return mocker.patch(
            "app.sockets.connect_to_websockets.create_websocket_connection"
        )

    def test_main_single_connection(
        self,
        mock_create_connection: MagicMock,
        mock_create_tokens_db: MagicMock,
        single_connection_cfg: DictConfig,
        mock_thread: MagicMock,
    ):
        """
        Test main function with single connection configuration.
        """
        # Setup mocks
        mock_create_connection.return_value = [mock_thread]

        main(single_connection_cfg)

        mock_create_tokens_db.assert_called_once()
        mock_create_connection.assert_called_once()
        mock_thread.join.assert_called_once()

    def test_main_multiple_connections(
        self,
        mock_create_connection: MagicMock,
        mock_create_tokens_db: MagicMock,
        multi_connection_cfg: DictConfig,
    ):
        """
        Test main function with multiple connection configurations.
        """
        # Setup mocks
        mock_thread1 = MagicMock(spec=Thread)
        mock_thread2 = MagicMock(spec=Thread)
        mock_thread3 = MagicMock(spec=Thread)

        mock_create_connection.side_effect = [
            [mock_thread1, mock_thread2],  # First connection returns 2 threads
            [mock_thread3],  # Second connection returns 1 thread
        ]

        main(multi_connection_cfg)

        mock_create_tokens_db.assert_called_once()
        assert mock_create_connection.call_count == 2

        # Verify all threads are joined
        mock_thread1.join.assert_called_once()
        mock_thread2.join.assert_called_once()
        mock_thread3.join.assert_called_once()

    def test_main_empty_connections(
        self,
        mock_create_connection: MagicMock,
        mock_create_tokens_db: MagicMock,
    ):
        """
        Test main function with empty connections list.
        """
        # Setup configuration with empty connections
        config_dict: dict[str, list] = {"connections": []}
        cfg = OmegaConf.create(config_dict)

        main(cfg)

        mock_create_tokens_db.assert_called_once()
        mock_create_connection.assert_not_called()

    def test_main_connection_returns_empty_list(
        self,
        mock_create_connection: MagicMock,
        mock_create_tokens_db: MagicMock,
        single_connection_cfg: DictConfig,
    ):
        """
        Test main function when create_websocket_connection returns empty list.
        """
        # Setup mock to return empty list
        mock_create_connection.return_value = []

        main(single_connection_cfg)

        mock_create_tokens_db.assert_called_once()
        mock_create_connection.assert_called_once()
        # No threads to join, so no join calls

    def test_main_preserves_connection_order(
        self,
        mock_create_connection: MagicMock,
        multi_connection_cfg: DictConfig,
    ):
        """
        Test that main function processes connections in the correct order.
        """
        # Setup mocks
        mock_thread1 = MagicMock(spec=Thread)
        mock_thread2 = MagicMock(spec=Thread)

        mock_create_connection.side_effect = [
            [mock_thread1],
            [mock_thread2],
        ]

        main(multi_connection_cfg)

        # Verify call order
        expected_calls = [
            call(multi_connection_cfg.connections[0]),
            call(multi_connection_cfg.connections[1]),
        ]
        mock_create_connection.assert_has_calls(expected_calls)

    def test_main_with_none_threads(
        self,
        mock_create_connection: MagicMock,
        single_connection_cfg: DictConfig,
    ):
        """
        Test main function behavior when connection list contains None values.
        """
        # Setup mock to return list with None
        mock_thread = MagicMock(spec=Thread)
        mock_create_connection.return_value = [mock_thread, None]

        with pytest.raises(AttributeError):
            main(single_connection_cfg)


class TestModuleIntegration:
    """
    Integration tests for the module functionality.
    """

    def test_end_to_end_flow(
        self,
        mock_sleep: MagicMock,
        mock_thread: MagicMock,
        mock_init_from_cfg: MagicMock,
        mock_create_tokens_db: MagicMock,
        single_connection_cfg: DictConfig,
    ):
        """
        Test the complete end-to-end flow from main to thread creation.
        """
        # Setup configuration

        single_connection_cfg.connections[0].connection.num_connections = 2

        main(single_connection_cfg)

        # Verify complete flow
        mock_create_tokens_db.assert_called_once()
        assert mock_init_from_cfg.call_count == 2
        assert mock_thread.call_count == 2
        assert mock_thread.return_value.start.call_count == 2
        assert mock_thread.return_value.join.call_count == 2
        assert mock_sleep.call_count == 2
