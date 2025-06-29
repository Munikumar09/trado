"""
Comprehensive tests for the StockTickerServerFactory.buildProtocol method.

This test suite covers:
- Successful protocol building with valid address
- Protocol building when parent returns None
- Protocol building when protocol doesn't have connection_manager attribute
- Connection manager assignment verification
- Logging functionality during protocol building
- Different address formats and edge cases
"""

from unittest.mock import MagicMock, patch

import pytest

from app.sockets.websocket_server_manager import ConnectionManager
from app.sockets.websocket_server_protocol import (
    StockTickerServerFactory,
    StockTickerServerProtocol,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_connection_manager():
    """
    Pytest fixture that returns a mock ConnectionManager instance for isolated testing without real Redis or WebSocket connections.
    """
    mock_manager = MagicMock(spec=ConnectionManager)
    return mock_manager


@pytest.fixture
def mock_logger():
    """
    Pytest fixture that provides a mocked module-level logger for verifying log output during tests.
    
    Yields:
        MagicMock: The patched logger instance for asserting logging calls and messages.
    """
    with patch("app.sockets.websocket_server_protocol.logger") as mock:
        yield mock


@pytest.fixture
def sample_factory(mock_connection_manager):
    """
    Provides a StockTickerServerFactory instance initialized with a mock connection manager and test URL for use in tests.
    """
    url = "ws://localhost:8080"
    factory = StockTickerServerFactory(url, mock_connection_manager)
    return factory


@pytest.fixture
def sample_addresses():
    """
    Provides a list of sample client address strings in various formats for use in parameterized tests of protocol building.
    """
    return [
        "127.0.0.1:12345",
        "192.168.1.100:8080",
        "localhost:9000",
        "10.0.0.1:3000",
        "example.com:443",
    ]


CLIENT_ADDRESS = "127.0.0.1:12345"


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


def test_factory_initialization_with_valid_connection_manager(
    mock_connection_manager, mock_logger
):
    """
    Test successful factory initialization with valid connection manager. Verifies that the
    factory initializes correctly with proper connection manager assignment and appropriate
    info logging.
    """
    url = "ws://localhost:8080"

    factory = StockTickerServerFactory(url, mock_connection_manager)

    assert factory.connection_manager is mock_connection_manager
    assert factory.protocol == StockTickerServerProtocol

    # Verify info logging
    mock_logger.info.assert_called_once_with(
        "StockTickerServerFactory initialized with a valid connection_manager and URL: %s",
        url,
    )


def test_factory_initialization_with_none_connection_manager(mock_logger):
    """
    Test that initializing StockTickerServerFactory with a None connection manager raises a ValueError and does not log any info messages.
    """
    url = "ws://localhost:8080"

    with pytest.raises(ValueError) as exc_info:
        StockTickerServerFactory(url, None)

    assert (
        str(exc_info.value)
        == "connection_manager cannot be None. Please provide a valid connection manager."
    )

    # Verify no info logging occurred due to early failure
    mock_logger.info.assert_not_called()


# =============================================================================
# BUILD PROTOCOL SUCCESS TESTS
# =============================================================================


def test_build_protocol_success_with_valid_address(sample_factory, mock_logger):
    """
    Test that buildProtocol successfully creates a protocol with a valid address, assigns the connection manager, and emits the expected debug logs.
    
    Ensures the parent buildProtocol is called, the returned protocol receives the factory's connection manager, and both building and initialization debug messages are logged.
    """

    # Create mock protocol with connection_manager attribute
    mock_protocol = MagicMock(spec=StockTickerServerProtocol)
    mock_protocol.connection_manager = None

    with patch.object(
        sample_factory.__class__.__bases__[0], "buildProtocol"
    ) as mock_parent_build:
        mock_parent_build.return_value = mock_protocol

        result = sample_factory.buildProtocol(CLIENT_ADDRESS)

        # Verify parent method was called
        mock_parent_build.assert_called_once_with(CLIENT_ADDRESS)

        # Verify protocol is returned
        assert result is mock_protocol

        # Verify connection_manager was assigned
        assert mock_protocol.connection_manager is sample_factory.connection_manager

        # Verify debug logging
        expected_debug_calls = [
            ("Building protocol for address: %s", CLIENT_ADDRESS),
            (
                "Protocol for %s has been initialized with connection_manager",
                CLIENT_ADDRESS,
            ),
        ]

        assert mock_logger.debug.call_count == 2
        actual_calls = [call.args for call in mock_logger.debug.call_args_list]
        assert actual_calls == expected_debug_calls


@pytest.mark.parametrize(
    "addr",
    [
        "127.0.0.1:12345",
        "192.168.1.100:8080",
        "localhost:9000",
        "10.0.0.1:3000",
        "example.com:443",
    ],
)
def test_build_protocol_with_various_addresses(sample_factory, mock_logger, addr):
    """
    Test that buildProtocol correctly handles and logs various client address formats.
    
    Verifies that the protocol receives the factory's connection manager and that debug logs are emitted for each address type.
    """
    mock_protocol = MagicMock(spec=StockTickerServerProtocol)
    mock_protocol.connection_manager = None

    with patch.object(
        sample_factory.__class__.__bases__[0], "buildProtocol"
    ) as mock_parent_build:
        mock_parent_build.return_value = mock_protocol

        result = sample_factory.buildProtocol(addr)

        assert result is mock_protocol
        assert mock_protocol.connection_manager is sample_factory.connection_manager

        # Verify address-specific debug logging
        mock_logger.debug.assert_any_call("Building protocol for address: %s", addr)
        mock_logger.debug.assert_any_call(
            "Protocol for %s has been initialized with connection_manager", addr
        )


def test_build_protocol_connection_manager_assignment_verification(
    sample_factory, mock_connection_manager
):
    """
    Verifies that the factory's connection manager is correctly assigned to the protocol instance created by buildProtocol.
    """
    mock_protocol = MagicMock(spec=StockTickerServerProtocol)
    mock_protocol.connection_manager = None

    with patch.object(
        sample_factory.__class__.__bases__[0], "buildProtocol"
    ) as mock_parent_build:
        mock_parent_build.return_value = mock_protocol

        sample_factory.buildProtocol(CLIENT_ADDRESS)

        # Verify the exact connection manager instance was assigned
        assert mock_protocol.connection_manager is mock_connection_manager
        assert mock_protocol.connection_manager is sample_factory.connection_manager


# =============================================================================
# BUILD PROTOCOL EDGE CASES
# =============================================================================


def test_build_protocol_when_parent_returns_none(sample_factory, mock_logger):
    """
    Test that buildProtocol returns None and skips connection manager assignment when the parent buildProtocol method returns None.
    """
    with patch.object(
        sample_factory.__class__.__bases__[0], "buildProtocol"
    ) as mock_parent_build:
        mock_parent_build.return_value = None

        result = sample_factory.buildProtocol(CLIENT_ADDRESS)

        # Verify None is returned
        assert result is None

        # Verify parent method was called
        mock_parent_build.assert_called_once_with(CLIENT_ADDRESS)

        # Verify only the building debug log, not the initialization log
        mock_logger.debug.assert_called_once_with(
            "Building protocol for address: %s", CLIENT_ADDRESS
        )

        # Verify initialization debug log was NOT called
        initialization_calls = [
            call
            for call in mock_logger.debug.call_args_list
            if "has been initialized with connection_manager" in str(call)
        ]
        assert len(initialization_calls) == 0


def test_build_protocol_without_connection_manager_attribute(
    sample_factory, mock_logger
):
    """
    Test that buildProtocol handles protocols lacking a connection_manager attribute without error.
    
    Verifies that if the protocol returned by the parent buildProtocol method does not have a connection_manager attribute, the method does not attempt assignment or raise an AttributeError, and only the building debug log is emitted.
    """
    # Create mock protocol without connection_manager attribute
    mock_protocol = MagicMock()

    # Explicitly remove the connection_manager attribute
    if hasattr(mock_protocol, "connection_manager"):
        delattr(mock_protocol, "connection_manager")

    with patch.object(
        sample_factory.__class__.__bases__[0], "buildProtocol"
    ) as mock_parent_build:
        mock_parent_build.return_value = mock_protocol

        result = sample_factory.buildProtocol(CLIENT_ADDRESS)

        # Verify protocol is returned unchanged
        assert result is mock_protocol

        # Verify parent method was called
        mock_parent_build.assert_called_once_with(CLIENT_ADDRESS)

        # Verify only building debug log, not initialization log
        mock_logger.debug.assert_called_once_with(
            "Building protocol for address: %s", CLIENT_ADDRESS
        )

        # Verify no connection_manager attribute was added
        assert not hasattr(mock_protocol, "connection_manager")


def test_build_protocol_with_protocol_having_existing_connection_manager(
    sample_factory, mock_logger
):
    """
    Tests that buildProtocol overwrites any existing connection_manager on the protocol with the factory's connection_manager and emits the expected debug logs.
    """
    # Create mock protocol with existing connection_manager
    existing_manager = MagicMock()
    mock_protocol = MagicMock(spec=StockTickerServerProtocol)
    mock_protocol.connection_manager = existing_manager

    with patch.object(
        sample_factory.__class__.__bases__[0], "buildProtocol"
    ) as mock_parent_build:
        mock_parent_build.return_value = mock_protocol

        result = sample_factory.buildProtocol(CLIENT_ADDRESS)

        # Verify protocol is returned
        assert result is mock_protocol

        # Verify connection_manager was overwritten with factory's manager
        assert mock_protocol.connection_manager is sample_factory.connection_manager
        assert mock_protocol.connection_manager is not existing_manager

        # Verify both debug logs were called
        assert mock_logger.debug.call_count == 2
        mock_logger.debug.assert_any_call(
            "Building protocol for address: %s", CLIENT_ADDRESS
        )
        mock_logger.debug.assert_any_call(
            "Protocol for %s has been initialized with connection_manager",
            CLIENT_ADDRESS,
        )


# =============================================================================
# LOGGING VERIFICATION TESTS
# =============================================================================


def test_build_protocol_debug_logging_format(sample_factory, mock_logger):
    """
    Verify that `buildProtocol` emits debug log messages with the correct format strings and arguments, ensuring logging output matches expected monitoring and debugging standards.
    """
    mock_protocol = MagicMock(spec=StockTickerServerProtocol)
    mock_protocol.connection_manager = None

    with patch.object(
        sample_factory.__class__.__bases__[0], "buildProtocol"
    ) as mock_parent_build:
        mock_parent_build.return_value = mock_protocol

        sample_factory.buildProtocol(CLIENT_ADDRESS)

        # Verify exact debug log calls
        expected_calls = [
            ("Building protocol for address: %s", CLIENT_ADDRESS),
            (
                "Protocol for %s has been initialized with connection_manager",
                CLIENT_ADDRESS,
            ),
        ]

        actual_calls = [call.args for call in mock_logger.debug.call_args_list]
        assert actual_calls == expected_calls


def test_build_protocol_no_info_or_error_logging(sample_factory, mock_logger):
    """
    Verify that buildProtocol emits only debug log messages and does not generate info, warning, error, exception, or critical logs during normal protocol creation.
    """
    mock_protocol = MagicMock(spec=StockTickerServerProtocol)
    mock_protocol.connection_manager = None

    with patch.object(
        sample_factory.__class__.__bases__[0], "buildProtocol"
    ) as mock_parent_build:
        mock_parent_build.return_value = mock_protocol

        sample_factory.buildProtocol(CLIENT_ADDRESS)

        # Verify no higher-level logging occurred
        mock_logger.info.assert_not_called()
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()
        mock_logger.exception.assert_not_called()
        mock_logger.critical.assert_not_called()


# =============================================================================
# INTEGRATION AND ERROR HANDLING TESTS
# =============================================================================


def test_build_protocol_parent_method_exception_propagation(
    sample_factory, mock_logger
):
    """
    Verify that exceptions raised by the parent buildProtocol method are propagated and not caught, and that the building debug log is emitted before the exception.
    """
    with patch.object(
        sample_factory.__class__.__bases__[0], "buildProtocol"
    ) as mock_parent_build:
        mock_parent_build.side_effect = RuntimeError("Parent build failed")

        with pytest.raises(RuntimeError) as exc_info:
            sample_factory.buildProtocol(CLIENT_ADDRESS)

        assert str(exc_info.value) == "Parent build failed"

        # Verify building debug log was called before the exception
        mock_logger.debug.assert_called_once_with(
            "Building protocol for address: %s", CLIENT_ADDRESS
        )


def test_build_protocol_hasattr_check_safety(sample_factory, mock_logger):
    """
    Tests that the `buildProtocol` method safely handles protocols whose `__getattr__` method raises exceptions when checking for the `connection_manager` attribute, ensuring no unexpected errors occur and only the appropriate debug log is emitted.
    """

    # Create mock protocol with custom __getattr__ that raises exception
    class CustomMagicMock(MagicMock):
        """
        Custom MagicMock that simulates a protocol with a failing __getattr__.
        """

        def __getattr__(self, name):
            """
            Raises an AttributeError with a custom message when accessing the 'connection_manager' attribute; otherwise, delegates attribute access to the superclass.
            """
            if name == "connection_manager":
                raise AttributeError("Custom getattr error")
            return super().__getattr__(name)

    mock_protocol = CustomMagicMock()

    with patch.object(
        sample_factory.__class__.__bases__[0], "buildProtocol"
    ) as mock_parent_build:
        mock_parent_build.return_value = mock_protocol

        # This should not raise an exception due to hasattr check
        result = sample_factory.buildProtocol(CLIENT_ADDRESS)

        # Verify protocol is returned without connection_manager assignment
        assert result is mock_protocol

        # Verify only building debug log was called
        mock_logger.debug.assert_called_once_with(
            "Building protocol for address: %s", CLIENT_ADDRESS
        )


def test_build_protocol_multiple_consecutive_calls(sample_factory, mock_logger):
    """
    Tests that multiple consecutive calls to buildProtocol correctly assign the connection manager to each protocol instance, emit the expected debug logs, and delegate to the parent method for each client address.
    """
    addresses = ["127.0.0.1:12345", "192.168.1.100:8080", "10.0.0.1:3000"]
    protocols = []

    with patch.object(
        sample_factory.__class__.__bases__[0], "buildProtocol"
    ) as mock_parent_build:
        for address in addresses:
            mock_protocol = MagicMock(spec=StockTickerServerProtocol)
            mock_protocol.connection_manager = None
            protocols.append(mock_protocol)

            mock_parent_build.return_value = mock_protocol

            result = sample_factory.buildProtocol(address)

            # Verify each protocol gets the connection manager
            assert result is mock_protocol
            assert mock_protocol.connection_manager is sample_factory.connection_manager

    # Verify all debug logs were called
    assert mock_logger.debug.call_count == len(addresses) * 2  # 2 logs per call

    # Verify parent method was called for each address
    assert mock_parent_build.call_count == len(addresses)


def test_build_protocol_with_none_address(sample_factory, mock_logger):
    """
    Test that buildProtocol correctly handles a None address by passing it to the parent method, assigning the connection manager, and emitting appropriate debug logs.
    """
    addr = None
    mock_protocol = MagicMock(spec=StockTickerServerProtocol)
    mock_protocol.connection_manager = None

    with patch.object(
        sample_factory.__class__.__bases__[0], "buildProtocol"
    ) as mock_parent_build:
        mock_parent_build.return_value = mock_protocol

        result = sample_factory.buildProtocol(addr)

        # Verify protocol is returned with connection manager assigned
        assert result is mock_protocol
        assert mock_protocol.connection_manager is sample_factory.connection_manager

        # Verify parent method was called with None
        mock_parent_build.assert_called_once_with(None)

        # Verify debug logging handled None address
        mock_logger.debug.assert_any_call("Building protocol for address: %s", None)
        mock_logger.debug.assert_any_call(
            "Protocol for %s has been initialized with connection_manager", None
        )
