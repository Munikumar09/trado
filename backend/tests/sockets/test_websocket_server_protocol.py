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
    Fixture providing a mock ConnectionManager for testing. Returns a mock instance that
    simulates the ConnectionManager interface without requiring actual Redis or WebSocket
    connections.
    """
    mock_manager = MagicMock(spec=ConnectionManager)
    return mock_manager


@pytest.fixture
def mock_logger():
    """
    Fixture providing a mock logger for testing log output. Mocks the module-level logger
    to verify logging calls and messages during protocol building operations.
    """
    with patch("app.sockets.websocket_server_protocol.logger") as mock:
        yield mock


@pytest.fixture
def sample_factory(mock_connection_manager):
    """
    Fixture providing a StockTickerServerFactory instance for testing. Creates a factory with
    a mock connection manager and test URL for consistent testing across all test functions.
    """
    url = "ws://localhost:8080"
    factory = StockTickerServerFactory(url, mock_connection_manager)
    return factory


@pytest.fixture
def sample_addresses():
    """
    Fixture providing various address formats for testing. Returns different address types that
    might be passed to buildProtocol to ensure robust handling of various client connection
    scenarios.
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
    Test factory initialization failure with None connection manager. Verifies that passing
    None as connection_manager raises ValueError with appropriate error message and no info
    logging occurs.
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
    Test successful protocol building with valid address.

    Verifies that buildProtocol correctly:
    - Calls parent buildProtocol method
    - Assigns connection_manager to the protocol
    - Logs debug messages for both building and initialization
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
    Test buildProtocol with various address formats. Verifies that the method works correctly
    with different types of client addresses that might be encountered in real usage.
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
    Test that connection_manager is properly assigned to protocol. Verifies the core functionality
    of assigning the factory's connection_manager to the newly created protocol instance.
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
    Test buildProtocol when parent buildProtocol returns None. Verifies that when the parent
    method returns None (connection refused), the method handles it gracefully without attempting
    attribute assignment.
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
    Test buildProtocol when protocol doesn't have connection_manager attribute. Verifies that
    protocols without connection_manager attribute are handled gracefully without attempting
    assignment or raising AttributeError.
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
    Test buildProtocol when protocol already has a connection_manager. Verifies that existing
    connection_manager values are properly overwritten with the factory's connection_manager
    instance.
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
    Test that debug logging uses correct format strings and arguments. Verifies the exact format
    and content of debug log messages to ensure proper monitoring and debugging capabilities.
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
    Test that buildProtocol only uses debug logging, not info/warning/error. Verifies that
    normal protocol building operations don't generate higher-level log messages that might
    clutter logs.
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
    Test that exceptions from parent buildProtocol are properly propagated. Verifies that errors
    in the parent WebSocketServerFactory.buildProtocol method are not caught and suppressed by
    our implementation.
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
    Test that hasattr check safely handles protocols with __getattr__ methods. Verifies that
    the hasattr check doesn't trigger unexpected behavior in protocols that might have custom
    attribute access methods.
    """

    # Create mock protocol with custom __getattr__ that raises exception
    class CustomMagicMock(MagicMock):
        """
        Custom MagicMock that simulates a protocol with a failing __getattr__.
        """

        def __getattr__(self, name):
            """
            Custom __getattr__ that raises an exception for a specific attribute.
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
    Test multiple consecutive calls to buildProtocol. Verifies that the method works correctly
    when called multiple times in succession, which would happen with multiple client connections.
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
    Test buildProtocol with None address. Verifies that None addresses are handled gracefully
    and passed to the parent method without causing issues in our implementation.
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
