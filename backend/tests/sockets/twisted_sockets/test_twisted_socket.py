# pylint: disable=protected-access
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture, MockType

from app.sockets.twisted_socket import MarketDataTwistedSocket

############################ FIXTURES ############################


@pytest.fixture
def mock_websocket_factory(mocker: MockerFixture):
    """
    Mock the MarketDataWebSocketClientFactory class.
    """
    return mocker.patch("app.sockets.twisted_socket.MarketDataWebSocketClientFactory")


@pytest.fixture
def mock_reactor(mocker: MockerFixture):
    """
    Mock the reactor from twisted.
    """
    return mocker.patch("app.sockets.twisted_socket.reactor")


@pytest.fixture
def mock_websocket_protocol(mocker: MockerFixture):
    """
    Mock the MarketDataWebSocketClientProtocol class.
    """
    return mocker.patch("app.sockets.twisted_socket.MarketDataWebSocketClientProtocol")


@pytest.fixture
def mock_logger(mocker: MockerFixture):
    """
    Mock the logger used in the WebSocket class.
    """
    return mocker.patch("app.sockets.twisted_socket.logger")


@pytest.fixture
def mock_connect_ws(mocker: MockerFixture):
    """
    Mock the connectWS function.
    """
    return mocker.patch("app.sockets.twisted_socket.connectWS")


@pytest.fixture
def mock_threading(mocker: MockerFixture):
    """
    Mock the threading module.
    """
    return mocker.patch("app.sockets.twisted_socket.threading")


class TestWebSocket(MarketDataTwistedSocket):
    """
    Test class for the MarketDataTwistedSocket class.
    """

    def __init__(self):
        super().__init__()
        self.headers = None

    def set_tokens(self, token_data):
        pass

    def _on_message(self, ws, payload, is_binary):
        pass

    def _on_open(self, ws):
        pass

    def subscribe(self, subscription_data):
        pass

    def unsubscribe(self, unsubscribe_data):
        pass

    def resubscribe(self):
        pass


@pytest.fixture
def websocket_instance() -> TestWebSocket:
    """
    Fixture to provide a TestWebSocket instance.
    """
    return TestWebSocket()


############################ TESTS ############################


# Test: 1
def test_initialization(websocket_instance: TestWebSocket):
    """
    Test the initialization values of the WebSocket instance.
    """
    assert websocket_instance.ping_interval == 10
    assert websocket_instance.ping_message == "ping"
    assert websocket_instance.max_retries == 5
    assert websocket_instance.reconnect_max_tries == 5
    assert websocket_instance.reconnect is True
    assert websocket_instance.connection_timeout == 5
    assert websocket_instance.debug is False
    assert websocket_instance.websocket_url is None
    assert websocket_instance.ws is None


# Test: 2
def test_create_connection(
    mock_websocket_factory: MockType, websocket_instance: TestWebSocket
):
    """
    Test the creation of the WebSocket connection and ensure factory setup is correct.
    """
    websocket_instance._create_connection("ws://mock-url")

    assert websocket_instance.factory is not None
    assert websocket_instance.factory == mock_websocket_factory.return_value
    mock_websocket_factory.assert_called_once_with(10, "ping", "ws://mock-url")

    assert websocket_instance.ws == mock_websocket_factory.return_value.ws
    assert websocket_instance.factory.on_connect == websocket_instance._on_connect
    assert websocket_instance.factory.on_open == websocket_instance._on_open
    assert websocket_instance.factory.on_message == websocket_instance._on_message
    assert websocket_instance.factory.on_close == websocket_instance._on_close
    assert websocket_instance.factory.on_reconnect == websocket_instance._on_reconnect
    assert (
        websocket_instance.factory.on_noreconnect == websocket_instance._on_noreconnect
    )
    assert websocket_instance.factory.on_error == websocket_instance._on_error


# Test: 3
def test_connect(
    mock_connect_ws: MockType,
    websocket_instance: TestWebSocket,
    mock_reactor: MockType,
    mock_threading: MockType,
):
    """
    Test the connect functionality with both threaded and non-threaded options.
    """
    websocket_instance.websocket_url = "ws://mock-url"
    websocket_instance.headers = {"Authorization": "Bearer token"}
    mock_reactor.running = False

    # Test: 3.1 ( Non-threaded connection )
    websocket_instance.connect(threaded=False)
    mock_connect_ws.assert_called_once()
    assert websocket_instance.websocket_thread is None
    mock_reactor.run.assert_called_once_with()
    mock_connect_ws.reset_mock()

    # Test: 3.2 ( Threaded connection )
    mock_threading.Thread.return_value = MagicMock()

    websocket_instance.connect(threaded=True)
    mock_threading.Thread.assert_called_once_with(
        target=mock_reactor.run, kwargs={"installSignalHandlers": False}
    )

    mock_connect_ws.assert_called_once()
    assert websocket_instance.websocket_thread is not None
    websocket_instance.websocket_thread.start.assert_called_once()
    mock_threading.rest_mock()


# Test: 4
def test_is_connected(websocket_instance: TestWebSocket):
    """
    Test the is_connected method to check WebSocket connection status.
    """
    websocket_instance.ws = MagicMock()
    websocket_instance.ws.STATE_OPEN = 1
    websocket_instance.ws.STATE_CLOSED = 2

    # Test: 4.1 ( Connected )
    websocket_instance.ws.state = 1
    assert websocket_instance.is_connected() is True

    # Test: 4.2 ( Disconnected )
    websocket_instance.ws.state = 2
    assert websocket_instance.is_connected() is False


# Test: 5
def test_close(websocket_instance: TestWebSocket):
    """
    Test the close method to ensure the WebSocket is properly closed.
    """
    websocket_instance.ws = MagicMock()
    websocket_instance.close(code=1000, reason="Test close")

    assert websocket_instance.ws is None


# Test: 6
def test_on_connect(websocket_instance: TestWebSocket):
    """
    Test the _on_connect method to ensure the on_connect callback is executed.
    """
    ws = MagicMock()
    response = MagicMock()
    websocket_instance.on_connect = MagicMock()
    websocket_instance._on_connect(ws, response)

    websocket_instance.on_connect.assert_called_once_with(ws, response)


# Test: 7
def test_on_close(websocket_instance: TestWebSocket):
    """
    Test the _on_close method to ensure the on_close callback is executed.
    """
    ws = MagicMock()
    websocket_instance.on_close = MagicMock()
    websocket_instance._on_close(ws, 1000, "Test close")

    websocket_instance.on_close.assert_called_once_with(ws, 1000, "Test close")


# Test: 8
def test_on_reconnect(websocket_instance: TestWebSocket):
    """
    Test the _on_reconnect method to ensure the on_reconnect callback is executed.
    """
    websocket_instance.on_reconnect = MagicMock()
    websocket_instance._on_reconnect(3)

    websocket_instance.on_reconnect.assert_called_once_with(websocket_instance, 3)


# Test: 9
def test_stop_retry(websocket_instance: TestWebSocket):
    """
    Test the stop_retry method to ensure retrying is properly stopped.
    """
    websocket_instance.factory = MagicMock()
    websocket_instance.stop_retry()

    # Ensure the stopTrying method is called on the factory
    websocket_instance.factory.stopTrying.assert_called_once()


# Test: 10
def test_stop_reactor(mock_reactor: MockType, websocket_instance: TestWebSocket):
    """
    Test the stop method to ensure the reactor is stopped.
    """
    websocket_instance.stop()
    mock_reactor.stop.assert_called_once()
