# pylint: disable=protected-access
import json
from typing import Any, Dict

import pytest
from pytest_mock import MockerFixture, MockType

from app.sockets.twisted_sockets.uplinksocket import UplinkSocket
from app.utils.common.types.financial_types import DataProviderType

WEBSOCKET_URL = "wss://api.uplink.tech/ws"

#################### Fixtures ####################


@pytest.fixture
def mock_credentials(mocker: MockerFixture) -> MockType:
    """
    Fixture to mock UplinkCredentials.
    """
    return mocker.patch("app.sockets.twisted_sockets.uplinksocket.UplinkCredentials")


@pytest.fixture
def mock_requests(mocker: MockerFixture) -> MockType:
    """
    Fixture to mock requests.
    """
    return mocker.patch("app.sockets.twisted_sockets.uplinksocket.requests")


@pytest.fixture
def mock_logger(mocker: MockerFixture) -> MockType:
    """
    Fixture to mock logger.
    """
    return mocker.patch("app.sockets.twisted_sockets.uplinksocket.logger")


@pytest.fixture
def init_config() -> Dict[str, Any]:
    """
    Fixture to provide initial configuration.
    """
    return {
        "guid": "test_guid",
        "subscription_mode": "ltpc",
        "debug": False,
        "ping_interval": 10,
        "ping_message": "ping",
    }


@pytest.fixture
def token_symbol_map(websocket_instrument_data: list[dict[str, Any]]) -> Dict[str, str]:
    """
    Fixture to provide a token to symbol map.
    """
    return {
        token["token"]: token["symbol"]
        for token in websocket_instrument_data
        if token["data_provider_id"] == DataProviderType.UPLINK.value
    }


def validate_socket_instance(
    uplink_socket: UplinkSocket, init_config: Dict[str, Any]
) -> None:
    """
    Validate the UplinkSocket instance with the given configuration.
    """
    assert uplink_socket is not None
    assert uplink_socket.websocket_url == WEBSOCKET_URL
    assert uplink_socket.guid == init_config["guid"]
    assert uplink_socket.subscription_mode == init_config["subscription_mode"]
    assert uplink_socket.debug == init_config["debug"]
    assert uplink_socket.ping_interval == init_config["ping_interval"]
    assert uplink_socket.ping_message == init_config["ping_message"]


@pytest.fixture
def uplink_socket_instance(
    init_config: Dict[str, Any], mocker: MockType
) -> UplinkSocket:
    """
    Fixture to provide an instance of UplinkSocket.
    """
    return UplinkSocket(
        WEBSOCKET_URL,
        init_config["guid"],
        init_config["subscription_mode"],
        on_data_save_callback=mocker.MagicMock(),
        debug=init_config["debug"],
        ping_interval=init_config["ping_interval"],
        ping_message=init_config["ping_message"],
    )


#################### Tests ####################


# Test: 1
def test_initialize_socket_invalid(
    mock_credentials: MockType, init_config: Dict[str, Any]
) -> None:
    """
    Test initializing socket with invalid credentials.
    """
    # Test: 1.1 ( Access token is missing )
    mock_credentials.get_credentials.return_value = mock_credentials
    mock_credentials.access_token = None
    with pytest.raises(ValueError) as val_err:
        UplinkSocket.initialize_socket(init_config)
    assert str(val_err.value) == "Access token is missing"

    # Test: 1.2 ( Failed to authorize with status code: 401 due to invalid access token )
    mock_credentials.access_token = "dummy_token"
    with pytest.raises(ValueError) as val_err:
        UplinkSocket.initialize_socket(init_config)
    assert str(val_err.value) == "Failed to authorize with status code: 401"


# Test: 2
def test_initialize_socket_valid(
    mock_credentials: MockType,
    init_config: Dict[str, Any],
    mock_requests: MockType,
) -> None:
    """
    Test initializing socket with valid credentials.
    """
    mock_credentials.get_credentials.return_value = mock_credentials
    mock_credentials.access_token = "valid_token"
    mock_requests.get.return_value.status_code = 200
    mock_requests.get.return_value.json.return_value = {
        "data": {"authorized_redirect_uri": WEBSOCKET_URL}
    }

    # Test: 2.1 ( Socket initialization from configuration )
    uplink_socket = UplinkSocket.initialize_socket(init_config)
    validate_socket_instance(uplink_socket, init_config)

    # Test: 2.2 ( Socket initialization from constructor )
    uplink_socket = UplinkSocket(
        WEBSOCKET_URL,
        init_config["guid"],
        init_config["subscription_mode"],
        on_data_save_callback=None,
        debug=init_config["debug"],
        ping_interval=init_config["ping_interval"],
        ping_message=init_config["ping_message"],
    )
    validate_socket_instance(uplink_socket, init_config)


# Test: 3
def test_set_tokens(
    uplink_socket_instance: UplinkSocket, token_symbol_map: Dict[str, str]
) -> None:
    """
    Test setting tokens in the UplinkSocket instance.
    """
    uplink_socket_instance.set_tokens(token_symbol_map)
    assert uplink_socket_instance._tokens == list(token_symbol_map.keys())
    assert uplink_socket_instance.token_map == token_symbol_map


# Test: 4
def test_on_open_with_no_tokens(
    uplink_socket_instance: UplinkSocket,
    mocker: MockType,
    mock_logger: MockType,
) -> None:
    """
    Test on_open method when no tokens are set.
    """
    ws = mocker.MagicMock()
    assert uplink_socket_instance._is_first_connect is True
    uplink_socket_instance._on_open(ws)
    assert ws.send.call_count == 0
    assert uplink_socket_instance._is_first_connect is False
    mock_logger.error.assert_any_call("No valid tokens to subscribe")


# Test: 5
def test_on_open_with_tokens(
    uplink_socket_instance: UplinkSocket,
    token_symbol_map: Dict[str, str],
    mocker: MockType,
    mock_logger: MockType,
) -> None:
    """
    Test on_open method when tokens are set.
    """
    ws = mocker.MagicMock()
    uplink_socket_instance.set_tokens(token_symbol_map)
    assert uplink_socket_instance._is_first_connect is True
    uplink_socket_instance._on_open(ws)
    assert uplink_socket_instance._is_first_connect is False
    mock_logger.error.assert_called_once_with("WebSocket connection is not open")


# Test: 6
def test_subscribe(
    uplink_socket_instance: UplinkSocket,
    mocker: MockType,
    token_symbol_map: Dict[str, str],
    mock_logger: MockType,
) -> None:
    """
    Test subscribing to tokens.
    """
    # Test: 6.1 ( Subscribing with no tokens )
    ws = mocker.MagicMock()
    assert uplink_socket_instance.subscribe([]) is False
    mock_logger.error.assert_called_once_with("No valid tokens to subscribe")
    mock_logger.reset_mock()

    # Test: 6.2 ( Subscribing with valid tokens and no WebSocket connection )
    uplink_socket_instance.set_tokens(token_symbol_map)
    assert uplink_socket_instance.subscribe(list(token_symbol_map.keys())) is False
    mock_logger.error.assert_called_once_with("WebSocket connection is not open")
    assert uplink_socket_instance.subscribed_tokens == {}
    mock_logger.reset_mock()

    # Test: 6.2 ( Subscribing with valid tokens and invalid tokens)
    uplink_socket_instance.ws = ws
    limited_tokens = dict(tuple(token_symbol_map.items())[:2])
    uplink_socket_instance.set_tokens(limited_tokens)
    assert uplink_socket_instance.subscribe(list(token_symbol_map.keys())) is True
    assert uplink_socket_instance.subscribed_tokens == limited_tokens
    mock_logger.error.assert_called_once_with(
        "Tokens not found in token map: %s. Please set tokens using set_tokens method",
        list(token_symbol_map.keys())[2:],
    )
    ws.reset_mock()
    mock_logger.reset_mock()

    # Test: 6.3 ( Subscribing with valid tokens and WebSocket connection )

    uplink_socket_instance.set_tokens(token_symbol_map)
    assert uplink_socket_instance.subscribe(list(token_symbol_map.keys())) is True
    assert uplink_socket_instance.subscribed_tokens == token_symbol_map

    ws.sendMessage.assert_called_once_with(
        json.dumps(
            {
                "guid": uplink_socket_instance.guid,
                "method": "sub",
                "data": {
                    "mode": uplink_socket_instance.subscription_mode,
                    "instrumentKeys": list(token_symbol_map.keys()),
                },
            }
        ).encode("utf-8"),
        isBinary=True,
    )
    mock_logger.error.assert_not_called()

    # Test: 6.4 ( Error handling )
    ws.sendMessage.side_effect = Exception("Error")
    with pytest.raises(Exception) as exc:
        uplink_socket_instance.subscribe(list(token_symbol_map.keys()))
    assert str(exc.value) == "Error"


# Test: 7
def test_unsubscribe(
    uplink_socket_instance: UplinkSocket,
    mocker: MockType,
    token_symbol_map: Dict[str, str],
    mock_logger: MockType,
) -> None:
    """
    Test unsubscribing from tokens.
    """
    # Test: 7.1 ( Unsubscribing with no tokens )
    ws = mocker.MagicMock()
    assert uplink_socket_instance.unsubscribe([]) is False
    mock_logger.error.assert_called_once_with("No tokens to unsubscribe")
    mock_logger.reset_mock()

    # Test: 7.2 ( Unsubscribing with valid tokens and no WebSocket connection )
    uplink_socket_instance.set_tokens(token_symbol_map)
    assert uplink_socket_instance.unsubscribe(list(token_symbol_map.keys())) is False
    mock_logger.error.assert_called_once_with(
        "Tokens not subscribed: %s", list(token_symbol_map.keys())
    )
    assert uplink_socket_instance.subscribed_tokens == {}
    mock_logger.reset_mock()

    # Test: 7.3 ( Unsubscribing with valid tokens and WebSocket connection )
    uplink_socket_instance.ws = ws
    uplink_socket_instance.subscribe(list(token_symbol_map.keys()))
    ws.reset_mock()
    assert uplink_socket_instance.unsubscribe(list(token_symbol_map.keys())) is True
    assert uplink_socket_instance.subscribed_tokens == {}

    ws.sendMessage.assert_called_once_with(
        json.dumps(
            {
                "guid": uplink_socket_instance.guid,
                "method": "unsub",
                "data": {
                    "mode": uplink_socket_instance.subscription_mode,
                    "instrumentKeys": list(token_symbol_map.keys()),
                },
            }
        ).encode("utf-8"),
        isBinary=True,
    )
    mock_logger.error.assert_not_called()

    # Test 7.4 ( Unsubscribing without subscribing )
    assert uplink_socket_instance.unsubscribe(list(token_symbol_map.keys())) is False
    mock_logger.error.assert_called_once_with(
        "Tokens not subscribed: %s", list(token_symbol_map.keys())
    )
    mock_logger.reset_mock()

    # Test: 7.5 ( Error handling )
    uplink_socket_instance.subscribe(list(token_symbol_map.keys()))
    ws.sendMessage.side_effect = Exception("Error")
    with pytest.raises(Exception) as exc:
        uplink_socket_instance.unsubscribe(list(token_symbol_map.keys()))
    assert str(exc.value) == "Error"


# Test: 8
def test_on_message(
    uplink_socket_instance: UplinkSocket,
    mocker: MockType,
    mock_logger: MockType,
    uplink_binary_and_decoded_data: Any,
    uplink_invalid_binary_and_decoded_data: Any,
) -> None:
    """
    Test handling of incoming messages.
    """

    sample_token_map = {"BSE_EQ|INE467B01029": "TCS"}
    ws = mocker.MagicMock()
    uplink_socket_instance.ws = ws
    uplink_socket_instance.set_tokens(sample_token_map)
    uplink_socket_instance.subscribe(list(sample_token_map.keys()))

    # Test: 8.1 ( Handling empty message )
    uplink_socket_instance._on_message(ws, b"", is_binary=True)
    mock_logger.error.assert_not_called()

    # Test: 8.2 ( Handling invalid binary message )
    uplink_socket_instance._on_message(
        ws,
        uplink_invalid_binary_and_decoded_data[0],
        is_binary=True,
    )
    assert uplink_socket_instance.on_data_save_callback is not None
    data_save_args = json.loads(
        uplink_socket_instance.on_data_save_callback.call_args_list[0].args[0]  # type: ignore
    )
    expected_save_data = {
        **uplink_binary_and_decoded_data[1],
        "last_traded_timestamp": -1,
    }
    expected_save_data["last_traded_quantity"] = -1
    for key in expected_save_data:
        assert data_save_args[key] == expected_save_data[key]
    uplink_socket_instance.on_data_save_callback.reset_mock()  # type: ignore

    # Test: 8.3 ( Handling valid binary message )
    uplink_socket_instance.set_tokens(sample_token_map)
    uplink_socket_instance._on_message(
        ws,
        uplink_binary_and_decoded_data[0],
        is_binary=True,
    )
    data_save_args = json.loads(
        uplink_socket_instance.on_data_save_callback.call_args_list[0].args[0]  # type: ignore
    )

    uplink_binary_and_decoded_data[1]["last_traded_quantity"] = -1
    for key in uplink_binary_and_decoded_data[1]:
        assert data_save_args[key] == uplink_binary_and_decoded_data[1][key]

    # Test: 8.4 ( Handling JSON message )
    uplink_socket_instance.set_tokens(sample_token_map)
    data = uplink_invalid_binary_and_decoded_data[1]
    data["feeds"][sample_token_map.popitem()[0]]["ltpc"]["ltt"] = "1740364366158"
    uplink_socket_instance._on_message(
        ws,
        json.dumps(data).encode("utf-8"),
        is_binary=False,
    )
    data_save_args = json.loads(
        uplink_socket_instance.on_data_save_callback.call_args_list[0].args[0]  # type: ignore
    )

    for key in uplink_binary_and_decoded_data[1]:
        assert data_save_args[key] == uplink_binary_and_decoded_data[1][key]
