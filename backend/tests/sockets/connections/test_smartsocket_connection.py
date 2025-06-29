"""
This module contains the tests for the SmartSocketConnection class.
"""

# pylint: disable=missing-function-docstring, redefined-outer-name, no-value-for-parameter, protected-access
from copy import deepcopy
from typing import cast
from unittest.mock import call

import pytest
from omegaconf import OmegaConf
from pytest import MonkeyPatch
from pytest_mock import MockerFixture

from app.data_layer.database.crud.crud_utils import insert_data
from app.data_layer.database.crud.instrument_crud import delete_all_data
from app.data_layer.database.db_connections.sqlite import get_session
from app.data_layer.database.models.instrument_model import Instrument
from app.sockets.connections import SmartSocketConnection
from app.utils.common.types.financial_types import DataProviderType, ExchangeType

#################### Fixtures ####################


@pytest.fixture(autouse=True)
def init_data(
    test_engine, session, websocket_instrument_data, monkeypatch: MonkeyPatch
):
    monkeypatch.setattr(
        "app.data_layer.database.db_connections.postgresql.get_session",
        lambda: get_session(test_engine),
    )
    insert_data(Instrument, websocket_instrument_data)
    yield session


@pytest.fixture
def connection_cfg() -> dict:
    """
    Provides a default configuration dictionary for initializing a SmartSocketConnection in tests.
    
    Returns:
        dict: Configuration parameters for the SmartSocketConnection, including provider, streaming, symbols, exchange type, and connection settings.
    """
    return {
        "name": "smartsocket_connection",
        "provider": {
            "name": "smartsocket",
            "correlation_id": "smart00001",
            "subscription_mode": "snap_quote",
            "debug": False,
        },
        "streaming": {
            "name": "kafka_producer",
            "kafka_topic": "smartsocket",
            "kafka_server": "localhost:23452",
        },
        "symbols": None,
        "exchange_type": "nse",
        "num_connections": 1,
        "current_connection_number": 0,
        "use_thread": True,
        "num_tokens_per_instance": 10,
    }


@pytest.fixture(autouse=True)
def smart_socket_mock(mocker: MockerFixture):
    """
    Automatically mocks the SmartSocket class used in SmartSocketConnection tests.
    
    Returns:
        MagicMock: The patched SmartSocket class mock.
    """
    return mocker.patch("app.sockets.connections.smartsocket_connection.SmartSocket")


@pytest.fixture(autouse=True)
def mock_producer(mocker: MockerFixture):
    """
    Automatically mocks the `init_from_cfg` function in the `smartsocket_connection` module for use in tests.
    
    Returns:
        MagicMock: The mock object replacing `init_from_cfg`.
    """
    return mocker.patch("app.sockets.connections.smartsocket_connection.init_from_cfg")


@pytest.fixture
def connection(
    connection_cfg: dict,
) -> SmartSocketConnection:
    """
    Create and return a SmartSocketConnection instance from the provided configuration dictionary.
    
    Parameters:
        connection_cfg (dict): Configuration parameters for initializing the SmartSocketConnection.
    
    Returns:
        SmartSocketConnection: An instance initialized with the given configuration.
    """
    cfg = OmegaConf.create(connection_cfg)

    return cast(SmartSocketConnection, SmartSocketConnection.from_cfg(cfg))


@pytest.fixture
def expected_tokens(websocket_instrument_data):
    return {
        record["token"]: record["symbol"]
        for record in websocket_instrument_data
        if record["data_provider_id"] == DataProviderType.SMARTAPI.value
    }


#################### Tests ####################


# Test: 1
def test_init_from_cfg_valid_cfg(
    connection_cfg: dict, smart_socket_mock, mock_producer, expected_tokens
):
    """
    Test that SmartSocketConnection initializes correctly from a valid configuration.
    
    Asserts that the returned object is a SmartSocketConnection instance, verifies correct initialization of the internal SmartSocket with the provider config and Kafka producer, and checks that tokens are set as expected.
    """

    cfg = OmegaConf.create(connection_cfg)
    connection = SmartSocketConnection.from_cfg(cfg)

    assert connection is not None
    assert isinstance(connection, SmartSocketConnection)
    smart_socket_mock.initialize_socket.assert_called_once_with(
        cfg.provider, mock_producer.return_value
    )
    assert smart_socket_mock.mock_calls[1] == call.initialize_socket().set_tokens(
        [{"exchangeType": 1, "tokens": expected_tokens}]
    )


# # Test: 2
def test_init_from_cfg_invalid_cfg(
    connection_cfg: dict,
):
    """
    Test the initialization of the SmartSocketConnection object from all the invalid
    configurations.
    """

    # Test: 2.1 ( Number of tokens per instance is 0, meaning no tokens to subscribe to )
    config_cp = deepcopy(connection_cfg)
    config_cp["num_tokens_per_instance"] = 0
    cfg = OmegaConf.create(config_cp)
    connection = SmartSocketConnection.from_cfg(cfg)
    assert connection is None

    # Test: 2.2 ( Invalid symbols )
    config_cp = deepcopy(connection_cfg)
    config_cp["symbols"] = ["FAKE_SYMBOL"]
    cfg = OmegaConf.create(config_cp)
    connection = SmartSocketConnection.from_cfg(cfg)
    assert connection is None
    delete_all_data()

    # Test: 2.3 ( Test invalid exchange type )
    config_cp = deepcopy(connection_cfg)
    config_cp["exchange_type"] = "FAKE_EXCHANGE"
    cfg = OmegaConf.create(config_cp)
    connection = SmartSocketConnection.from_cfg(cfg)
    assert connection is None

    # Test: 2.4 ( No tokens in the database )
    config_cp = deepcopy(connection_cfg)
    cfg = OmegaConf.create(config_cp)
    connection = SmartSocketConnection.from_cfg(cfg)
    assert connection is None


# Test: 3
def test_get_tokens(
    connection: SmartSocketConnection, websocket_instrument_data, expected_tokens
):
    """
    Test the get_tokens method of the SmartSocketConnection class with all
    the possible scenarios.
    """
    # Test: 3.1 ( Test with exchange type as None and symbols as None )
    tokens = connection._get_tokens(symbols=None)  # type: ignore
    assert tokens == expected_tokens

    symbols = [record["symbol"] for record in websocket_instrument_data]

    tokens = connection._get_tokens(symbols, ExchangeType.NSE)
    assert tokens == expected_tokens

    # Test: 3.2 ( Test with lowercase symbols )
    symbols = [symbol.lower() for symbol in symbols]
    tokens = connection._get_tokens(symbols, ExchangeType.NSE)
    assert tokens == expected_tokens

    #  Test: 3.3 ( Test with single symbol )
    token_symbol = tuple(expected_tokens.items())[0]
    tokens = connection._get_tokens(token_symbol[1], ExchangeType.NSE)
    assert tokens == {token_symbol[0]: token_symbol[1]}

    # Test: 3.4 ( Test with invalid symbols )
    stocks = connection._get_tokens(["FAKE_SYMBOL"], ExchangeType.NSE)
    assert stocks == {}
