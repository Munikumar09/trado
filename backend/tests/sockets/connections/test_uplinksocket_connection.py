"""
This module contains the tests for the UplinkSocketConnection class.
"""

# pylint: disable=missing-function-docstring, redefined-outer-name no-value-for-parameter, protected-access
from copy import deepcopy
from typing import Any, Dict, cast
from unittest.mock import call

import pytest
from omegaconf import OmegaConf
from pytest import MonkeyPatch
from pytest_mock import MockerFixture

from app.data_layer.database.crud.crud_utils import insert_data
from app.data_layer.database.crud.instrument_crud import delete_all_data
from app.data_layer.database.db_connections.sqlite import get_session
from app.data_layer.database.models.instrument_model import Instrument
from app.sockets.connections import UplinkSocketConnection
from app.utils.common.types.financial_types import DataProviderType, ExchangeType

#################### Fixtures ####################


@pytest.fixture(autouse=True)
def init_data(
    test_engine: Any,
    websocket_instrument_data: Dict[str, Any],
    monkeypatch: MonkeyPatch,
):
    monkeypatch.setattr(
        "app.data_layer.database.db_connections.postgresql.get_session",
        lambda: get_session(test_engine),
    )
    insert_data(Instrument, websocket_instrument_data)


@pytest.fixture
def connection_cfg() -> Dict[str, Any]:
    return {
        "name": "uplinksocket_connection",
        "provider": {
            "correlation_id": "uplink0001",
            "subscription_mode": "ltpc",
            "debug": False,
        },
        "streaming": {
            "name": "kafka",
            "kafka_topic": "smartsocket",
            "kafka_server": "localhost:23452",
        },
        "symbols": None,
        "num_connections": 1,
        "exchange_type": "bse",
        "current_connection_number": 0,
        "use_thread": True,
        "num_tokens_per_instance": 10,
    }


@pytest.fixture(autouse=True)
def smart_socket_mock(mocker: MockerFixture) -> Any:
    return mocker.patch("app.sockets.connections.uplinksocket_connection.UplinkSocket")


@pytest.fixture
def connection(
    connection_cfg: Dict[str, Any],
) -> UplinkSocketConnection:
    cfg = OmegaConf.create(connection_cfg)

    return cast(UplinkSocketConnection, UplinkSocketConnection.from_cfg(cfg))


@pytest.fixture
def expected_tokens(websocket_instrument_data: list[dict[str, Any]]) -> Dict[int, str]:
    return {
        record["token"]: record["symbol"]
        for record in websocket_instrument_data
        if record["data_provider_id"] == DataProviderType.UPLINK.value
    }


#################### Tests ####################


# Test: 1
def test_init_from_cfg_valid_cfg(
    connection_cfg: Dict[str, Any],
    smart_socket_mock: Any,
    expected_tokens: Dict[int, str],
) -> None:
    """
    Test the initialization of the UplinkSocketConnection object from a valid configuration.
    """

    cfg = OmegaConf.create(connection_cfg)
    connection = UplinkSocketConnection.from_cfg(cfg)

    assert connection is not None
    assert isinstance(connection, UplinkSocketConnection)
    smart_socket_mock.initialize_socket.assert_called_once_with(cfg.provider, None)
    assert smart_socket_mock.mock_calls[1] == call.initialize_socket().set_tokens(
        expected_tokens
    )


# Test: 2
def test_init_from_cfg_invalid_cfg(
    connection_cfg: Dict[str, Any],
) -> None:
    """
    Test the initialization of the UplinkSocketConnection object from all the invalid
    configurations.
    """

    # Test: 2.1 ( Number of tokens per instance is 0, meaning no tokens to subscribe to )
    config_cp = deepcopy(connection_cfg)
    config_cp["num_tokens_per_instance"] = 0
    cfg = OmegaConf.create(config_cp)
    connection = UplinkSocketConnection.from_cfg(cfg)
    assert connection is None

    # Test: 2.2 ( Invalid symbols )
    config_cp = deepcopy(connection_cfg)
    config_cp["symbols"] = ["FAKE_SYMBOL"]
    cfg = OmegaConf.create(config_cp)
    connection = UplinkSocketConnection.from_cfg(cfg)
    assert connection is None

    # Test: 2.3 ( Invalid exchange type )
    config_cp = deepcopy(connection_cfg)
    config_cp["exchange_type"] = "FAKE_EXCHANGE"
    cfg = OmegaConf.create(config_cp)
    connection = UplinkSocketConnection.from_cfg(cfg)
    assert connection is None

    delete_all_data()
    # Test: 2.4 ( No tokens in the database )
    config_cp = deepcopy(connection_cfg)
    cfg = OmegaConf.create(config_cp)
    connection = UplinkSocketConnection.from_cfg(cfg)
    assert connection is None


# Test: 3
def test_get_tokens(
    connection: UplinkSocketConnection, websocket_instrument_data, expected_tokens
):
    """
    Test the get_tokens method of the UplinkSocketConnection class with all
    the possible scenarios.
    """
    # Test: 3.1 ( Test with exchange type as None and symbols as None )
    tokens = connection._get_tokens(symbols=None)  # type: ignore
    assert tokens == expected_tokens

    symbols = [record["symbol"] for record in websocket_instrument_data]

    tokens = connection._get_tokens(symbols, ExchangeType.BSE)
    assert tokens == expected_tokens

    # Test: 3.2 ( Test with lowercase symbols )
    symbols = [symbol.lower() for symbol in symbols]
    tokens = connection._get_tokens(symbols, ExchangeType.BSE)
    assert tokens == expected_tokens

    #  Test: 3.3 ( Test with single symbol )
    token_symbol = tuple(expected_tokens.items())[0]
    tokens = connection._get_tokens(token_symbol[1], ExchangeType.BSE)
    assert tokens == {token_symbol[0]: token_symbol[1]}

    # Test: 3.4 ( Test with invalid symbols )
    stocks = connection._get_tokens(["FAKE_SYMBOL"], ExchangeType.BSE)
    assert stocks == {}
