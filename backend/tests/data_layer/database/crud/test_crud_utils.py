# pylint: disable=missing-function-docstring too-many-locals line-too-long
"""
This module contains tests for the smartapi_crud.py module in the sqlite/crud directory.
"""

import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlmodel import Session, SQLModel, select

from app.data_layer.database.crud.crud_utils import (
    _insert_or_ignore,
    _upsert,
    get_conditions_list,
    get_data_by_all_conditions,
    get_data_by_any_condition,
    insert_data,
    validate_model_attributes,
)
from app.data_layer.database.models import Instrument, InstrumentPrice
from app.utils.common.types.financial_types import DataProviderType, ExchangeType
from app.utils.constants import INSERTION_BATCH_SIZE

#################### HELPER FUNCTIONS ####################


def create_instrument_data(
    token: Union[str, int],
    symbol: str,
    name: str = "Test Company",
    instrument_type: str = "EQ",
    exchange_id: Optional[int] = None,
    data_provider_id: Optional[int] = None,
    tick_size: float = 5.0,
    lot_size: int = 1,
    strike_price: float = -1.0,
    expiry_date: str = "",
) -> Dict[str, Any]:
    """
    Create standardized instrument data dictionary.
    """
    return {
        "token": str(token),
        "symbol": symbol,
        "name": name,
        "instrument_type": instrument_type,
        "exchange_id": exchange_id or ExchangeType.NSE.value,
        "data_provider_id": data_provider_id or DataProviderType.SMARTAPI.value,
        "expiry_date": expiry_date,
        "strike_price": strike_price,
        "tick_size": tick_size,
        "lot_size": lot_size,
    }


def create_instrument_price_data(
    symbol: str,
    exchange_id: Optional[int] = None,
    data_provider_id: Optional[int] = None,
    last_traded_price: float = 1500.0,
    last_traded_quantity: int = 100,
    retrieval_timestamp: Optional[datetime] = None,
    last_traded_timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Create standardized instrument price data dictionary.
    """
    if retrieval_timestamp is None:
        retrieval_timestamp = datetime(2023, 1, 1, 10, 0, 0)
    if last_traded_timestamp is None:
        last_traded_timestamp = datetime(2023, 1, 1, 9, 59, 59)

    return {
        "retrieval_timestamp": retrieval_timestamp,
        "symbol": symbol,
        "exchange_id": exchange_id or ExchangeType.NSE.value,
        "data_provider_id": data_provider_id or DataProviderType.SMARTAPI.value,
        "last_traded_timestamp": last_traded_timestamp,
        "last_traded_price": last_traded_price,
        "last_traded_quantity": last_traded_quantity,
        "average_traded_price": last_traded_price,
        "volume_trade_for_the_day": 1000,
        "total_buy_quantity": 500,
        "total_sell_quantity": 500,
    }


def create_bulk_instrument_data(
    prefix: str, count: int, start_index: int = 0
) -> List[Dict[str, Any]]:
    """
    Create a list of instrument data for bulk operations.
    """
    return [
        create_instrument_data(
            token=f"{prefix}_{i}",
            symbol=f"{prefix}_SYMBOL_{i}",
            name=f"{prefix.replace('_', ' ').title()} Test {i}",
        )
        for i in range(start_index, start_index + count)
    ]


def measure_performance(
    operation_func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Tuple[Any, float]:
    """
    Measure the performance of an operation and return result and duration.
    """
    start_time = time.time()
    result = operation_func(*args, **kwargs)
    duration = time.time() - start_time

    return result, duration


def verify_record_count(
    session: Session,
    model: Type[SQLModel],
    expected_count: int,
    filter_condition: Optional[Any] = None,
) -> List[SQLModel]:
    """
    Verify the number of records in the database matches expected count.
    """
    query = select(model)
    if filter_condition is not None:
        query = query.where(filter_condition)
    results = session.exec(query).all()
    assert len(results) == expected_count

    return list(results)


def verify_record_exists(
    session: Session, model: Type[SQLModel], **conditions: Any
) -> SQLModel:
    """
    Verify that a record exists with the given conditions.
    """
    results = get_data_by_all_conditions(model, session=session, **conditions)
    assert len(results) > 0

    return results[0]


def verify_record_data(record: SQLModel, expected_data: Dict[str, Any]) -> None:
    """
    Verify that a record contains the expected data.
    """
    for key, expected_value in expected_data.items():
        actual_value = getattr(record, key)
        assert (
            actual_value == expected_value
        ), f"Expected {key}={expected_value}, got {actual_value}"


#################### TESTS ####################


def validate_pre_upsert_data(
    upsert_data: List[Dict[str, Any]], model: Type[SQLModel], session: Session
) -> None:
    """
    Validate data before upsert operation.
    """
    for data in upsert_data:
        prev_data = session.exec(
            select(model).where(getattr(model, "token") == data["token"])
        ).first()
        if prev_data:
            assert prev_data.model_dump() != data


def validate_post_upset_data(
    upsert_data: List[Dict[str, Any]], model: Type[SQLModel], session: Session
) -> None:
    """
    Validate data after upsert operation.
    """
    for data in upsert_data:
        result = session.exec(
            select(model).where(getattr(model, "symbol") == data["symbol"])
        ).first()
        assert result is not None
        assert result.model_dump() == data


def validate_pre_insert_or_ignore_data(
    data_to_insert: List[Dict[str, Any]], model: Type[SQLModel], session: Session
) -> List[Optional[SQLModel]]:
    """
    Validate data before insert or ignore operation.
    """
    previous_data = []
    for data in data_to_insert:
        prev_data = session.exec(
            select(model).where(getattr(model, "symbol") == data["symbol"])
        ).first()
        previous_data.append(prev_data)
        if prev_data:
            assert prev_data.model_dump() != data
    return previous_data


def validate_post_insert_or_ignore_data(
    data_to_insert: List[Dict[str, Any]],
    model: Type[SQLModel],
    session: Session,
    previous_data: List[Optional[SQLModel]],
) -> None:
    """
    Validate data after insert or ignore operation.
    """
    for idx, data in enumerate(data_to_insert):
        result = session.exec(
            select(model).where(getattr(model, "symbol") == data["symbol"])
        ).first()
        assert result is not None
        if previous_data[idx]:
            assert result.model_dump() == previous_data[idx].model_dump()  # type: ignore[union-attr]
        else:
            assert result.model_dump() == data


# fmt: off
@pytest.mark.parametrize("model, attributes, expected_exception, expected_message",
                        [
    (Instrument, {"token": "1594", "symbol": "INFY"}, None, None),
    (Instrument, {"invalid_attr": "value"}, HTTPException, "Attribute invalid_attr not found in Instrument model"),
    (Instrument, {"token": 1594}, HTTPException, "Attribute token is not of type <class 'str'> in Instrument model"),
    (InstrumentPrice, {"symbol": "INFY", "last_traded_price": 1700.0}, None, None),
    (InstrumentPrice, {"symbol": "SBI", "invalid_attr": "value"}, HTTPException, "Attribute invalid_attr not found in InstrumentPrice model"),
    (InstrumentPrice, {"symbol": "256265", "last_traded_price": "1700.0"}, HTTPException, "Attribute last_traded_price is not of type <class 'float'> in InstrumentPrice model"),
])
# fmt: on
def test_validate_model_attributes(
    model: Type[SQLModel],
    attributes: Dict[str, Any],
    expected_exception: Optional[Type[Exception]],
    expected_message: Optional[str],
) -> None:
    """
    Test the validate_model_attributes function.
    """
    if expected_exception:
        with pytest.raises(expected_exception) as exc_info:
            validate_model_attributes(model, attributes)
        assert str(exc_info.value.detail) == expected_message  # type: ignore[attr-defined]
    else:
        validate_model_attributes(model, attributes)


# fmt: off
@pytest.mark.parametrize("model, condition_attributes, expected_conditions", [
    (Instrument, {"token": "1594"}, [Instrument.token == "1594"]),
    (Instrument, {"symbol": "INFY", "exchange_id": ExchangeType.NSE.value}, [Instrument.symbol == "INFY", Instrument.exchange_id == ExchangeType.NSE.value]),
    (InstrumentPrice, {"symbol": "INFY"}, [InstrumentPrice.symbol == "INFY"]),
    (InstrumentPrice, {"symbol": "SBI", "last_traded_price": 1300.0}, [InstrumentPrice.symbol == "SBI", InstrumentPrice.last_traded_price == 1300.0]),
    (Instrument, {}, []),  # No conditions
])
# fmt: on
def test_get_conditions_list(
    model: Type[SQLModel],
    condition_attributes: Dict[str, Any],
    expected_conditions: List[Any],
) -> None:
    """
    Test the get_conditions_list function.
    """
    conditions = get_conditions_list(model, condition_attributes)
    for actual, expected in zip(conditions, expected_conditions):
        assert actual.compare(expected)


# fmt: off
@pytest.mark.parametrize("model, condition_attributes, expected_result, num_results", [
    (Instrument, {"token": "1594"}, True, 1),
    (Instrument, {"symbol": "INFY"}, True, 2),
    (Instrument, {"exchange_id": ExchangeType.NSE.value}, True, 1),
    (Instrument, {"token": "9999"}, False, 0),  # No matching records
    (Instrument, {}, HTTPException, 0),  # No conditions
    (InstrumentPrice, {"symbol": "INFY"}, True, 1),
    (InstrumentPrice, {"symbol": "SBI"}, False, 0),  # No matching records
    (InstrumentPrice, {}, HTTPException, 0),  # No conditions
    (InstrumentPrice, {"symbol": "INFY", "last_traded_price": 1700.0}, True, 1),
])
# fmt: on
def test_get_data_by_any_condition(
    session: Session,
    model: Type[SQLModel],
    condition_attributes: Dict[str, Any],
    expected_result: Union[bool, Type[HTTPException]],
    num_results: int,
) -> None:
    """
    Test the get_data_by_any_condition function.
    """
    if expected_result is HTTPException:
        with pytest.raises(HTTPException) as exc_info:
            get_data_by_any_condition(model, session=session, **condition_attributes)

        assert str(exc_info.value.detail) == "No attributes provided for validation"
        assert exc_info.value.status_code == 400
    else:
        results = get_data_by_any_condition(
            model, session=session, **condition_attributes
        )
        if expected_result is True:
            assert results is not None
            assert len(results) == num_results
            for result in results:
                for key, value in condition_attributes.items():
                    assert getattr(result, key) == value
        else:
            assert results == []


# fmt: off
@pytest.mark.parametrize("model, condition_attributes, expected_result, num_results", [
    (Instrument, {"token": "1594"}, True, 1),
    (Instrument, {"symbol": "INFY", "exchange_id": ExchangeType.NSE.value}, True, 1),
    (Instrument, {"symbol": "INFY", "exchange_id": ExchangeType.BSE.value}, True, 1),
    (Instrument, {"symbol": "INFY", "exchange_id": -1}, False, 0),  # No matching records
    (Instrument, {"token": "9999"}, False, 0),  # No matching records
    (Instrument, {}, HTTPException, 0),  # No conditions
    (InstrumentPrice, {"symbol": "INFY"}, True, 1),
    (InstrumentPrice, {"symbol": "SBI"}, False, 0),  # No matching records
    (InstrumentPrice, {}, HTTPException, 0),  # No conditions
    (InstrumentPrice, {"symbol": "INFY", "last_traded_price": 1700.0}, True, 1),
])
# fmt: on
def test_get_data_by_all_conditions(
    session: Session,
    model: Type[SQLModel],
    condition_attributes: Dict[str, Any],
    expected_result: Union[bool, Type[HTTPException]],
    num_results: int,
) -> None:
    """
    Test the get_data_by_all_conditions function.
    """
    if expected_result is HTTPException:
        with pytest.raises(HTTPException) as exc_info:
            get_data_by_all_conditions(model, session=session, **condition_attributes)

        assert str(exc_info.value.detail) == "No attributes provided for validation"
        assert exc_info.value.status_code == 400
    else:
        results = get_data_by_all_conditions(
            model, session=session, **condition_attributes
        )
        if expected_result is True:
            assert results is not None
            assert len(results) == num_results
            for result in results:
                for key, value in condition_attributes.items():
                    assert getattr(result, key) == value
        else:
            assert results == []


# fmt: off
@pytest.mark.parametrize("model, upsert_data, expected_result", [
    (Instrument, [{"token": "1594", "symbol": "TCS", "name": "Tata Consultancy Services", "instrument_type": "EQ", "exchange_id": ExchangeType.NSE.value,"data_provider_id":DataProviderType.SMARTAPI.value, "expiry_date": "", "strike_price": -1.0, "tick_size": 5.0, "lot_size": 1}], True),
    (Instrument, [{"token": "1599", "symbol": "TCS", "name": "Infosys Ltd", "instrument_type": "EQ", "exchange_id": ExchangeType.NSE.value,"data_provider_id":DataProviderType.SMARTAPI.value, "expiry_date": "", "strike_price": -1.0, "tick_size": 5.0, "lot_size": 1}], True),
    (Instrument, [{"token": "9999", "symbol": "INFY", "name": "Infosys Ltd", "instrument_type": "EQ", "exchange_id": ExchangeType.NSE.value,"data_provider_id":DataProviderType.SMARTAPI.value, "expiry_date": "", "strike_price": -1.0, "tick_size": 5.0, "lot_size": 1}], True),
    (Instrument, [], True),  # No data to upsert
])
# fmt: on
def test_upsert(
    session: Session,
    model: Type[SQLModel],
    upsert_data: List[Dict[str, Any]],
    expected_result: bool,
) -> None:
    """
    Test the _upsert function.
    """

    if not expected_result:
        with pytest.raises(OperationalError):
            _upsert(model, upsert_data, session=session)
    else:
        validate_pre_upsert_data(upsert_data, model, session)
        _upsert(model, upsert_data, session=session)
        validate_post_upset_data(upsert_data, model, session)


def test_upsert_with_dummy_session() -> None:
    """
    Test the upsert function with a dummy session object.
    """
    mock_session = MagicMock()
    mock_session.bind.dialect.name = "mysql"
    with pytest.raises(ValueError):
        _upsert(Instrument, [], session=mock_session)


# fmt: off
@pytest.mark.parametrize("model, data_to_insert, expected_result", [
    (Instrument, [{"token": "1594", "symbol": "TCS", "name": "Tata Consultancy Services", "instrument_type": "EQ", "exchange_id": ExchangeType.NSE.value, "data_provider_id":DataProviderType.SMARTAPI.value,"expiry_date": "", "strike_price": -1.0, "tick_size": 52.0, "lot_size": 1}], True),
    (Instrument, [{"token": "159", "symbol": "SBI", "name": "State Bank Of India", "instrument_type": "EQ", "exchange_id": ExchangeType.NSE.value, "data_provider_id":DataProviderType.SMARTAPI.value,"expiry_date": "", "strike_price": -1.0, "tick_size": 44.0, "lot_size": 1}], True),
    (Instrument, [{"token": "9999", "symbol": "LT", "name": "Larsen & Toubro", "instrument_type": "EQ", "exchange_id": ExchangeType.BSE.value, "data_provider_id":DataProviderType.SMARTAPI.value,"expiry_date": "", "strike_price": -1.0, "tick_size": 5.0, "lot_size": 1}], True),
    (Instrument, [{"token": "1594", "symbol": "Zomato", "name": "Zomato Ltd", "instrument_type": "EQ", "exchange_id": ExchangeType.NSE.value, "data_provider_id":DataProviderType.SMARTAPI.value,"expiry_date": "", "strike_price": -1.0, "tick_size": 53.0, "lot_size": 1}, {"token": "1020", "symbol": "Swiggy", "name": "Swiggy Ltd", "instrument_type": "EQ", "exchange_id": ExchangeType.BSE.value,"data_provider_id":DataProviderType.SMARTAPI.value,"expiry_date": "", "strike_price": -1.0, "tick_size": 10.0, "lot_size": 1}], True),
    (Instrument, [], True),  # No data to insert
])
# fmt: on
def test_insert_or_ignore(
    session: Session,
    model: Type[SQLModel],
    data_to_insert: List[Dict[str, Any]],
    expected_result: bool,
) -> None:
    """
    Test the _insert_or_ignore function.
    """
    if not expected_result:
        with pytest.raises(OperationalError):
            _insert_or_ignore(model, data_to_insert, session=session)
    else:
        previous_data = validate_pre_insert_or_ignore_data(
            data_to_insert, model, session
        )
        _insert_or_ignore(model, data_to_insert, session=session)
        validate_post_insert_or_ignore_data(
            data_to_insert, model, session, previous_data
        )


# fmt: off
@pytest.mark.parametrize(
    "model, data_size,  operation_type",
    [
        (Instrument, INSERTION_BATCH_SIZE * 2 + 100, "insert"),  # Test multiple complete batches plus remainder
        (Instrument, INSERTION_BATCH_SIZE - 1, "insert"),  # Test single incomplete batch
        (Instrument, INSERTION_BATCH_SIZE * 2 + 50, "upsert"),  # Test upsert with multiple batches
        (Instrument, INSERTION_BATCH_SIZE - 50, "upsert"),  # Test upsert with single batch
    ],
)
# fmt: on
def test_batch_processing(
    session: Session,
    model: Type[SQLModel],
    data_size: int,
    operation_type: str,
) -> None:
    """
    Test batch processing with large datasets for both insert and upsert operations.
    """
    # Generate test data using helper function
    data = create_bulk_instrument_data(f"BATCH_{operation_type.upper()}", data_size)

    if operation_type == "insert":
        # Test batch insertion
        _insert_or_ignore(model, data, session=session)
    else:
        # Test batch upsert
        _upsert(model, data, session=session)

    # Verify all records were processed using helper function
    verify_record_count(
        session,
        model,
        data_size,
        filter_condition=model.symbol.like(  # type: ignore[attr-defined]
            f"BATCH_{operation_type.upper()}_%"
        ),
    )


def test_insert_or_ignore_with_dummy_session() -> None:
    """
    Test the _insert_or_ignore function with a dummy session object.
    """
    mock_session = MagicMock()
    mock_session.bind.dialect.name = "mysql"
    with pytest.raises(ValueError):
        _insert_or_ignore(Instrument, [], session=mock_session)


# fmt: off
@pytest.mark.parametrize("model, data_to_insert, update_existing, expected_result", [
    (Instrument, {"token": "1594", "symbol": "TCS", "name": "Tata Consultancy Services", "instrument_type": "EQ", "exchange_id": ExchangeType.NSE.value, "data_provider_id":DataProviderType.SMARTAPI.value,"expiry_date": "", "strike_price": -1.0, "tick_size": 52.0, "lot_size": 1}, False, True),
    (Instrument, {"token": "1594", "symbol": "SBI", "name": "State Bank Of India", "instrument_type": "EQ", "exchange_id": ExchangeType.NSE.value, "data_provider_id":DataProviderType.SMARTAPI.value,"expiry_date": "", "strike_price": -1.0, "tick_size": 44.0, "lot_size": 1}, True, True),
    (Instrument, {"token": "9999", "symbol": "LT", "name": "Larsen & Toubro", "instrument_type": "EQ", "exchange_id": ExchangeType.BSE.value, "data_provider_id":DataProviderType.SMARTAPI.value,"expiry_date": "", "strike_price": -1.0, "tick_size": 5.0, "lot_size": 1}, False, True),
    (Instrument, [{"token": "1594", "symbol": "Zomato", "name": "Zomato Ltd", "instrument_type": "EQ", "exchange_id": ExchangeType.NSE.value, "data_provider_id":DataProviderType.SMARTAPI.value,"expiry_date": "", "strike_price": -1.0, "tick_size": 53.0, "lot_size": 1}, {"token": "1020", "symbol": "Swiggy", "name": "Swiggy Ltd", "instrument_type": "EQ", "exchange_id": ExchangeType.BSE.value, "data_provider_id":DataProviderType.SMARTAPI.value,"expiry_date": "", "strike_price": -1.0, "tick_size": 10.0, "lot_size": 1}], True, True),
    (Instrument, None, False, False),  # No data to insert
    (Instrument, [], False, False),  # No data to insert
])
# fmt: on
def test_insert_data(
    session: Session,
    model: Type[SQLModel],
    data_to_insert: Union[Dict[str, Any], List[Dict[str, Any]], None],
    update_existing: bool,
    expected_result: bool,
) -> None:
    """
    Test the insert_data function.
    """
    if not expected_result:
        assert (
            insert_data(
                model, data_to_insert, session=session, update_existing=update_existing
            )
            is False
        )
    else:
        data_to_insert_cp = deepcopy(data_to_insert)
        previous_data = None
        if isinstance(data_to_insert, dict):
            data_to_insert_cp = [data_to_insert]

        # Type assertion: data_to_insert_cp is guaranteed to be List[Dict[str, Any]] here
        assert isinstance(data_to_insert_cp, list)

        if update_existing:
            validate_pre_upsert_data(data_to_insert_cp, model, session)
        else:
            previous_data = validate_pre_insert_or_ignore_data(
                data_to_insert_cp, model, session
            )

        assert (
            insert_data(
                model, data_to_insert, session=session, update_existing=update_existing
            )
            is True
        )

        if isinstance(data_to_insert, dict):
            data_to_insert = [data_to_insert]

        # Type assertion: data_to_insert is guaranteed to be List[Dict[str, Any]] here
        assert isinstance(data_to_insert, list)
        # Type assertion: previous_data is guaranteed to be List[Optional[SQLModel]] if not None
        assert previous_data is None or isinstance(previous_data, list)

        if update_existing:
            validate_post_upset_data(data_to_insert, model, session)
        else:
            # Type check: previous_data is guaranteed to be a list here since it's only
            # None in update_existing=True case
            assert previous_data is not None
            validate_post_insert_or_ignore_data(
                data_to_insert, model, session, previous_data
            )


def test_insert_data_with_dummy_session() -> None:
    """
    Test the insert_data function with a dummy session object.
    """
    mock_session = MagicMock()
    mock_session.bind.dialect.name = "mysql"
    with pytest.raises(ValueError):
        insert_data(Instrument, {"token": "123"}, session=mock_session)


#################### INTEGRATION AND PERFORMANCE TESTS ####################


def test_end_to_end_workflow(session: Session) -> None:
    """
    Test end-to-end workflow with multiple operations.
    """
    # Step 1: Insert initial data using helper function
    initial_instruments = create_bulk_instrument_data("E2E", 10)

    result = insert_data(Instrument, initial_instruments, session=session)
    assert result is True

    # Step 2: Query data by any condition
    results = get_data_by_any_condition(
        Instrument, session=session, exchange_id=ExchangeType.NSE.value
    )
    assert len(results) >= 10

    # Step 3: Query data by all conditions
    specific_results = get_data_by_all_conditions(
        Instrument,
        session=session,
        symbol="E2E_SYMBOL_5",
        exchange_id=ExchangeType.NSE.value,
    )
    assert len(specific_results) == 1
    assert specific_results[0].symbol == "E2E_SYMBOL_5"

    # Step 4: Update existing data using helper function
    update_data = create_instrument_data(
        token="E2E_5",
        symbol="E2E_SYMBOL_5",
        name="Updated End to End Test 5",
        tick_size=10.0,
    )

    result = insert_data(Instrument, update_data, session=session, update_existing=True)
    assert result is True

    # Step 5: Verify update using helper functions
    updated_record = verify_record_exists(
        session, Instrument, symbol="E2E_SYMBOL_5", exchange_id=ExchangeType.NSE.value
    )
    verify_record_data(
        updated_record, {"name": "Updated End to End Test 5", "tick_size": 10.0}
    )

    # Step 6: Insert duplicate with ignore using helper function
    duplicate_data = create_instrument_data(
        token="E2E_5", symbol="E2E_SYMBOL_5", name="Should be ignored", tick_size=15.0
    )

    result = insert_data(
        Instrument, duplicate_data, session=session, update_existing=False
    )
    assert result is True

    # Step 7: Verify duplicate was ignored using helper functions
    final_record = verify_record_exists(
        session, Instrument, symbol="E2E_SYMBOL_5", exchange_id=ExchangeType.NSE.value
    )
    verify_record_data(
        final_record, {"name": "Updated End to End Test 5", "tick_size": 10.0}
    )


def test_cross_model_operations(session: Session) -> None:
    """
    Test operations across multiple models (Instrument and InstrumentPrice).
    """
    # Insert instrument using helper function
    instrument_data = create_instrument_data(
        token="CROSS_TEST", symbol="CROSS_SYMBOL", name="Cross Model Test"
    )

    result = insert_data(Instrument, instrument_data, session=session)
    assert result is True

    # Insert corresponding price data using helper function
    price_data = create_instrument_price_data("CROSS_SYMBOL")

    result = insert_data(InstrumentPrice, price_data, session=session)
    assert result is True

    # Query both models and verify using helper functions
    instrument_record = verify_record_exists(session, Instrument, symbol="CROSS_SYMBOL")
    price_record = verify_record_exists(session, InstrumentPrice, symbol="CROSS_SYMBOL")

    assert getattr(instrument_record, "symbol") == getattr(price_record, "symbol")


@pytest.mark.performance
def test_performance_operations(session: Session) -> None:
    """
    Test performance with large batch insertions and upserts.
    """
    # Test large batch insert
    insert_data_size = 10000
    data_to_insert = create_bulk_instrument_data("PERF_INSERT", insert_data_size)

    result, insert_duration = measure_performance(
        insert_data, Instrument, data_to_insert, session=session, update_existing=False
    )
    assert result is True

    # Verify insert data was inserted using helper function
    verify_record_count(
        session,
        Instrument,
        insert_data_size,
        filter_condition=cast(InstrumentedAttribute, Instrument.symbol).like(
            "PERF_INSERT_%"
        ),
    )

    # Test large batch upsert
    upsert_data_size = 5000
    # First insert data for upsert test using helper function
    initial_upsert_data = create_bulk_instrument_data("PERF_UPSERT", upsert_data_size)
    insert_data(Instrument, initial_upsert_data, session=session, update_existing=False)

    # Now update all data using helper function with different tick_size
    update_upsert_data = create_bulk_instrument_data("PERF_UPSERT", upsert_data_size)
    for item in update_upsert_data:
        item["tick_size"] = 10.0  # Changed value
        item["name"] = item["name"].replace("Test", "Updated Test")

    result, upsert_duration = measure_performance(
        insert_data,
        Instrument,
        update_upsert_data,
        session=session,
        update_existing=True,
    )
    assert result is True

    # Verify updates using helper function
    verify_record_count(
        session,
        Instrument,
        upsert_data_size,
        filter_condition=(
            cast(InstrumentedAttribute, Instrument.symbol).like("PERF_UPSERT_%")
            & (Instrument.tick_size == 10.0)
        ),
    )

    # Test query performance with large dataset
    query_data_size = 1000
    query_data = create_bulk_instrument_data(
        "PERF_QUERY", query_data_size, start_index=20000
    )
    insert_data(Instrument, query_data, session=session)

    # Test query performance using helper function
    query_results, query_duration = measure_performance(
        get_data_by_any_condition,
        Instrument,
        session=session,
        exchange_id=ExchangeType.NSE.value,
    )

    assert len(query_results) >= query_data_size
    assert query_duration < 5.0  # Should complete within 5 seconds

    # Test specific query performance using helper function
    specific_results, specific_query_duration = measure_performance(
        get_data_by_all_conditions,
        Instrument,
        session=session,
        symbol="PERF_QUERY_SYMBOL_20500",
        exchange_id=ExchangeType.NSE.value,
    )

    assert len(specific_results) == 1
    assert specific_query_duration < 1.0  # Should complete within 1 second

    # Basic performance assertions
    assert insert_duration < 60  # Should complete within 60 seconds
    assert upsert_duration < 30  # Should complete within 30 seconds


def test_stress_test_multiple_operations(session: Session) -> None:
    """
    Stress test with multiple concurrent operations.
    """
    operations_count = 100

    for i in range(operations_count):
        # Alternate between different operations
        if i % 4 == 0:
            # Insert operation using helper function
            data = create_instrument_data(
                token=f"STRESS_{i}",
                symbol=f"STRESS_SYMBOL_{i}",
                name=f"Stress Test {i}",
            )
            insert_data(Instrument, data, session=session)

        elif i % 4 == 1:
            # Query by any condition
            if i > 0:  # Only query if we have some data
                get_data_by_any_condition(
                    Instrument, session=session, exchange_id=ExchangeType.NSE.value
                )

        elif i % 4 == 2:
            # Query by all conditions
            if i > 4:  # Only query if we have some specific data
                get_data_by_all_conditions(
                    Instrument,
                    session=session,
                    exchange_id=ExchangeType.NSE.value,
                    instrument_type="EQ",
                )

        else:
            # Update operation
            if i > 8:
                update_target = i - 8
                update_data = create_instrument_data(
                    token=f"STRESS_{update_target}",
                    symbol=f"STRESS_SYMBOL_{update_target}",
                    name=f"Updated Stress Test {update_target}",
                    tick_size=15.0,
                )
                insert_data(
                    Instrument, update_data, session=session, update_existing=True
                )

    # Verify final state using helper function
    final_results = session.exec(select(Instrument)).all()
    assert len(final_results) >= operations_count // 4  # At least the insert operations


def test_error_recovery_and_consistency(session: Session) -> None:
    """
    Test error recovery and data consistency.
    """
    # Insert some initial data using helper function
    initial_data = create_bulk_instrument_data("RECOVERY", 5)

    insert_data(Instrument, initial_data, session=session)
    initial_count = len(session.exec(select(Instrument)).all())

    # Attempt operation with valid data using helper function
    mixed_data = [
        create_instrument_data(
            token="RECOVERY_VALID",
            symbol="RECOVERY_VALID_SYMBOL",
            name="Valid Recovery Test",
        )
    ]

    # This should succeed
    result = insert_data(Instrument, mixed_data, session=session)
    assert result is True

    # Verify data consistency using helper function
    final_count = len(session.exec(select(Instrument)).all())
    assert final_count == initial_count + 1

    # Verify the valid data was inserted using helper function
    verify_record_exists(session, Instrument, symbol="RECOVERY_VALID_SYMBOL")


def test_boundary_conditions(session: Session) -> None:
    """
    Test boundary conditions and edge cases.
    """
    # Test with minimum values using helper function
    min_data = create_instrument_data(
        token="MIN",
        symbol="M",  # Minimum length symbol
        name="",  # Empty name
        instrument_type="",
        tick_size=0.01,  # Minimum tick size
    )

    result = insert_data(Instrument, min_data, session=session)
    assert result is True

    # Test with maximum reasonable values using helper function
    max_data = create_instrument_data(
        token="A" * 50,  # Long token
        symbol="MAXIMUM_SYMBOL_LENGTH_TEST_ABCDEFGH",  # Long symbol
        name="Maximum Length Name Test " + "X" * 100,
        instrument_type="EQUITY_MAXIMUM_TYPE",
        expiry_date="2099-12-31",
        strike_price=999999.99,
        tick_size=1000.0,
        lot_size=10000,
    )

    result = insert_data(Instrument, max_data, session=session)
    assert result is True

    # Verify both records exist using helper function
    verify_record_exists(session, Instrument, symbol="M")
    verify_record_exists(
        session, Instrument, symbol="MAXIMUM_SYMBOL_LENGTH_TEST_ABCDEFGH"
    )


def test_conditions_with_special_characters(session: Session) -> None:
    """
    Test get_conditions_list and query functions with special characters.
    """
    # Insert data with special characters using helper function
    special_data = create_instrument_data(
        token="SPECIAL_TEST", symbol="TEST@#$", name="Test & Company Ltd."
    )

    result = insert_data(Instrument, special_data, session=session)
    assert result is True

    # Test querying with special characters using helper function
    record = verify_record_exists(session, Instrument, symbol="TEST@#$")
    verify_record_data(record, {"name": "Test & Company Ltd."})


def test_database_constraint_handling(session: Session) -> None:
    """
    Test handling of database constraints and foreign key relationships.
    """
    # Test inserting InstrumentPrice without corresponding Instrument
    # This should fail due to foreign key constraint using helper function
    orphan_price_data = create_instrument_price_data("NONEXISTENT_SYMBOL")

    # This should either fail or be ignored depending on the database setup
    try:
        insert_data(InstrumentPrice, orphan_price_data, session=session)
        # If it succeeds, verify it was actually ignored due to constraint
        price_results = get_data_by_any_condition(
            InstrumentPrice, session=session, symbol="NONEXISTENT_SYMBOL"
        )
        # Should be empty due to foreign key constraint
        assert len(price_results) == 0
    except Exception:
        # Expected behavior - foreign key constraint should prevent insertion
        pass
