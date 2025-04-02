# pylint: disable=missing-function-docstring line-too-long unused-argument
"""" 
This module contains tests for the smartapi_crud.py module in the sqlite/crud directory.
"""

from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import OperationalError
from sqlmodel import select

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

#################### TESTS ####################


def validate_pre_upsert_data(upsert_data, model, session):
    """
    Validate data before upsert operation.
    """
    for data in upsert_data:
        prev_data = session.exec(
            select(model).where(model.token == data["token"])
        ).first()
        if prev_data:
            assert prev_data.model_dump() != data


def validate_post_upset_data(upsert_data, model, session):
    """
    Validate data after upsert operation.
    """
    for data in upsert_data:
        result = session.exec(
            select(model).where(model.symbol == data["symbol"])
        ).first()
        assert result is not None
        assert result.model_dump() == data


def validate_pre_insert_or_ignore_data(data_to_insert, model, session):
    """
    Validate data before insert or ignore operation.
    """
    previous_data = []
    for data in data_to_insert:
        prev_data = session.exec(
            select(model).where(model.symbol == data["symbol"])
        ).first()
        previous_data.append(prev_data)
        if prev_data:
            assert prev_data.model_dump() != data
    return previous_data


def validate_post_insert_or_ignore_data(data_to_insert, model, session, previous_data):
    """
    Validate data after insert or ignore operation.
    """
    for idx, data in enumerate(data_to_insert):
        result = session.exec(
            select(model).where(model.symbol == data["symbol"])
        ).first()
        assert result is not None
        if previous_data[idx]:
            assert result.model_dump() == previous_data[idx].model_dump()
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
    model, attributes, expected_exception, expected_message
):
    """
    Test the validate_model_attributes function.
    """
    if expected_exception:
        with pytest.raises(expected_exception) as exc_info:
            validate_model_attributes(model, attributes)
        assert str(exc_info.value.detail) == expected_message
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
def test_get_conditions_list(model, condition_attributes, expected_conditions):
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
    session,
    model,
    condition_attributes,
    expected_result,
    num_results,
    create_insert_sample_data,
):
    """
    Test the get_data_by_any_condition function.
    """
    if expected_result is HTTPException:
        with pytest.raises(expected_result) as exc_info:
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
    session,
    model,
    condition_attributes,
    expected_result,
    num_results,
    create_insert_sample_data,
):
    """
    Test the get_data_by_all_conditions function.
    """
    if expected_result is HTTPException:
        with pytest.raises(expected_result) as exc_info:
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
    session, model, upsert_data, expected_result, create_insert_sample_data
):
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


def test_upsert_with_dummy_session():
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
    session, model, data_to_insert, expected_result, create_insert_sample_data
):
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
    "model, data_size, expected_result",
    [
        (Instrument, INSERTION_BATCH_SIZE * 2 + 100, True),  # Test multiple complete batches plus remainder
        (Instrument, INSERTION_BATCH_SIZE - 1, True),  # Test single incomplete batch
    ],
)
# fmt: on
def test_insert_or_ignore_batch_processing(session, model, data_size, expected_result):
    """
    Test batch processing with large datasets.
    """
    # Generate test data
    data = [
        {
            "token": str(i),
            "symbol": f"SYMBOL_{i}",
            "name": f"Company {i}",
            "instrument_type": "EQ",
            "exchange_id": ExchangeType.NSE.value,
            "data_provider_id": DataProviderType.SMARTAPI.value,
            "expiry_date": "",
            "strike_price": -1.0,
            "tick_size": 5.0,
            "lot_size": 1,
        }
        for i in range(data_size)
    ]

    # Test batch insertion
    _insert_or_ignore(model, data, session=session)

    # Verify all records were inserted
    results = session.exec(select(model)).all()
    assert len(results) == data_size


# fmt: off
@pytest.mark.parametrize(
    "model, data_size, expected_result",
    [
        (Instrument, INSERTION_BATCH_SIZE * 2 + 100, True),  # Test multiple complete batches plus remainder
        (Instrument, INSERTION_BATCH_SIZE - 1, True),  # Test single incomplete batch
    ],
)
# fmt: on
def test_upsert_batch_processing(session, model, data_size, expected_result):
    """
    Test batch processing with large datasets.
    """
    # Generate test data
    data = [
        {
            "token": str(i),
            "symbol": f"SYMBOL_{i}",
            "name": f"Company {i}",
            "instrument_type": "EQ",
            "exchange_id": ExchangeType.NSE.value,
            "data_provider_id": DataProviderType.SMARTAPI.value,
            "expiry_date": "",
            "strike_price": -1.0,
            "tick_size": 5.0,
            "lot_size": 1,
        }
        for i in range(data_size)
    ]

    # Test batch insertion
    _upsert(model, data, session=session)

    # Verify all records were inserted
    results = session.exec(select(model)).all()
    assert len(results) == data_size


def test_insert_or_ignore_with_dummy_session():
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
    session,
    model,
    data_to_insert,
    update_existing,
    expected_result,
    create_insert_sample_data,
):
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

        if update_existing:
            validate_post_upset_data(data_to_insert, model, session)
        else:
            validate_post_insert_or_ignore_data(
                data_to_insert, model, session, previous_data
            )


def test_insert_data_with_dummy_session():
    """
    Test the insert_data function with a dummy session object.
    """
    mock_session = MagicMock()
    mock_session.bind.dialect.name = "mysql"
    with pytest.raises(ValueError):
        insert_data(Instrument, {"token": "123"}, session=mock_session)
