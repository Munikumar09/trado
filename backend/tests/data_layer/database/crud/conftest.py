import pytest

from app.data_layer.database.models import Instrument, InstrumentPrice
from app.utils.common.types.financial_types import DataProviderType, ExchangeType


@pytest.fixture
def sample_instrument_data() -> dict[str, str | int | float]:
    """
    Sample instrument data
    """
    return {
        "token": "1594",
        "symbol": "INFY",
        "name": "Infosys Limited",
        "instrument_type": "EQ",
        "exchange_id": ExchangeType.NSE.value,
        "data_provider_id": DataProviderType.SMARTAPI.value,
        "expiry_date": "",
        "strike_price": -1.0,
        "tick_size": 5.0,
        "lot_size": 1,
    }


@pytest.fixture
def sample_instrument(sample_instrument_data) -> Instrument:
    """
    Sample instrument object
    """
    return Instrument(**sample_instrument_data)


@pytest.fixture
def sample_instrument_price_data() -> dict[str, str | int | None]:
    """
    Sample stock price info dictionary
    """
    return {
        "retrieval_timestamp": "2021-09-30 10:00:00",
        "last_traded_timestamp": "2021-09-30 09:59:59",
        "symbol": "INFY",
        "exchange_id": ExchangeType.NSE.value,
        "data_provider_id": DataProviderType.SMARTAPI.value,
        "last_traded_price": "1700.0",
        "last_traded_quantity": "100",
        "average_traded_price": "1700.0",
        "volume_trade_for_the_day": "1000",
        "total_buy_quantity": None,
        "total_sell_quantity": None,
    }


@pytest.fixture
def sample_instrument_price(sample_instrument_price_data) -> InstrumentPrice:
    """
    Sample InstrumentPrice object
    """
    return InstrumentPrice(
        symbol=sample_instrument_price_data["symbol"],
        exchange_id=sample_instrument_price_data["exchange_id"],
        data_provider_id=sample_instrument_price_data["data_provider_id"],
        retrieval_timestamp=sample_instrument_price_data["retrieval_timestamp"],
        last_traded_timestamp=sample_instrument_price_data["last_traded_timestamp"],
        last_traded_price=sample_instrument_price_data["last_traded_price"],
        last_traded_quantity=sample_instrument_price_data["last_traded_quantity"],
        average_traded_price=sample_instrument_price_data["average_traded_price"],
        volume_trade_for_the_day=sample_instrument_price_data[
            "volume_trade_for_the_day"
        ],
        total_buy_quantity="500",
        total_sell_quantity="500",
    )


@pytest.fixture
def create_insert_sample_data(session, sample_instrument, sample_instrument_price):
    """
    Insert sample data into the database
    """
    bse_instrument = Instrument(
        **sample_instrument.model_dump(),
    )
    bse_instrument.exchange_id = ExchangeType.BSE.value
    bse_instrument.token = "1020"

    session.add(bse_instrument)
    session.add(sample_instrument)
    session.add(sample_instrument_price)
    session.commit()
