from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    ForeignKey,
    ForeignKeyConstraint,
    PrimaryKeyConstraint,
    SmallInteger,
)
from sqlmodel import TIMESTAMP, Field, SQLModel, text


class Exchange(SQLModel, table=True):  # type: ignore
    """
    This class holds the information about the exchanges.

    Attributes
    ----------
    id: ``int``
        The unique identifier of the exchange
        Eg: 1
    symbol: ``str``
        The symbol of the exchange
        Eg: "NSE"
    """

    id: int = Field(sa_column=Column(SmallInteger(), primary_key=True))
    symbol: str = Field(min_length=3, max_length=10)

    def to_dict(self):
        """
        Returns the object as a dictionary.
        """
        return {
            "id": self.id,
            "symbol": self.symbol,
        }


class DataProvider(SQLModel, table=True):  # type: ignore
    """
    This class holds the information about the stock data providers.

    Attributes
    ----------
    id: ``int``
        The unique identifier of the data provider
        Eg: 1
    name: ``str``
        The name of the data provider
        Eg: "SMARTAPI"
    """

    id: int = Field(sa_column=Column(SmallInteger(), primary_key=True))
    name: str = Field(min_length=3, max_length=30)
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            TIMESTAMP(timezone=True),
            server_default=text("CURRENT_TIMESTAMP"),
            onupdate=text("CURRENT_TIMESTAMP"),
        ),
    )

    def to_dict(self):
        """
        Returns the object as a dictionary.
        """
        return {
            "id": self.id,
            "name": self.name,
            "last_updated": self.last_updated.replace(tzinfo=None),
        }


class Instrument(SQLModel, table=True):  # type: ignore
    """
    This class holds the information about financial instruments.

    Attributes
    ----------
    symbol: ``str``
        The unique symbol of the financial instrument
        Eg: "RELIANCE"
    exchange_id: ``int``
        The unique identifier of the exchange
        Eg: 1
    data_provider_id: ``int``
        The unique identifier of the data provider
        Eg: 1
    token: ``str``
        The unique token of the financial instrument
        Eg: "12313533"
    name: ``str``
        The name of the financial instrument
        Eg: "Reliance Industries Ltd"
    instrument_type: ``str``
        The type of the financial instrument
        Eg: "EQ" (Equity)
    expiry_date: ``str``
        Expiry date is the date on which the option contract expires
        Eg: "2021-12-31"
    strike_price: ``float``
        Strike price is the price at which the option holder can buy or sell the underlying asset
        Eg: 2000.0
    lot_size: ``int``
        Lot size means the minimum quantity of a trading instrument that can be traded
        Eg: 25
    tick_size: ``float``
        Tick size means the minimum price movement of a trading instrument
        Eg: 0.05
    """

    symbol: str = Field(max_length=40)
    exchange_id: int = Field(
        sa_column=Column(SmallInteger(), ForeignKey("exchange.id"), nullable=False)
    )
    data_provider_id: int = Field(
        sa_column=Column(SmallInteger(), ForeignKey("dataprovider.id"), nullable=False)
    )
    token: str
    name: str
    instrument_type: str
    expiry_date: str | None = None
    strike_price: float | None = None
    lot_size: int | None = None
    tick_size: float | None = None

    # Add a composite primary key and unique constraint
    __table_args__ = (
        PrimaryKeyConstraint("symbol", "exchange_id", "data_provider_id"),
    )

    def to_dict(self):
        """
        Returns the object as a dictionary.
        """
        return {
            "symbol": self.symbol,
            "exchange_id": self.exchange_id,
            "data_provider_id": self.data_provider_id,
            "token": self.token,
            "name": self.name,
            "instrument_type": self.instrument_type,
            "expiry_date": self.expiry_date,
            "strike_price": self.strike_price,
            "lot_size": self.lot_size,
            "tick_size": self.tick_size,
        }


class InstrumentPrice(SQLModel, table=True):  # type: ignore
    """
    This class holds the price information of the financial instrument given by the data provider.

    Attributes
    ----------
    retrieval_timestamp: ``datetime``
        The timestamp at which the data was retrieved
        Eg: "2021-09-01 09:00:00"
    symbol: ``str``
        The unique symbol of the financial instrument
        Eg: "RELIANCE"
    exchange_id: ``int``
        The unique identifier of the exchange
        Eg: 1
    data_provider_id: ``int``
        The unique identifier of the data provider
        Eg: 1
    last_traded_timestamp: ``datetime``
        The timestamp at which the last trade was made
        Eg: "2021-09-01 09:00:00"
    last_traded_price: ``float``
        The price at which the last trade was made at the given timestamp
        Eg: 2000.0
    last_traded_quantity: ``int``
        The quantity of the last trade
        Eg: 100
    average_traded_price: ``float``
        The average traded price for the day
        Eg: 2000.0
    volume_trade_for_the_day: ``int``
        The total volume traded for the day
        Eg: 10000
    total_buy_quantity: ``int``
        The total buy quantity till the given timestamp
        Eg: 5000
    total_sell_quantity: ``int``
        The total sell quantity till the given timestamp
        Eg: 5000
    """

    retrieval_timestamp: datetime
    symbol: str
    exchange_id: int
    data_provider_id: int
    last_traded_timestamp: datetime
    last_traded_price: float = Field(
        ge=-1
    )  # -1 indicates the value is missing in the data
    last_traded_quantity: int | None = Field(default=None, ge=-1)
    average_traded_price: float | None = Field(default=None, ge=-1)
    volume_trade_for_the_day: int | None = Field(default=None, ge=-1)
    total_buy_quantity: int | None = Field(default=None, ge=-1)
    total_sell_quantity: int | None = Field(default=None, ge=-1)

    # Add foreign key constraint referencing the composite primary key of Instrument
    __table_args__ = (
        PrimaryKeyConstraint(
            "symbol", "exchange_id", "data_provider_id", "retrieval_timestamp"
        ),
        ForeignKeyConstraint(
            ["symbol", "exchange_id", "data_provider_id"],
            [
                "instrument.symbol",
                "instrument.exchange_id",
                "instrument.data_provider_id",
            ],
        ),
    )

    def to_dict(self):
        """
        Returns the object as a dictionary
        """
        return {
            "retrieval_timestamp": self.retrieval_timestamp.replace(tzinfo=None),
            "symbol": self.symbol,
            "exchange_id": self.exchange_id,
            "data_provider_id": self.data_provider_id,
            "last_traded_timestamp": self.last_traded_timestamp.replace(tzinfo=None),
            "last_traded_price": self.last_traded_price,
            "last_traded_quantity": self.last_traded_quantity,
            "average_traded_price": self.average_traded_price,
            "volume_trade_for_the_day": self.volume_trade_for_the_day,
            "total_buy_quantity": self.total_buy_quantity,
            "total_sell_quantity": self.total_sell_quantity,
        }
