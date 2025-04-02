from typing import Optional

from pydantic import BaseModel


class Option(BaseModel):
    """
    Option model to represent PE or CE option data
    """

    ltp: float
    change: float
    percent_change: float
    change_in_oi: float
    percent_change_in_oi: float


class StrikePriceData(BaseModel):
    """
    StrikePriceData model represents the PE and CE information of a derivative
    at a particular strike price
    """

    strike_price: float
    ce: Optional[Option]
    pe: Optional[Option]


class ExpiryOptionData(BaseModel):
    """
    ExpiryOptionData represents the option chain information of a derivative
    at particular expiry date.
    """

    strike_prices: list[StrikePriceData]
    expiry_date: str
