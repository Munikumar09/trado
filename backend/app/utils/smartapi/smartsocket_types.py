from enum import Enum
from typing import Type, TypeVar

from app.utils.common.types.financial_types import ExchangeType

# Declare a TypeVar for enum types
E = TypeVar("E", bound=Enum)


def get_enum_member(enum_class: Type[E], value: str | int) -> E:
    """
    This function is used to get an enum member from an enum class based on the enum name or value.

    Example:
    --------
    >>> from enum import Enum
    >>> class Color(Enum):
    ...     RED = 1
    ...     GREEN = 2
    ...     BLUE = 3
    ...
    >>> get_enum_member(Color, "RED")
    <Color.RED: 1>

    Parameters
    ----------
    enum_class: ``Type[E]``
        The enum class from which to get the enum member
    value: ``str`` | ``int``
        The enum name or value to get the enum member from the enum class

    Returns
    -------
    ``E``
        The enum member from the enum class based on the enum name or value
    """
    if isinstance(value, str):
        value = value.upper()
        try:
            return enum_class[value]
        except KeyError:
            valid_names = [member.name for member in enum_class]

            raise ValueError(
                f"{enum_class.__name__} name '{value}' is not supported. "
                f"Supported {enum_class.__name__} names are: {valid_names}"
            ) from None
    elif isinstance(value, int):
        try:
            return enum_class(value)
        except ValueError:
            valid_values = [member.value for member in enum_class]

            raise ValueError(
                f"{enum_class.__name__} value '{value}' is not supported. "
                f"Supported {enum_class.__name__} values are: {valid_values}"
            ) from None
    else:
        raise TypeError(
            f"{enum_class.__name__} symbol must be of type str or int, not {type(value).__name__}."
        )


class SubscriptionMode(Enum):
    """
    This enumeration class is used to define the subscription mode for the WebSocket client.
    """

    LTP = 1
    QUOTE = 2
    SNAP_QUOTE = 3
    DEPTH = 4

    @staticmethod
    def get_subscription_mode(subscription_mode: str | int) -> "SubscriptionMode":
        """
        This method is used to get the subscription mode based on the subscription mode symbol or value.

        Parameters
        ----------
        subscription_mode: ``str`` | ``int``
            The subscription mode symbol or value

        Returns
        -------
        ``SubscriptionMode``
            The subscription mode based on the subscription mode symbol or value
        """
        return get_enum_member(SubscriptionMode, subscription_mode)


class SmartAPIExchangeSegment(Enum):
    """
    This enumeration class is used to define the exchange type for the WebSocket client.
    """

    NSE_CM = 1
    BSE_CM = 2

    @staticmethod
    def get_exchange(exchange_symbol: str | int) -> "SmartAPIExchangeSegment":
        """
        This method is used to get the exchange type based on the exchange symbol or value.

        Parameters
        ----------
        exchange_symbol: ``str`` | ``int``
            The exchange symbol or value

        Returns
        -------
        ``ExchangeType``
            The exchange type based on the exchange symbol or value
        """
        return get_enum_member(SmartAPIExchangeSegment, exchange_symbol)


class SubscriptionAction(Enum):
    """
    This enumeration class is used to define the subscription action for the WebSocket client.
    """

    SUBSCRIBE = 1
    UNSUBSCRIBE = 2


SMARTAPI_EXCHANGETYPE_MAP = {
    SmartAPIExchangeSegment.NSE_CM: ExchangeType.NSE,
    SmartAPIExchangeSegment.BSE_CM: ExchangeType.BSE,
}
EXCHANGETYPE_SMARTAPI_MAP = {v: k for k, v in SMARTAPI_EXCHANGETYPE_MAP.items()}
