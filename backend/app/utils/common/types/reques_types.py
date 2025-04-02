# pylint: disable=missing-class-docstring
from enum import Enum

from app.utils.common.exceptions.historical_data import IntervalNotFoundException


class RequestType(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


class CandlestickInterval(Enum):
    """
    Enumeration class representing possible candlestick intervals.

    Each member except the last one corresponds to a specific time interval
    and its associated value which is a tuple of numerical value of interval
    in minutes and max days possible per request. The last member corresponds
    to the possible input intervals given by the users
    """

    ONE_MINUTE = (1, 30)
    THREE_MINUTE = (3, 60)
    FIVE_MINUTE = (5, 100)
    TEN_MINUTE = (10, 100)
    FIFTEEN_MINUTE = (15, 200)
    THIRTY_MINUTE = (30, 200)
    ONE_HOUR = (60, 400)
    ONE_DAY = (1440, 2000)
    POSSIBLE_INPUT_INTERVALS = (
        ("1min", "1minute", "oneminute", ONE_MINUTE),
        ("3min", "3minute", "threeminute", THREE_MINUTE),
        ("5min", "5minute", "fiveminute", FIVE_MINUTE),
        ("10min", "10minute", "tenminute", TEN_MINUTE),
        ("15min", "15minute", "fifteenminute", FIFTEEN_MINUTE),
        ("30min", "30minute", "thirtyminute", THIRTY_MINUTE),
        ("1hr", "1hour", "onehour", ONE_HOUR),
        ("1d", "1day", "oneday", ONE_DAY),
    )

    @staticmethod
    def validate_interval(interval: str) -> "CandlestickInterval":
        """
        Validates an interval string and returns the corresponding enum member.

        Parameters:
        -----------
        interval: ``str``
            The interval string to validate (e.g., "one minute", "15m")

        Exceptions:
        -----------
        ``IntervalNotFoundException``
            If the interval is not a valid enum member

        Return:
        -------
        ``CandlestickInterval``
            The enum member corresponding to the validated interval
        """
        # Normalize the interval string by removing spaces, -, _, and converting all characters to lower case.
        normalized_interval = (
            interval.lower().replace(" ", "").replace("-", "").replace("_", "")
        )

        # Remove last s in the interval if exist.
        if normalized_interval.endswith("s"):
            normalized_interval = normalized_interval[:-1]

        for (
            possible_input_interval
        ) in CandlestickInterval.POSSIBLE_INPUT_INTERVALS.value:
            if normalized_interval in possible_input_interval:
                return CandlestickInterval(possible_input_interval[3])

        raise IntervalNotFoundException(interval)
