import pytest

from app.utils.common.types.reques_types import CandlestickInterval


@pytest.fixture
def valid_intervals_io() -> list[tuple[str, CandlestickInterval]]:
    """
    Test cases for valid intervals that may contains spaces, hyphens, underscores, and trailing 's'.

    Return:
    -------
    ``list[tuple[str, CandlestickInterval]]``
        List of input intervals and their expected CandlestickInterval member
    """
    return [
        ("1minute", CandlestickInterval.ONE_MINUTE),
        ("3min", CandlestickInterval.THREE_MINUTE),
        ("tenminute", CandlestickInterval.TEN_MINUTE),
        ("30min", CandlestickInterval.THIRTY_MINUTE),
        ("1hr", CandlestickInterval.ONE_HOUR),
        ("1d", CandlestickInterval.ONE_DAY),
        ("oneday", CandlestickInterval.ONE_DAY),
        ("1 min", CandlestickInterval.ONE_MINUTE),
        ("1-minute", CandlestickInterval.ONE_MINUTE),
        ("one_minute", CandlestickInterval.ONE_MINUTE),
        ("5MINS", CandlestickInterval.FIVE_MINUTE),
        ("1 hr", CandlestickInterval.ONE_HOUR),
        ("ONE HOUR", CandlestickInterval.ONE_HOUR),
        ("1_D", CandlestickInterval.ONE_DAY),
    ]


@pytest.fixture
def invalid_intervals_io() -> list[str]:
    """
    Test cases for invalid intervals.

    Return:
    -------
    ``list[str]``
        List of invalid input intervals
    """
    return [
        "2min",
        "invalid",
        "30 sec",
        "fivemin",
        "",
        "10",
    ]
