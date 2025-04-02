import pytest

from app.utils.common.exceptions.historical_data import IntervalNotFoundException
from app.utils.common.types.reques_types import CandlestickInterval


def test_validate_interval_valid(
    valid_intervals_io: list[tuple[str, CandlestickInterval]]
):
    """
    Test function to check whether the validate_interval function is returning a valid
    CandlestickInterval member corresponds to the given valid input interval string.

    Parameters:
    -----------
    valid_intervals_io: ``list[tuple[str, CandlestickInterval]]``
        List of input intervals and their expected CandlestickInterval member
    """
    for valid_interval in valid_intervals_io:
        assert (
            CandlestickInterval.validate_interval(valid_interval[0])
            == valid_interval[1]
        )


def test_validate_interval_invalid(invalid_intervals_io: list[str]):
    """
    Test function to check whether the validate_interval function is raising an exception or not
    for the given invalid input interval string.

    Parameters:
    -----------
    invalid_intervals_io: ``list[str]``
        List of invalid input intervals
    """
    for invalid_interval in invalid_intervals_io:
        with pytest.raises(IntervalNotFoundException) as excinfo:
            CandlestickInterval.validate_interval(invalid_interval)
        assert excinfo.value.status_code == 404
        assert (
            f"Candlestick interval {invalid_interval} not found. Please provide a valid interval."
            == excinfo.value.detail
        )
