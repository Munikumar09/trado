from datetime import datetime

import pytest

from app.utils.common.exceptions.historical_data import InvalidDateTimeFormatException
from app.utils.date_utils import validate_datetime_format


def test_validate_datetime_format_valid(
    valid_datetime_formats_io: list[tuple[str, datetime]]
):
    """
    Test validate_datetime_format function for valid datetime formats.
    i.e. year-month-day hour:minute or year/month/day hour:minute

    Parameters:
    -----------
    valid_datetime_formats_io: ``list[tuple[str, datetime]]``
        List of valid datetime format of input string and their expected datetime object
    """
    for valid_datetime in valid_datetime_formats_io:
        assert validate_datetime_format(valid_datetime[0]) == valid_datetime[1]


def test_validate_datetime_format_invalid(invalid_datetime_formats_io: list[str]):
    """
    Test validate_datetime_format function for invalid datetime formats.
    i.e. not year-month-day hour:minute or year/month/day hour:minute

    Parameters:
    -----------
    invalid_datetime_formats_io: ``list[str]``
        List of invalid datetime format of input strings
    """
    for invalid_datetime in invalid_datetime_formats_io:
        with pytest.raises(InvalidDateTimeFormatException) as excinfo:
            validate_datetime_format(invalid_datetime)

        assert excinfo.value.status_code == 400
        assert (
            f"Given datetime format {invalid_datetime} is invalid. "
            "Please provide a valid datetime that should be in the form 'year-month-day hour:minute'."
        ) in str(excinfo.value.detail)


def test_validate_datetime_format_edge_cases(invalid_datetime_value_io: list[str]):
    """
    Test for edge cases like invalid day or month or year or time.
    i.e. valid datetime format but invalid datetime value.

    e.g. `2023-13-12 3:22` here month 13 is invalid since there are only 12 months.

    Parameters:
    -----------
    date_time: ``list[str]``
        List of invalid datetime format of input string
    """
    for datetime_value in invalid_datetime_value_io:
        with pytest.raises(InvalidDateTimeFormatException) as excinfo:
            validate_datetime_format(datetime_value)

        assert excinfo.value.status_code == 400
        assert (
            f"Given datetime format {datetime_value} is invalid. "
            "Please provide a valid datetime that should be in the form 'year-month-day hour:minute'."
        ) in str(excinfo.value.detail)
