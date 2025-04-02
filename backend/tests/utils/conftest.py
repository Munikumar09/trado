from datetime import datetime

import pytest


@pytest.fixture
def valid_datetime_formats_io() -> list[tuple[str, datetime]]:
    """
    Test cases for valid date and time formats

    Return:
    -------
    ``list[tuple[str, datetime]]``
        List of valid datetime format of input string and their expected datetime object
    """
    return [
        ("2024-04-30 21:10", datetime(2024, 4, 30, 21, 10)),
        ("2023/Apr/30 9:15", datetime(2023, 4, 30, 9, 15)),
        ("2021/April/30 21:10", datetime(2021, 4, 30, 21, 10)),
        ("2022-OCTOber-5 11:0", datetime(2022, 10, 5, 11, 0)),
    ]


@pytest.fixture
def invalid_datetime_formats_io() -> list[str]:
    """
    Test cases for invalid date and time formats

    Return:
    -------
    ``list[str]``
        List of invalid datetime format of input strings
    """
    return [
        "30-04-2024 21:10",
        "04/30/2024 21:10",
        "2024/30/Apr 21:10",
        "2023_3/2 12:32",
        "2022-3-4T3:21",
        "2024/30/Apr 21-10",
        "2024/30/Apr-21:10",
    ]


@pytest.fixture
def invalid_datetime_value_io() -> list[str]:
    """
    Test cases for edge cases like invalid day, month, year, hour, minute.

    Return:
    -------
    ``list[str]``
        List of invalid datetime value of input string
    """
    return [
        "2024-04-31 21:10",  # Invalid day
        "2022-13-30 21:10",  # Invalid month
        "2023-02-29 21:10",  # Invalid leap year date
        "2021-02-29 40:10",  # invalid hours
        "2020-02-29 21:71",  # invalid minutes
    ]
