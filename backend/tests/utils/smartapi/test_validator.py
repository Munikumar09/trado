from datetime import datetime, timedelta

import pytest

from app.utils.common.exceptions.historical_data import (
    AllDaysHolidayException,
    DataUnavailableException,
    InvalidDateRangeBoundsException,
    InvalidTradingHoursException,
)
from app.utils.common.types.reques_types import CandlestickInterval
from app.utils.smartapi.validator import (
    check_data_availability,
    find_open_market_days,
    validate_dates,
)


def test_open_market_days_no_weekends_no_holidays():
    """
    Test open_market_days function for a range of dates without weekends or holidays.
    """
    start_datetime = datetime(2023, 1, 2)
    end_datetime = datetime(2023, 1, 6)  # A week without holidays
    expected_days = [start_datetime + timedelta(days=i) for i in range(5)]
    assert find_open_market_days(start_datetime, end_datetime) == expected_days


def test_open_market_days_with_weekends_and_holidays():
    """
    Test open_market_days function for a range of dates including weekends and holidays
    and market working days.
    """
    start_datetime = datetime(2024, 1, 22)
    end_datetime = datetime(2024, 1, 30)  # Includes a weekend and a holiday
    expected_days = [
        datetime(2024, 1, 23),
        datetime(2024, 1, 24),
        datetime(2024, 1, 25),  # Weekdays before the holiday
        datetime(2024, 1, 29),
        datetime(2024, 1, 30),  # Weekdays after the holiday
    ]
    assert find_open_market_days(start_datetime, end_datetime) == expected_days


def test_open_market_days_only_weekends_and_holidays():
    """
    Test open_market_days function for an empty list when the range is only weekends and holidays.
    """
    start_datetime = datetime(2024, 1, 26)  # A holiday
    end_datetime = datetime(2024, 1, 28)  # The following weekend
    assert not find_open_market_days(start_datetime, end_datetime)


def test_open_market_days_edge_cases():
    """
    Test open_market_days function for edge cases like:
    - Start date is after the end date
    - Start date and end date are the same and it's a holiday
    - Start date and end date are the same and it's not a holiday
    """
    # Start date is after the end date
    with pytest.raises(ValueError):
        find_open_market_days(datetime(2024, 1, 30), datetime(2024, 1, 24))

    # Start date and end date are the same and it's a holiday
    assert not find_open_market_days(datetime(2024, 1, 26), datetime(2024, 1, 26))

    # Start date and end date are the same and it's not a holiday
    assert find_open_market_days(datetime(2024, 1, 29), datetime(2024, 1, 29)) == [
        datetime(2024, 1, 29)
    ]


def test_data_available_for_requested_start_date():
    """
    Test check_data_availability function when data is available for the requested start date.
    """
    start_datetime = datetime(2023, 1, 2)
    end_datetime = datetime(2023, 1, 10)
    stock_symbol = "ABB"
    interval = CandlestickInterval.ONE_MINUTE
    assert (
        check_data_availability(start_datetime, end_datetime, stock_symbol, interval)
        == start_datetime
    )


def test_data_not_available_for_requested_start_date():
    """
    Test check_data_availability function when data is not available
    for the requested start date but is available later before the end date.
    """
    start_datetime = datetime(2016, 9, 28)
    end_datetime = datetime(2016, 10, 10)
    stock_symbol = "ACC"
    interval = CandlestickInterval.THREE_MINUTE
    expected_start_date = datetime(2016, 10, 3)
    assert (
        check_data_availability(start_datetime, end_datetime, stock_symbol, interval)
        == expected_start_date
    )


def test_end_date_before_data_availability_start_date():
    """
    Test check_data_availability function when data is not available for the requested dates.
    i.e. given date range is below the data available starting date.
    """
    start_datetime = datetime(2016, 5, 1)
    end_datetime = datetime(2016, 5, 25)
    stock_symbol = "APLAPOLLO"
    interval = CandlestickInterval.TEN_MINUTE
    data_start_date = "2016-10-03"

    with pytest.raises(DataUnavailableException) as excinfo:
        check_data_availability(start_datetime, end_datetime, stock_symbol, interval)

    assert excinfo.value.status_code == 404
    assert (
        "Data for the provided dates is unavailable; please use a date range "
        f"starting from the {data_start_date} date onwards."
    ) in str(excinfo.value.detail)


def test_data_unavailable_for_stock():
    """
    Test check_data_availability function when there is no data available for the given stock and interval.
    i.e data starting date is none for the given stock symbol and interval.
    """
    start_datetime = datetime(2019, 12, 31)
    end_datetime = datetime(2020, 1, 1)
    stock_symbol = "DOMS"
    interval = CandlestickInterval.ONE_DAY

    with pytest.raises(DataUnavailableException) as excinfo:
        check_data_availability(start_datetime, end_datetime, stock_symbol, interval)

    assert excinfo.value.status_code == 404
    assert f"No data available for this stock {stock_symbol}" in str(
        excinfo.value.detail
    )


def test_validate_dates_valid():
    """
    Test validate_dates function for valid dates and their range and trading hours
    """
    from_date = "2024-01-02 09:30"
    to_date = "2024-01-03 15:00"
    interval = CandlestickInterval.THREE_MINUTE
    stock_symbol = "ACC"
    start_datetime, end_datetime = validate_dates(
        from_date, to_date, interval, stock_symbol
    )

    assert start_datetime == datetime(2024, 1, 2, 9, 30)
    assert end_datetime == datetime(2024, 1, 3, 15, 0)


def test_validate_dates_invalid_date_range_bounds():
    """
    Test validate_dates function for invalid date range.
    i.e. Either exceeds the limit per request or start date is greater than end date.
    """
    from_date = "2024-01-02 09:30"
    to_date = "2024-03-10 15:00"  # Exceeds the allowed range for ONE_MINUTE interval
    interval = CandlestickInterval.ONE_MINUTE
    stock_symbol = "ACC"

    with pytest.raises(InvalidDateRangeBoundsException) as excinfo:
        validate_dates(from_date, to_date, interval, stock_symbol)

    assert excinfo.value.status_code == 416
    assert (
        f"The date range from {from_date} to {to_date} is invalid. Please ensure that the end date is greater than or "
        f"equal to start date and difference between them does not exceed {interval.value[1]} for given interval {interval.name}."
    ) == excinfo.value.detail


def test_validate_dates_all_days_holidays():
    """
    Test validate_dates function for all days being holidays.
    i.e. the date range from start_date to end_date are all holidays.
    """
    from_date = "2024-01-26 09:30"
    to_date = "2024-01-28 15:00"
    interval = CandlestickInterval.ONE_MINUTE
    stock_symbol = "ABB"

    # All days are holidays
    with pytest.raises(AllDaysHolidayException) as excinfo:
        validate_dates(from_date, to_date, interval, stock_symbol)

    assert excinfo.value.status_code == 400
    assert f"All days from {from_date} to {to_date} are market holidays."


def test_validate_dates_invalid_trading_hours():
    """
    Test valid_dates function for invalid trading hours.
    i.e. before the market open or after the market close hours.
    """
    from_date = "2024-01-02 07:00"  # Before market open
    to_date = "2024-01-02 9:00"
    interval = CandlestickInterval.ONE_MINUTE
    stock_symbol = "TCS"

    with pytest.raises(InvalidTradingHoursException) as excinfo:
        validate_dates(from_date, to_date, interval, stock_symbol)

    assert excinfo.value.status_code == 400
    assert (
        "Attempted to access trading system outside of trading hours."
        == excinfo.value.detail
    )
