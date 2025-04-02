import calendar
from datetime import date, datetime, timedelta

from app.utils.common.exceptions.historical_data import InvalidDateTimeFormatException
from app.utils.fetch_data import fetch_data
from app.utils.type_utils import SymbolType
from app.utils.urls import INDEX_OPTION_CHAIN_URL, STOCK_OPTION_CHAIN_URL


def last_date_of_weekday(year: int, month: int, weekday: int):
    """
    Returns the last date of a given weekday in a given month and year.

    Parameters
    ----------
    year: ``int``
        The year as a four-digit number, such as 2023
    month: ``int``
        The month as a number from 1 to 12, where 1 is January and 12 is December
    weekday: ``int``
        The weekday as a number from 0 to 6, where 0 is Monday and 6 is Sunday

    Returns
    -------
    ``datetime.date``
        The date object representing the last occurrence of the input weekday
        in the input month and year

    Examples
    --------
    >>> last_date_of_weekday(2023, 10, 6)
    datetime.date(2023, 10, 29)

    >>> last_date_of_weekday(2023, 2, 0)
    datetime.date(2023, 2, 27)

    Raises
    ------
    ``ValueError``
        If the input year, month or weekday is out of range

    """
    try:
        datetime(year, month, weekday)
    except ValueError as exc:
        raise ValueError(f"{weekday}/{month}/{year} is not a valid date") from exc

    last_day = calendar.monthrange(year, month)[1]

    # create a date object for the last day of the current month
    last_date = date(year, month, last_day)

    # get the weekday number of the last day of the current month
    last_weekday = last_date.weekday()

    # calculate the number of days to subtract to get the last date of the input weekday
    days_to_subtract = (last_weekday - weekday + 7) % 7

    # subtract the number of days from the last date and get the result as a date object
    result_date = last_date - timedelta(days=days_to_subtract)

    return result_date


def get_date(weekday: str, monthly: bool = False):
    """
    Returns the date of a given weekday in the next week of last week in the
    month based on monthly flag. If the weekday and today is same then returns
    today date.

    Parameters
    ----------
    weekday : ``str``
        The name or abbreviation of the weekday, such as "monday" or "mon".
        The input is case-insensitive and must be a valid weekday

    monthly : ``bool``, (default = False)
        A flag indicating whether to return the date of the given weekday in the next week
        or last week in the month.

        If True, the function will return the last date of the weekday in the current month
        if it is not before today, otherwise it will return the last date of the weekday in
        the next month.

        If False, the function will return the date of the next occurrence of the weekday
        from today.

    Returns
    -------
    ``str``
        The date of the weekday in the format "%d-%b-%Y", such as "08-Oct-2023"

    Raises
    ------
    ``ValueError``
        If the input weekday is invalid

    Examples
    --------
    >>> get_date("sunday")
    '08-Oct-2023'

    `if today= "wed" and 25-Oct-2023`
    >>> get_date("wed", monthly=True)
    '25-Oct-2023'

    >>> get_date("fri", monthly=True)
    '03-Nov-2023'

    >>> get_date("abc")
    ValueError: abc is not a valid day
    """

    weekday = weekday.lower()
    weekdays = {
        "monday": 0,
        "mon": 0,
        "tuesday": 1,
        "tue": 1,
        "wednesday": 2,
        "wed": 2,
        "thursday": 3,
        "thu": 3,
        "friday": 4,
        "fri": 4,
        "saturday": 5,
        "sat": 5,
        "sunday": 6,
        "sun": 6,
    }
    if weekday not in weekdays:
        raise ValueError(f"{weekday} is not a valid day")

    today = date.today()
    today_num = today.weekday()
    weekday_num = weekdays[weekday]
    days_to_add = (weekday_num - today_num + 7) % 7

    year = today.year
    month = today.month
    weekday_date = today + timedelta(days=days_to_add)

    if monthly:
        weekday_date = last_date_of_weekday(year, month, weekday_num)
        if weekday_date < today:
            if month == 12:
                year += 1
                month = 1
            weekday_date = last_date_of_weekday(year, month, weekday_num)

    return weekday_date.strftime("%d-%b-%Y")


def get_expiry_dates(
    symbol: str, symbol_type: SymbolType = SymbolType.EQUITY
) -> list[str]:
    """
    Returns the expiry dates list for the given stock or index symbol.

    Parameters:
    -----------
    symbol: ``str``
        The symbol that represents stock or index to which expiry dates are required
    symbol_type: ``SymbolType``
        The type of symbol it is, means either `equity` or `derivative`

    Returns:
    --------
    ``list[str]``
        The list of expiry dates for the given symbol
    """

    base_url = STOCK_OPTION_CHAIN_URL

    if symbol_type == SymbolType.DERIVATIVE:
        base_url = INDEX_OPTION_CHAIN_URL

    option_chain_url = f"{base_url}{symbol.upper()}"
    option_data = fetch_data(option_chain_url)

    return option_data["records"]["expiryDates"]


def validate_datetime_format(date_time: str) -> datetime:
    """
    Validates given string is in a valid datetime format or not. Datetime must be in the
    form of 'year-month-day hour:minute'.

    Parameters:
    -----------
    date_time: ``str``
        date and time to be validated

    Exceptions:
    -----------
    ``InvalidDateTimeFormatException``
        If the date and time is in wrong format

    Return:
    -------
    ``datetime``
        validated given date and time and returns valid datetime object
    """
    date_separator = "/" if len(date_time.split("/")) > 1 else "-"
    month_format_codes = ["m", "b", "B"]

    for month_format_code in month_format_codes:
        try:
            datetime_format = (
                f"%Y{date_separator}%{month_format_code}{date_separator}%d %H:%M"
            )
            datetime_obj = datetime.strptime(date_time, datetime_format)

            return datetime_obj

        except ValueError:
            continue

    raise InvalidDateTimeFormatException(date_time)
