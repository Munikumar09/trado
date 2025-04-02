# pylint: disable=chained-comparison
from bisect import bisect_left
from datetime import datetime, time, timedelta

from app.utils.common.exceptions.historical_data import (
    AllDaysHolidayException,
    DataUnavailableException,
    InvalidDateRangeBoundsException,
    InvalidTradingHoursException,
    SymbolNotFoundException,
)
from app.utils.common.types.financial_types import ExchangeType
from app.utils.common.types.reques_types import CandlestickInterval
from app.utils.date_utils import validate_datetime_format
from app.utils.file_utils import get_symbols, load_json_data, read_text_data
from app.utils.smartapi.constants import (
    BSE_SYMBOLS_PATH,
    DATA_STARTING_DATES_PATH,
    NSE_HOLIDAYS_PATH,
    NSE_SYMBOLS_PATH,
)


def validate_symbol_and_get_token(
    stock_exchange: ExchangeType, stock_symbol: str
) -> tuple[str, str]:
    """
    Validate the stock symbol and get the stock token from the symbols data.
    Ref NSE website for information about stock symbols.

    Parameters:
    -----------
    stock_exchange: ``ExchangeType``
        The stock exchange of the stock symbol
    stock_symbol: ``str``
        The stock symbol to be validated and get the token

    Raises:
    -------
    ``SymbolNotFoundException``
        If the stock symbol is not found in the symbols data

    Returns:
    --------
    ``tuple[str, str]``
        The stock token and the stock symbol

    """
    stock_symbol = stock_symbol.upper()
    symbols_path = BSE_SYMBOLS_PATH

    if stock_exchange == ExchangeType.NSE:
        symbols_path = NSE_SYMBOLS_PATH

    all_symbols_data = get_symbols(symbols_path)

    if stock_symbol not in all_symbols_data:
        raise SymbolNotFoundException(stock_symbol)

    return all_symbols_data[stock_symbol], stock_symbol


def find_open_market_days(
    start_datetime: datetime, end_datetime: datetime
) -> list[datetime]:
    """
    Finds the open market days between start_datetime and end_datetime.

    Parameters:
    -----------
    start_datetime: ``datetime``
        The initial date from which to find the open market days
    end_datetime: ``datetime``
        The final date upto which to find the open market days

    Return:
    -------
    ``list[datetime]``
        List of open market days between the given start date and end date
    """
    if end_datetime < start_datetime:
        raise ValueError()

    # Read the holidays data into list
    holidays_data = read_text_data(NSE_HOLIDAYS_PATH)
    open_days = []

    # Check for any market open day between given dates.
    # If you find any open day then definitely the data is not empty otherwise raise an error.
    for day in range((end_datetime.date() - start_datetime.date()).days + 1):
        current_datetime = start_datetime + timedelta(days=day)
        index = bisect_left(holidays_data, current_datetime.strftime("%Y-%m-%d"))
        if (
            current_datetime.weekday() < 5
            and index != len(holidays_data)
            and holidays_data[index] != current_datetime.strftime("%Y-%m-%d")
        ):
            open_days.append(current_datetime)

    return open_days


def check_data_availability(
    start_datetime: datetime,
    end_datetime: datetime,
    stock_symbol: str,
    interval: CandlestickInterval,
) -> datetime:
    """
    Verifies the availability of stock data for a given stock symbol and date range
    through SmartAPI. Returns the earliest date from which data is available. If data
    is available for the requested start date, it returns that date otherwise, it
    returns the earliest available date with data.


    Parameters:
    -----------
    start_datetime: ``datetime``
        The initial date from which historical stock data should be retrieved
    end_datetime: ``datetime``
        The final date up to which historical stock data should be retrieved
    stock_symbol: ``str``
        The symbol of the stock
    interval: ``CandlestickInterval``
        The interval of the Candlestick

    Exceptions:
    -----------
    ``DataUnavailableException``
        Raised when the data is unavailable for the requested dates from the SmartAPI

    Return:
    -------
    ``datetime``
        Either the requested start date or the earliest available date with data
    """

    # If end date is less than the date from where the data availability starts, then
    # no data can be retrieved; therefore, an error should be raised.
    data_starting_dates = load_json_data(DATA_STARTING_DATES_PATH)

    if stock_symbol in data_starting_dates:
        data_starting_date = data_starting_dates.get(stock_symbol).get(interval.name)
        if not data_starting_date:
            raise DataUnavailableException(data_starting_date, stock_symbol)
        data_starting_date = datetime.strptime(data_starting_date, "%Y-%m-%d")
        if data_starting_date > end_datetime:
            raise DataUnavailableException(data_starting_date, stock_symbol)
    else:
        data_starting_date = start_datetime

    return max(start_datetime, data_starting_date)


def validate_dates(
    from_date: str, to_date: str, interval: CandlestickInterval, stock_symbol: str
) -> tuple[datetime, datetime]:
    """
    Validate given dates and their range.

    Parameters:
    -----------
    from_date: ``str``
        Start date and time to be validated
    to_date: ``str``
        End date and time to be validated
    interval: ``CandlestickInterval``
        The interval of the candlestick
    stock_symbol: ``str``
        The symbol of the stock

    Exceptions:
    -----------
    ``InvalidDateRangeBoundsException``
        If the specified date range is invalid for given interval

    ``InvalidTradingHoursException``
        If the time accessed outside trading hours of stock market

    ``AllDaysHolidayException``
        Raised when all days in the given date range are market holidays

    Return:
    -------
    ``Tuple[datetime, datetime]``
        validated start and end datetimes
    """
    start_datetime = validate_datetime_format(from_date)
    end_datetime = validate_datetime_format(to_date)

    # check data is available or not between given dates.
    start_datetime = check_data_availability(
        start_datetime, end_datetime, stock_symbol, interval
    )

    # check date range should not exceed specific days per request based on given interval.
    total_days = (end_datetime - start_datetime).days
    if total_days < 0 or total_days > interval.value[1]:
        raise InvalidDateRangeBoundsException(
            from_date, to_date, interval.value[1], interval.name
        )

    # check given dates range are market holidays or not.
    open_dates = find_open_market_days(start_datetime, end_datetime)
    if not open_dates:
        raise AllDaysHolidayException(
            start_datetime.strftime("%Y-%m-%d"), end_datetime.strftime("%Y-%m-%d")
        )

    # check given timings are market active trading hours.
    start_time = time(9, 15)
    end_time = time(15, 29)

    if (
        interval.name != "ONE_DAY"
        and open_dates[0].date() == end_datetime.date()
        and (end_datetime.time() < start_time or start_datetime.time() > end_time)
    ):
        raise InvalidTradingHoursException()

    return start_datetime, end_datetime
