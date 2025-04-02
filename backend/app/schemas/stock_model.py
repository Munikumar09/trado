from pydantic import BaseModel


class StockPriceInfo(BaseModel):
    """
    StockPriceInfo model represents the stock price information of a stock
    """

    symbol: str
    last_traded_price: float
    day_open: float
    day_low: float
    day_high: float
    change: float
    percent_change: float


class HistoricalStockPriceInfo(BaseModel):
    """
    HistoricalStockPriceInfo model represents the historical stock price
    information of a stock at a given time
    """

    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    stock_symbol: str
    candle_interval: str


class SmartAPIStockPriceInfo(StockPriceInfo):
    """
    SmartAPIStockPriceInfo model represents the stock price information of a stock
    """

    symbol_token: str
    prev_day_close: float


class HistoricalStockDataBundle(BaseModel):
    """
    HistoricalStockDataBundle model represents the available historical stock price
    information and timestamps of missing data points of a stock at a given time
    """

    available_stock_data: list[HistoricalStockPriceInfo]
    missing_timestamps: list[str]
