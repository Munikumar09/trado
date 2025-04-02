# import pytest
# from fastapi import HTTPException

# from app.routers.nse.equity.data_retrieval import (
#     get_index_data,
#     # get_nifty_index_stocks,
#     get_stock_trade_info,
# )
# from app.schemas.stock_model import StockPriceInfo


# def test_get_nifty_index_stocks():
#     """
#     Test function for get_nifty_index_stocks.

#     This function tests the following cases:
#     1. Test for valid input
#     2. Test for invalid input
#     3. Test for empty input
#     4. Test for invalid URL
#     """
#     # Test case 1: Test for valid input
#     url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
#     stocks = get_nifty_index_stocks(url, max_tries=10)
#     assert isinstance(stocks, list)
#     assert all(isinstance(stock, StockPriceInfo) for stock in stocks)

#     # Test case 2: Test for invalid input
#     url = "https://www.nseindia.com/api/equity-stockIndices?index=INVALID"

#     with pytest.raises(HTTPException) as http_exc:
#         get_nifty_index_stocks(url)
#         assert http_exc.status_code == 404
#         assert http_exc.detail == {"Error": "Resource not found or invalid Url"}

#     # Test case 3: Test for empty input
#     url = ""
#     with pytest.raises(ValueError) as value_exc:
#         get_nifty_index_stocks(url)
#         assert value_exc.value == "Url can't be empty"

#     # Test case 4: Test for invalid URL
#     url = "https://www.nseindia.com/ap/equity-stockIndices?index=NIFTY%2050"
#     with pytest.raises(HTTPException) as http_exc:
#         get_nifty_index_stocks(url)
#         assert http_exc.status_code == 404
#         assert http_exc.detail == {"Error": "Resource not found or invalid Url"}


# def test_get_stock_trade_info():
#     """
#     Test function for the get_stock_trade_info function.
#     """
#     # Test case 1: Test for valid input
#     try:
#         symbol = "SBIN"
#         stock_data = get_stock_trade_info(symbol)
#         assert isinstance(stock_data, StockPriceInfo)
#         assert stock_data.symbol == symbol
#     except HTTPException as http_exception:
#         assert http_exception.status_code == 503
#         assert http_exception.detail == {
#             "Error": "no data found for the given symbol at the moment"
#         }

#     # Test case 2: Test for empty input
#     symbol = ""
#     with pytest.raises(ValueError) as value_exc:
#         get_stock_trade_info(symbol)
#         assert str(value_exc.value) == "Symbol can't be empty"

#     # Test case 3: Test for None input
#     symbol = None
#     with pytest.raises(ValueError) as value_exc:
#         get_stock_trade_info(symbol)
#         assert str(value_exc.value) == "Symbol can't be empty"


# def test_get_index_data():
#     """
#     Test function for the get_index_data function.
#     """
#     # Test case 1: Test for valid input
#     symbol = "NIFTY 50"
#     stock_data = get_index_data(symbol)
#     assert isinstance(stock_data, StockPriceInfo)
#     assert stock_data.symbol == symbol

#     # Test case 2: Test for invalid input
#     symbol = "INVALID"
#     with pytest.raises(UnboundLocalError) as unbounded_exc:
#         get_index_data(symbol)
#         assert (
#             unbounded_exc.value
#             == "local variable 'stock_data' referenced before assignment"
#         )

#     # Test case 3: Test for empty input
#     symbol = ""
#     with pytest.raises(ValueError) as value_exc:
#         get_index_data(symbol)
#         assert str(value_exc.value) == "Symbol can't be empty"

#     # Test case 4: Test for None input
#     symbol = None
#     with pytest.raises(ValueError) as value_exc:
#         get_index_data(symbol)
#         assert str(value_exc.value) == "Symbol can't be empty"
