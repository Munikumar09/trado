# pylint: disable=missing-function-docstring
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.routers.nse.equity.equity import router

# from app.schemas.stock_model import StockPriceInfo

client = TestClient(router)


def validate_exception(endpoint_url, expected_error):
    try:
        # Make a GET request to the endpoint URL
        client.get(endpoint_url)
    except HTTPException as http_exc:
        # Check if the status code of the exception matches the expected error status code
        assert http_exc.status_code == expected_error["status_code"]

        # Check if the detail message of the exception matches the expected error detail
        assert http_exc.detail == expected_error["detail"]


# def test_get_stock_data(get_stock_data_io):
#     for stock_data in get_stock_data_io:
#         stock_symbol = stock_data["input"]
#         endpoint_url = f"/nse/equity/stock/{stock_symbol}"

#         # Check if the status code is 200
#         try:
#             if stock_data["status_code"] == 200:
#                 # Send a GET request to the endpoint URL
#                 response = client.get(endpoint_url)

#                 # Assert that the response status code matches the expected status code
#                 assert response.status_code == stock_data["status_code"]

#                 # Assert that the response contains JSON data
#                 assert response.json() is not None

#                 # Parse the response JSON into a StockPriceInfo object
#                 stock_price_info = StockPriceInfo.parse_obj(response.json())

#                 # Assert that the symbol in the StockPriceInfo object matches the stock symbol
#                 assert stock_price_info.symbol == stock_symbol

#                 # Assert that the stock_price_info object is an instance of StockPriceInfo
#                 assert isinstance(stock_price_info, StockPriceInfo)
#         except HTTPException as http_exc:
#             assert http_exc.status_code == 503
#             assert http_exc.detail == {"Error": "no data found"}
#         else:
#             # If the status code is not 200, validate the exception with the expected error
#             validate_exception(endpoint_url, expected_error=stock_data)


# def test_nifty_index_stocks(get_nifty_index_stocks_io):
#     for index_data in get_nifty_index_stocks_io:
#         index_symbol = index_data["symbol"]
#         endpoint_url = f"nse/equity/index_stocks/{index_symbol}"
#         if index_data["status_code"] == 200:
#             response = client.get(endpoint_url)
#             assert response.status_code == index_data["status_code"]
#             assert response.json() is not None

#             stocks_price_info = [
#                 StockPriceInfo.parse_obj(stock_price) for stock_price in response.json()
#             ]
#             assert len(stocks_price_info) == index_data["total_stocks"]
#             assert isinstance(stocks_price_info, list)
#         else:
#             validate_exception(endpoint_url, index_data)


# def test_nse_index_data(nse_index_data_io):
#     for index_data in nse_index_data_io:
#         index_symbol = index_data["symbol"]
#         endpoint_url = f"nse/equity/index/{index_symbol}"
#         if index_data["status_code"] == 200:
#             response = client.get(endpoint_url)
#             assert response.status_code == index_data["status_code"]
#             assert response.json() is not None
#             stock_price_info = StockPriceInfo.parse_obj(response.json())
#             assert stock_price_info.symbol == index_symbol
#             assert isinstance(stock_price_info, StockPriceInfo)

#         else:
#             validate_exception(endpoint_url, index_data)


# def test_stock_listing_date_nse(nse_stock_listing_date_io):
#     for stock_data in nse_stock_listing_date_io:
#         stock_symbol = stock_data["input"]
#         endpoint_url = f"/nse/equity/listing/{stock_symbol}"

#         # Check if the status code is 200
#         try:
#             if stock_data["status_code"] == 200:
#                 # Send a GET request to the endpoint URL
#                 response = client.get(endpoint_url)

#                 # Assert that the response status code matches the expected status code
#                 assert response.status_code == stock_data["status_code"]

#                 # Assert that the response contains JSON data
#                 assert response.json() is not None

#                 assert response.json() == stock_data["listing_date"]

#         except HTTPException as http_exc:
#             assert http_exc.status_code == 503
#             assert http_exc.detail == {"Error": "no data found"}
#         else:
#             # If the status code is not 200, validate the exception with the expected error
#             validate_exception(endpoint_url, expected_error=stock_data)
