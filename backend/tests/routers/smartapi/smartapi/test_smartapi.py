# pylint: disable=missing-function-docstring
from fastapi.testclient import TestClient

from app.routers.smartapi.smartapi.smartapi import router
from app.schemas.stock_model import SmartAPIStockPriceInfo
from tests.utils.common.exception_validators import validate_exception

client = TestClient(router)


def test_latest_price_quotes(stock_symbol_io):
    for stock_symbol_data in stock_symbol_io:
        endpoint_url = f"/smart-api/equity/price/{stock_symbol_data['input']}"

        if stock_symbol_data["status_code"] == 200:
            # Send a GET request to the endpoint URL
            response = client.get(endpoint_url)

            # Assert that the response status code matches the expected status code
            assert response.status_code == stock_symbol_data["status_code"]

            # Assert that the response contains JSON data
            assert response.json() is not None

            # Parse the response JSON into a SmartAPIStockPriceInfo object
            smart_api_stock_price_info = SmartAPIStockPriceInfo.parse_obj(
                response.json()
            )

            # Assert that the stock_price_info object is an instance of StockPriceInfo
            assert isinstance(smart_api_stock_price_info, SmartAPIStockPriceInfo)

            # Check if the stock token and symbol in the SmartAPIStockPriceInfo object matches the stock symbol data
            assert (
                smart_api_stock_price_info.symbol_token
                == stock_symbol_data["symbol_token"]
            )
            assert smart_api_stock_price_info.symbol == stock_symbol_data["symbol"]
        else:
            validate_exception(endpoint_url, stock_symbol_data, client)
