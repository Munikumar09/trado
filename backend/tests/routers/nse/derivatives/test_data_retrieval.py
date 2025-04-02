# # pylint: disable=missing-function-docstring
# from typing import Any

# import pytest
# from fastapi import HTTPException

# from app.routers.nse.derivatives.data_retrieval import get_option_chain
# from app.schemas.option_model import ExpiryOptionData, Option, StrikePriceData


# def test_get_option_chain(get_option_chain_io: list[dict[str, Any]]):
#     for option_chain_io in get_option_chain_io:
#         option_chain_input = option_chain_io["input"]

#         if option_chain_io["output"] is not None:
#             with pytest.raises(HTTPException) as http_exception:
#                 get_option_chain(*option_chain_input)
#             assert (
#                 http_exception.value.status_code
#                 == option_chain_io["output"]["status_code"]
#             )
#             assert http_exception.value.detail == option_chain_io["output"]["detail"]
#         else:
#             actual_option_data = get_option_chain(*option_chain_input)

#             assert isinstance(actual_option_data, ExpiryOptionData)
#             assert actual_option_data.expiry_date == option_chain_input[0]
#             assert isinstance(actual_option_data.strike_prices, list)
#             assert len(actual_option_data.strike_prices) > 0

#             # For far strike prices from the current spot price, the option contracts are not in
#             # trade and we will get empty data. So the middle strike price is taken to check the
#             # option contracts.
#             strike_price_data = actual_option_data.strike_prices[
#                 len(actual_option_data.strike_prices) // 2
#             ]

#             assert isinstance(strike_price_data, StrikePriceData)

#             assert isinstance(strike_price_data.ce, Option)
#             assert isinstance(strike_price_data.pe, Option)
