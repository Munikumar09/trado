import pytest

from app.utils.common.types.financial_types import DataProviderType, ExchangeType


@pytest.fixture
def kafka_data() -> list[dict]:
    """
    This fixture returns the sample data that is retrieved from kafka consumer.

    Returns:
    --------
    ``list[dict]``
        Sample data retrieved from kafka consumer
    """
    return [
        {
            "subscription_mode": 3,
            "exchange_type": 1,
            "token": "10893",
            "sequence_number": 18537152,
            "exchange_timestamp": 1729506514000,
            "last_traded_price": 13468,
            "subscription_mode_val": "SNAP_QUOTE",
            "last_traded_quantity": 414,
            "average_traded_price": 13529,
            "volume_trade_for_the_day": 131137,
            "total_buy_quantity": 0.2,
            "total_sell_quantity": 0.1,
            "open_price_of_the_day": 13820,
            "high_price_of_the_day": 13820,
            "low_price_of_the_day": 13365,
            "closed_price": 13726,
            "last_traded_timestamp": 1729504796,
            "exchange_id": ExchangeType.NSE.value,
            "data_provider_id": DataProviderType.SMARTAPI.value,
            "open_interest": 0,
            "open_interest_change_percentage": 0,
            "symbol": "DBOL",
            "socket_name": "smartapi",
            "retrieval_timestamp": 1729532024.309936,
        },
        # {
        #     "subscription_mode": 3,
        #     "exchange_type": 1,
        #     "token": "13658",
        #     "sequence_number": 18578253,
        #     "exchange_timestamp": 1729506939000,
        #     "last_traded_price": 40260,
        #     "subscription_mode_val": "SNAP_QUOTE",
        #     "last_traded_quantity": 50,
        #     "average_traded_price": 40510,
        #     "volume_trade_for_the_day": 13846,
        #     "total_buy_quantity": 0.5,
        #     "total_sell_quantity": 0.45,
        #     "open_price_of_the_day": 41220,
        #     "high_price_of_the_day": 41270,
        #     "low_price_of_the_day": 39685,
        #     "exchange_id":ExchangeType.NSE.value,
        #     "data_provider_id":DataProviderType.SMARTAPI.value,
        #     "closed_price": 41000,
        #     "last_traded_timestamp": 1729505946,
        #     "open_interest": 0,
        #     "open_interest_change_percentage": 0,
        #     "symbol": "GEECEE",
        #     "socket_name": "smartapi",
        #     "retrieval_timestamp": 1729532024.31136,
        # },
    ]
