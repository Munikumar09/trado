# pylint: disable=line-too-long
from typing import Any

import pytest


@pytest.fixture
def websocket_instrument_data() -> list[dict[str, Any]]:
    """
    Sample data for the instruments table
    """

    # fmt: off
    return [
        {"symbol": "KAKTEX","exchange_id": 2,"data_provider_id": 2,"token": "BSE_EQ|INE092E01011","name": "KAKATIYA TEXTILES LTD.","instrument_type": "EQ","expiry_date": None,"strike_price": None,"lot_size": 1,"tick_size": 1.0,},
        {"symbol": "RAJKSYN","exchange_id": 2,"data_provider_id": 2,"token": "BSE_EQ|INE376L01013","name": "RAJKAMAL SYNTHETICS LTD.","instrument_type": "EQ","expiry_date": None,"strike_price": None,"lot_size": 1,"tick_size": 1.0,},
        {"symbol": "NILKAMAL","exchange_id": 2,"data_provider_id": 2,"token": "BSE_EQ|INE310A01015","name": "NILKAMAL LTD.","instrument_type": "EQ","expiry_date": None,"strike_price": None,"lot_size": 1,"tick_size": 5.0,},
        {"symbol": "ZEEMEDIA","exchange_id": 2,"data_provider_id": 2,"token": "BSE_EQ|INE966H01019","name": "ZEE MEDIA CORPORATION LIMITED","instrument_type": "EQ","expiry_date": None,"strike_price": None,"lot_size": 1,"tick_size": 1.0,},
        {"symbol": "COSMOFE","exchange_id": 2,"data_provider_id": 2,"token": "BSE_EQ|INE124B01018","name": "COSMO FERRITES LTD.","instrument_type": "EQ","expiry_date": None,"strike_price": None,"lot_size": 1,"tick_size": 5.0,},
        {"symbol": "KJMCFIN","exchange_id": 2,"data_provider_id": 2,"token": "BSE_EQ|INE533C01018","name": "KJMC FINANCIAL SERVICES LTD.","instrument_type": "EQ","expiry_date": None,"strike_price": None,"lot_size": 1,"tick_size": 5.0,},
        {"symbol": "LERTHAI","exchange_id": 2,"data_provider_id": 2,"token": "BSE_EQ|INE347D01011","name": "LERTHAI FINANCE LIMITED","instrument_type": "EQ","expiry_date": None,"strike_price": None,"lot_size": 1,"tick_size": 5.0,},
        {"symbol": "HARIAAPL","exchange_id": 2,"data_provider_id": 2,"token": "BSE_EQ|INE493N01012","name": "HARIA APPARELS LTD","instrument_type": "EQ","expiry_date": None,"strike_price": None,"lot_size": 1,"tick_size": 1.0,},
        {"symbol": "SIGNPOST","exchange_id": 2,"data_provider_id": 2,"token": "BSE_EQ|INE0KGZ01021","name": "Signpost India Limited","instrument_type": "EQ","expiry_date": None,"strike_price": None,"lot_size": 1,"tick_size": 5.0,},
        {"symbol": "REALECO","exchange_id": 2,"data_provider_id": 2,"token": "BSE_EQ|INE055E01034","name": "REAL ECO-ENERGY LIMITED","instrument_type": "EQ","expiry_date": None,"strike_price": None,"lot_size": 1,"tick_size": 1.0,},
        {"symbol": "GUJRAFFIA","exchange_id": 1,"data_provider_id": 1,"token": "10097","name": "GUJRAFFIA","instrument_type": "BE","expiry_date": None,"strike_price": -1.0,"lot_size": 1,"tick_size": 1.0,},
        {"symbol": "KCK","exchange_id": 1,"data_provider_id": 1,"token": "10296","name": "KCK","instrument_type": "ST","expiry_date": None,"strike_price": -1.0,"lot_size": 2500,"tick_size": 5.0,},
        {"symbol": "UNIVAFOODS","exchange_id": 1,"data_provider_id": 1,"token": "10366","name": "UNIVAFOODS","instrument_type": "BE","expiry_date": None,"strike_price": -1.0,"lot_size": 1,"tick_size": 1.0,},
        {"symbol": "MUTHOOTCAP","exchange_id": 1,"data_provider_id": 1,"token": "10415","name": "MUTHOOTCAP","instrument_type": "EQ","expiry_date": None,"strike_price": -1.0,"lot_size": 1,"tick_size": 5.0,},
        {"symbol": "AXISBNKETF","exchange_id": 1,"data_provider_id": 1,"token": "1044","name": "AXISBNKETF","instrument_type": "EQ","expiry_date": None,"strike_price": -1.0,"lot_size": 1,"tick_size": 1.0,},
        {"symbol": "LUPIN","exchange_id": 1,"data_provider_id": 1,"token": "10440","name": "LUPIN","instrument_type": "EQ","expiry_date": None,"strike_price": -1.0,"lot_size": 1,"tick_size": 5.0,},
        {"symbol": "UNITDSPR","exchange_id": 1,"data_provider_id": 1,"token": "10447","name": "UNITDSPR","instrument_type": "EQ","expiry_date": None,"strike_price": -1.0,"lot_size": 1,"tick_size": 5.0,},
        {"symbol": "UTINIFTETF","exchange_id": 1,"data_provider_id": 1,"token": "10511","name": "UTINIFTETF","instrument_type": "EQ","expiry_date": None,"strike_price": -1.0,"lot_size": 1,"tick_size": 1.0,},
        {"symbol": "CONS","exchange_id": 1,"data_provider_id": 1,"token": "10512","name": "CONS","instrument_type": "EQ","expiry_date": None,"strike_price": -1.0,"lot_size": 1,"tick_size": 1.0,},
        {"symbol": "RAMAPHO","exchange_id": 1,"data_provider_id": 1,"token": "10568","name": "RAMAPHO","instrument_type": "EQ","expiry_date": None,"strike_price": -1.0,"lot_size": 1,"tick_size": 1.0,},
    ]
    # fmt: on


@pytest.fixture
def uplink_binary_and_decoded_data() -> tuple[bytes, dict[str, Any]]:
    """
    Valid binary data and decoded data for the uplink socket
    """
    binary_data = b"\x08\x01\x122\n\x13BSE_EQ|INE467B01029\x12\x1b\n\x19\t\x00\x00\x00\x00\x80\x93\xad@\x10\xea\x91\xf7\xbf\xd22!ffff\xe6\x83\xad@\x18\xce\x82\xb3\xae\xd32"
    data_to_save = {
        "symbol": "TCS",
        "exchange_id": 2,
        "data_provider_id": 2,
        "last_traded_price": 3785.75,
        "last_traded_timestamp": "1740132698346",
        "last_traded_quantity": 0,
        "close_price": 3777.95,
    }
    return binary_data, data_to_save


@pytest.fixture
def uplink_invalid_binary_and_decoded_data():
    """
    Invalid binary data and decoded data for the uplink socket, with missing fields such as `ltt`
    """
    binary_data = b"\x08\x01\x12+\n\x13BSE_EQ|INE467B01029\x12\x14\n\x12\t\x00\x00\x00\x00\x80\x93\xad@!ffff\xe6\x83\xad@\x18\xce\x82\xb3\xae\xd32"
    decoded_data = {
        "type": "live_feed",
        "feeds": {"BSE_EQ|INE467B01029": {"ltpc": {"ltp": 3785.75, "cp": 3777.95}}},
        "currentTs": "1740364366158",
    }
    return binary_data, decoded_data
