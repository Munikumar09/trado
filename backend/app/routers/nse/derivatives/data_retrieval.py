from fastapi import HTTPException

from app.routers.nse.derivatives.data_processor import filter_option_chain
from app.schemas.option_model import ExpiryOptionData
from app.utils.fetch_data import fetch_data
from app.utils.urls import INDEX_OPTION_CHAIN_URL, STOCK_OPTION_CHAIN_URL


def get_option_chain(
    expiry_date: str, derivative_symbol: str, derivative_type: str
) -> ExpiryOptionData:
    """
    Fetch the option chain data of the given symbol from the Nse Website.

    Parameters:
    -----------
    expiry_date: ``str``
        Option expiry date in "dd-MM-yyyy" format
            eg: 28-Sep-2023
    derivative_symbol: ``str``
        derivative symbol to get the option chain
    derivative_type: ``str``
        The derivative type that is either "stock" or "index"

    Raises:
    -------
    ``HTTPException``
        If there is no expiry for the given derivative on the given expiry date

    Return:
    -------
    ``ExpiryOptionData``
        Option expiry data that contains the option chain information about the given derivative
    """
    base_url = INDEX_OPTION_CHAIN_URL

    if derivative_type == "stock":
        base_url = STOCK_OPTION_CHAIN_URL

    option_chain_url = f"{base_url}{derivative_symbol}"
    option_chain_data = fetch_data(option_chain_url)

    if expiry_date not in option_chain_data["records"]["expiryDates"]:
        raise HTTPException(
            status_code=400,
            detail={"Error": f"No expiry for {derivative_symbol} on {expiry_date}"},
        )

    filtered_strike_price_data = [
        record
        for record in option_chain_data["records"]["data"]
        if record["expiryDate"] == expiry_date
    ]
    expiry_option_data = filter_option_chain(filtered_strike_price_data, expiry_date)

    return expiry_option_data
