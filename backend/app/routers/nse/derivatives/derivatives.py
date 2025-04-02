from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query

from app.routers.nse.derivatives.data_retrieval import get_option_chain
from app.schemas.option_model import ExpiryOptionData
from app.utils.validators import (
    validate_and_reformat_date,
    validate_derivative_symbol_with_type,
)

router = APIRouter(prefix="/nse/derivatives", tags=["derivatives"])


@router.get(
    "/{derivative_symbol}",
    response_model=ExpiryOptionData,
)
async def option_chain_data(
    derivative_symbol: Annotated[str, Path()],
    expiry_date: Annotated[
        str,
        Query(
            example="28-Sep-2023",
        ),
    ],
    derivative_type: Annotated[
        str,
        Query(
            examples={
                "stock": {"value": "stock", "description": "Option chain for stock"},
                "index": {"value": "index", "description": "option chain for index"},
            }
        ),
    ],
):
    """
    Provide the option chain data for a given derivative symbol and expiry date.

    Parameters:
    -----------
    - **derivative_symbol**:
        Symbol of a derivative, for which you want option chain data.
        It should be available in nse official website
        eg: `NIFTY`, `TCS`
    - **expiry date**:
        Expiry date of the derivate contract.
        Based on the expiry date the option chain date will be provided.
        eg: `28-sep-2023` or `28/09/2023`
    - **derivative type**:
        It should be `stock` or `index`.

    """
    validate_derivative_symbol_with_type(derivative_symbol, derivative_type)
    formatted_expiry_date, is_date_valid = validate_and_reformat_date(expiry_date)

    if not is_date_valid:
        raise HTTPException(
            status_code=400,
            detail={
                "Error": f"{expiry_date} is not valid. It should be dd-MM-yyyy. eg, 28-Sep-2023"
            },
        )

    return get_option_chain(formatted_expiry_date, derivative_symbol, derivative_type)
