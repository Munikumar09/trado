"""
This script contains the CRUD operations for the Instrument table.
"""

from pathlib import Path

from sqlmodel import Session, delete, select

from app.data_layer.database.db_connections.postgresql import with_session
from app.data_layer.database.models import Instrument, InstrumentPrice
from app.utils.common.logger import get_logger

logger = get_logger(Path(__file__).name)


@with_session
def delete_all_data(session):
    """
    Deletes all data from the Instrument table.
    """
    try:
        statement = delete(Instrument)
        session.exec(statement)
        session.commit()
    except Exception as e:
        logger.error("Failed to delete all data from the Instrument table: %s", str(e))
        raise e


@with_session
def get_all_stock_price_info(session: Session) -> list[InstrumentPrice]:
    """
    Retrieve all the data from the InstrumentPrice table in the Postgresql database.

    Parameters
    ----------
    session: ``Session``
        The SQLModel session object to use for the database operations

    Returns
    -------
    ``List[InstrumentPrice]``
        The list of all the InstrumentPrice objects present in the table
    """
    stmt = select(InstrumentPrice)
    return session.exec(stmt).all()  # type: ignore
