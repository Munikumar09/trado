"""
This script contains the CRUD operations for all the tables in the PostgreSQL database.
Most of the functions are generic and can be used for any table in the database.The 
functions are used to perform Insert, Update, Delete and Select operations on the tables.
"""

from pathlib import Path
from typing import Any, Sequence, cast

from fastapi import HTTPException, status
from sqlalchemy import Table
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.postgresql.dml import Insert as PostgresInsert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.sqlite.dml import Insert as SQLiteInsert
from sqlalchemy.sql.elements import BinaryExpression
from sqlmodel import Session, SQLModel, or_, select

from app.data_layer.database.db_connections.postgresql import with_session
from app.utils.common.logger import get_logger
from app.utils.constants import INSERTION_BATCH_SIZE

logger = get_logger(Path(__file__).name)

################### CRUD Operations ###################


def validate_model_attributes(model: type[SQLModel], attributes: dict[str, Any]):
    """
    Validate the attributes of the model against the provided attributes.

    Parameters
    ----------
    model: ``SQLModel``
        The SQLAlchemy model class to validate the attributes
    attributes: ``dict[str, Any]``
        The attributes to validate against the model

    Raises
    ------
    ``HTTPException``:
        If the model does not have the specified attributes or if the attributes are
        of the wrong type
    """
    if not attributes:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "No attributes provided for validation"
        )
    for key in attributes.keys():
        model_name = model.__name__
        if not hasattr(model, key):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Attribute {key} not found in {model_name} model",
            )

        # Check if the attribute type matches the model attribute type
        if not model.__annotations__[key] is type(attributes[key]):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Attribute {key} is not of type {model.__annotations__[key]} in {model_name} model",
            )


def get_conditions_list(
    model: type[SQLModel], condition_attributes: dict[str, str]
) -> list[BinaryExpression]:
    """
    Generate a list of conditions based on the provided attributes. This function takes
    a dictionary of attributes and their corresponding values as input. It generates a
    list of SQLAlchemy BinaryExpression objects, which can be used as conditions in a
    database query.

    Parameters
    ----------
    sql_model: ``SQLModel``
        The SQLAlchemy model class used to generate the conditions
    condition_attributes: ``dict[str, str]``
        A dictionary of attribute names and their corresponding values

    Returns
    -------
    conditions: ``list[BinaryExpression]``
        A list of SQLAlchemy BinaryExpression objects
    """
    return [getattr(model, key) == value for key, value in condition_attributes.items()]


@with_session
def get_data_by_any_condition(
    model: type[SQLModel], session: Session, **kwargs
) -> Sequence[SQLModel]:
    """
    Retrieve a list of SQLModel objects based on the specified conditions.
    The possible keyword arguments are the attributes of the SQLModel model.
    The function returns a list of SQLModel objects that match any of the
    specified conditions. Refer the `app/data_layer/database/models` module
    for all the available models and their attributes.

    Parameters
    ----------
    sql_model: ``SQLModel``
        The SQLModel model class used to query the table in the database
    **kwargs: ``Dict[str, str]``
        The attribute names and their corresponding values to filter the data.
        The attribute names should be the columns of the SQLModel model

    Returns
    -------
    result: ``List[SQLModel]``
        A list of SQLModel objects that match the any of the specified conditions

    >>> Example:
    >>> get_smartapi_tokens_by_any_condition(Instrument, symbol="INFY", exchange="NSE")
    >>> [SQLModel(symbol='INFY', exchange='NSE', token='1224', ...),
            SQLModel(symbol='TCS', exchange='NSE', token='1225', ...), ...]

    The above example will return all the SQLModel objects with symbol 'INFY' or
    exchange 'NSE'.
    """
    validate_model_attributes(model, kwargs)
    conditions = get_conditions_list(model, kwargs)

    statement = select(model).where(or_(*conditions))
    result = session.exec(statement).all()

    return result


@with_session
def get_data_by_all_conditions(
    model: type[SQLModel],
    session: Session,
    **kwargs,
) -> Sequence[SQLModel]:
    """
    Retrieve a list of SQLModel objects based on the specified conditions.
    The possible keyword arguments are the attributes of the SQLModel model.
    The function returns a list of SQLModel objects that match all of the
    specified conditions.

    Parameters
    ----------
    sql_model: ``SQLModel``
        The SQLAlchemy model class used to query the table in the database
    session: ``Session``
        The SQLModel session object to use for the database operations
    **kwargs: ``Dict[str, str]``
        The attribute names and their corresponding values to filter the data.
        The attribute names should be the columns of the SQLModel model

    Returns
    -------
    result: ``List[SQLModel]``
        A list of SQLModel objects that match the all of the specified conditions

    >>> Example:
    >>> get_smartapi_tokens_by_all_conditions(Instrument, symbol="INFY", exchange="NSE")
    >>> [SQLModel(symbol='INFY', exchange='NSE', token='1224', ...)]

    The above example will return all the SQLModel objects with symbol 'INFY' and
    exchange 'NSE'.
    """
    validate_model_attributes(model, kwargs)
    conditions = get_conditions_list(model, kwargs)
    statement = select(model).where(*conditions)
    result = session.exec(statement).all()

    return result


@with_session
def _upsert(
    model: type[SQLModel],
    upsert_data: list[dict[str, Any]],
    session: Session,
):
    """
    Upsert means insert the data into the table if it does not already exist.
    If the data already exists, it will be updated with the new data.

    Note
    ----
    This function is a private function and should not be used directly.
    Use the `insert_data` function to upsert data into the table.

    Parameters
    ----------
    model: ``SQLModel``
        The SQLAlchemy model class to use for the upsert operation
    upsert_data: ``list[dict[str, Any]]``
        The data to upsert into the table
    session: ``Session``
        The SQLAlchemy session object to use for the database operations


    Example:
    --------
    >>> If the table has the following data:
    | id | symbol | price |
    |----|--------|-------|
    | 1  | AAPL   | 100   |
    | 2  | MSFT   | 200   |

    >>> If the following data is upserted:
    | id | symbol | price |
    |----|--------|-------|
    | 1  | AAPL   | 150   |
    | 3  | GOOGL  | 300   |

    >>> The table will be updated as:
    | id | symbol | price |
    |----|--------|-------|
    | 1  | AAPL   | 150   |
    | 2  | MSFT   | 200   |
    | 3  | GOOGL  | 300   |
    """
    if session.bind is None:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Session is not bound to a database connection",
        )

    db_type = session.bind.dialect.name
    table: Table = model.__table__  # type: ignore
    upsert_stmt: SQLiteInsert | PostgresInsert

    # Handle database-specific logic
    if db_type == "sqlite":

        # Using batch insertion due to SQLite's limitation to handle large data
        # in a single transaction
        for i in range(0, len(upsert_data), INSERTION_BATCH_SIZE):

            try:
                batch_data = upsert_data[i : i + INSERTION_BATCH_SIZE]
                upsert_stmt = sqlite_insert(table).values(batch_data)

                # SQLite requires `DO UPDATE SET` without `index_elements`
                # SQLite automatically updates the row based on the primary key
                columns = {
                    column.name: getattr(upsert_stmt.excluded, column.name)
                    for column in table.columns
                }
                upsert_stmt = upsert_stmt.on_conflict_do_update(
                    index_elements=[key.name for key in table.primary_key],
                    set_=columns,
                )
                # Execute the statement and commit the transaction
                session.exec(upsert_stmt)  # type: ignore
                session.commit()  # Commit after each successful batch
            except Exception as e:
                session.rollback()  # Rollback failed batch
                logger.error(
                    "Failed to upsert batch %s due to error: %s",
                    (i // INSERTION_BATCH_SIZE + 1),
                    e,
                )
    elif db_type == "postgresql":
        # PostgreSQL supports `ON CONFLICT DO UPDATE`
        upsert_stmt = postgres_insert(table).values(upsert_data)
        columns = {
            column.name: getattr(upsert_stmt.excluded, column.name)
            for column in table.columns
            if column.name not in table.primary_key  # Exclude primary key columns
        }
        upsert_stmt = upsert_stmt.on_conflict_do_update(
            index_elements=[key.name for key in table.primary_key],
            set_=columns,
        )
        session.exec(upsert_stmt)  # type: ignore
        session.commit()
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


@with_session
def _insert_or_ignore(
    model: type[SQLModel],
    data_to_insert: list[dict[str, Any]],
    session: Session,
):
    """
    Add the provided data into the given table if the data does not already exist.
    If the data already exists, it will be ignored.

    Parameters
    ----------
    model: ``SQLModel``
        The SQLAlchemy model class to use for the insert operation
    data_to_insert: ``list[dict[str, Any]]``
        The data to insert into the table
    session: ``Session``
        The SQLModel session object to use for the database operations
    """
    if session.bind is None:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Session is not bound to a database connection",
        )
    db_name = session.bind.dialect.name
    table: Table = model.__table__  # type: ignore
    insert_stmt: PostgresInsert | SQLiteInsert

    if db_name == "sqlite":
        for i in range(0, len(data_to_insert), INSERTION_BATCH_SIZE):
            try:
                batch_data = data_to_insert[i : i + INSERTION_BATCH_SIZE]
                insert_stmt = sqlite_insert(table).values(batch_data)
                insert_stmt = insert_stmt.on_conflict_do_nothing()
                session.exec(insert_stmt)  # type: ignore
                session.commit()  # Commit after each successful batch
            except Exception as e:
                session.rollback()
                logger.error(
                    "Failed to insert batch %s due to error: %s",
                    (i // INSERTION_BATCH_SIZE + 1),
                    e,
                )

    elif db_name == "postgresql":
        insert_stmt = postgres_insert(table).values(data_to_insert)
        insert_stmt = insert_stmt.on_conflict_do_nothing()
        session.exec(insert_stmt)  # type: ignore

    else:
        raise ValueError(f"Unsupported database type: {db_name}")

    session.commit()


@with_session
def insert_data(
    model: type[SQLModel],
    data: SQLModel | dict[str, Any] | list[SQLModel | dict[str, Any]] | None,
    session: Session,
    update_existing: bool = False,
) -> bool:
    """
    Insert the provided data into the SQLModel table in the SQLite database. It
    will handle both single and multiple data objects. If the data already exists in the table,
    it will either update the existing data or ignore the new data based on the value of the
    `update_existing` parameter

    Parameters
    ----------
    model: ``SQLModel``
        The SQLAlchemy model class to use for the insert operation
    data: ``SQLModel | dict[str, Any] | list[SQLModel | dict[str, Any]] | None``
        The data to insert into the table
    session: ``Session``
        The SQLModel session object to use for the database operations. If not provided,
        a new session will be created from the database connection pool
    update_existing: ``bool``, ( defaults = False )
        If True, the existing data in the table will be updated with the new data
    """
    if not data:
        logger.warning("Provided data is empty. Skipping insertion.")
        return False

    if isinstance(data, (SQLModel, dict)):
        data = [data]

    # Convert list of SQLModel to a list of dicts
    data_to_insert = cast(
        list[dict[str, Any]],
        [item.model_dump() if isinstance(item, SQLModel) else item for item in data],
    )

    if update_existing:
        _upsert(model, data_to_insert, session=session)
    else:
        _insert_or_ignore(model, data_to_insert, session=session)

    return True
