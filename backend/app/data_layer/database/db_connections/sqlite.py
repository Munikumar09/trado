""" 
This script is responsible for creating a connection to the SQLite database. 
It also provides a function to create the database and tables and a function
to get a session object to interact with the database.
"""

from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generator

from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from app.utils.common.logger import get_logger
from app.utils.urls import SQLITE_DB_URL

logger = get_logger(Path(__file__).name)


sqlite_engine = create_engine(SQLITE_DB_URL)


def create_db_and_tables(db_engine: Engine | None = None):
    """
    Create the database and tables if they do not exist
    """
    logger.info("Creating database and tables")

    # If the database engine is not provided, use the default engine
    db_engine = db_engine or sqlite_engine

    try:
        SQLModel.metadata.create_all(db_engine)
        logger.info("Database and tables created successfully")
    except Exception as e:
        logger.error("Failed to create database and tables: %s", e)
        raise


@contextmanager
def get_session(db_engine: Engine | None = None) -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Ensures proper handling of commits and rollbacks.
    """
    db_engine = db_engine or sqlite_engine
    session = Session(db_engine)

    try:
        yield session
    except Exception as e:
        session.rollback()
        logger.info("Session rollback due to error: %s", e)
        raise
    finally:
        session.close()


def with_session(func: Callable) -> Callable:
    """
    This decorator function is used to provide a session object to the decorated function.
    If the session object is not provided, a new session object is created and used for the
    database operations.
    """

    @wraps(func)
    def wrapper(*args, session: Session | None = None, **kwargs) -> Any:
        # Use provided session or create a new one
        if session is None:
            with get_session() as new_session:
                return func(*args, session=new_session, **kwargs)
        return func(*args, session=session, **kwargs)

    return wrapper
