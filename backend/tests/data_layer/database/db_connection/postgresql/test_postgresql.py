# pylint: disable=unused-argument
from pathlib import Path
from urllib.parse import quote_plus

import psycopg2
import pytest
from pytest import MonkeyPatch
from sqlalchemy import inspect
from sqlmodel import create_engine, select

from app.data_layer.database.db_connections.postgresql import (
    create_db_and_tables,
    get_session,
)
from app.data_layer.database.models import (
    DataProvider,
    Exchange,
    Instrument,
    InstrumentPrice,
    User,
    UserVerification,
)
from app.utils.common.logger import get_logger
from app.utils.fetch_data import get_required_env_var

logger = get_logger(Path(__file__).name)

POSTGRES_USER = "POSTGRES_USER"
POSTGRES_PASSWORD = "POSTGRES_PASSWORD"
POSTGRES_HOST = "POSTGRES_HOST"
POSTGRES_PORT = "POSTGRES_PORT"
POSTGRES_DB = "POSTGRES_DB"

model_classes = {
    "instrument": Instrument,
    "instrumentprice": InstrumentPrice,
    "user": User,
    "userverification": UserVerification,
    "dataprovider": DataProvider,
    "exchange": Exchange,
}


def create_database_if_not_exists() -> bool:
    """
    Creates the database if it doesn't already exist and returns True if created, False otherwise.
    """
    conn = None
    try:
        conn = psycopg2.connect(
            user=get_required_env_var(POSTGRES_USER),
            password=get_required_env_var(POSTGRES_PASSWORD),
            host=get_required_env_var(POSTGRES_HOST),
            port=get_required_env_var(POSTGRES_PORT),
            database="postgres",
        )
        conn.autocommit = True

        cursor = conn.cursor()
        db_name = get_required_env_var(POSTGRES_DB)
        quoted_db_name = quote_plus(db_name)

        cursor.execute("SELECT 1 FROM pg_database WHERE datname=%s;", (db_name,))
        db_exists = cursor.fetchone()

        if not db_exists:
            cursor.execute(f"CREATE DATABASE {quoted_db_name}")
            logger.info("Database '%s' created.", db_name)
            return True  # Return True if the database was created

        logger.info("Database '%s' already exists.", db_name)
        return False  # Return false if the database was not created

    except psycopg2.Error as e:
        logger.error("Error creating/checking database: %s", e)
        raise
    finally:
        if conn:
            conn.close()


def drop_database_if_created(created: bool):
    """
    Drops the database only if it was created by the create_database_if_not_exists function.
    """
    if not created:
        logger.info("Skipping database drop as it was not created in this session.")
        return

    conn = None
    try:

        conn = psycopg2.connect(
            host=get_required_env_var(POSTGRES_HOST),
            port=get_required_env_var(POSTGRES_PORT),
            user=get_required_env_var(POSTGRES_USER),
            password=get_required_env_var(POSTGRES_PASSWORD),
            database="postgres",
        )
        conn.autocommit = True

        cursor = conn.cursor()
        db_name = get_required_env_var(POSTGRES_DB)
        quoted_db_name = quote_plus(db_name)

        # Terminate other sessions (same as before)
        cursor.execute(
            """
            SELECT pg_terminate_backend(pg_stat_activity.pid)
            FROM pg_stat_activity
            WHERE pg_stat_activity.datname = %s;
            """,
            (db_name,),
        )

        cursor.execute(f"DROP DATABASE {quoted_db_name}")
        logger.info("Database '%s' dropped.", db_name)

    except psycopg2.Error as e:
        logger.error("Error dropping database: %s", e)
        raise
    finally:
        if conn:
            conn.close()


@pytest.fixture
def set_env_vars(monkeypatch: MonkeyPatch):
    """
    Set the environment variables required for the tests.
    """
    monkeypatch.setenv(POSTGRES_USER, "testuser")
    monkeypatch.setenv(POSTGRES_PASSWORD, "testuser123")
    monkeypatch.setenv(POSTGRES_HOST, "localhost")
    monkeypatch.setenv(POSTGRES_PORT, "5432")
    monkeypatch.setenv(POSTGRES_DB, "test_db")


# Test the creation of the database and tables
def test_create_db_and_tables(set_env_vars):
    """
    Test the create_db_and_tables function to ensure it creates the tables.
    """
    created = create_database_if_not_exists()
    try:
        db_url = (
            f"postgresql://{quote_plus(get_required_env_var(POSTGRES_USER))}:"
            f"{quote_plus(get_required_env_var(POSTGRES_PASSWORD))}@"
            f"{get_required_env_var(POSTGRES_HOST)}:"
            f"{get_required_env_var(POSTGRES_PORT)}/"
            f"{get_required_env_var(POSTGRES_DB)}"
        )
        engine = create_engine(db_url, echo=True)
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        # Ensure the database is created and the tables are empty
        assert tables == []

        create_db_and_tables(engine)
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        # Ensure the tables are created
        assert set(tables) == set(model_classes.keys())

        with get_session(engine) as session:
            # Check if the session is active and able to connect to the database
            assert session.is_active
            assert session.connection
            assert session.bind
            try:
                for model_class in model_classes.values():
                    assert not session.exec(select(model_class)).all()

            except Exception as e:
                raise AssertionError(
                    f"Failed to interact with the database: {e}"
                ) from e
    finally:

        drop_database_if_created(created)
