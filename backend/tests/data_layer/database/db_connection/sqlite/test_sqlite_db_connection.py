""" 
Test the database connection and interaction with the SQLite database.
"""

from sqlmodel import create_engine, inspect, select

from app.data_layer.database.db_connections.sqlite import (
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

model_classes = {
    "instrument": Instrument,
    "instrumentprice": InstrumentPrice,
    "user": User,
    "userverification": UserVerification,
    "dataprovider": DataProvider,
    "exchange": Exchange,
}


def test_database_init_and_interaction():
    """
    Test if the database is created and tables are created and empty and able to interact with the database.
    """
    engine = create_engine("sqlite:///:memory:")
    create_db_and_tables(engine)

    # Check if the tables are created
    db_tables = inspect(engine).get_table_names()
    assert set(db_tables) == set(model_classes.keys())

    with get_session(engine) as session:
        assert session.is_active
        assert session.connection
        assert session.bind

        # Check if the tables are empty
        try:
            for model_class in model_classes.values():
                assert not session.exec(select(model_class)).all()

        except Exception as e:
            raise AssertionError(f"Failed to interact with the database: {e}") from e
