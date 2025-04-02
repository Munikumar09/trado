import uuid
from datetime import datetime
from typing import Generator

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from app.data_layer.database.db_connections.sqlite import (
    create_db_and_tables,
    get_session,
)
from app.data_layer.database.models.user_model import User
from app.routers.authentication.authenticate import get_hash_password
from app.schemas.user_model import UserSignup


# Mock data
@pytest.fixture
def sign_up_data() -> UserSignup:
    """
    Fixture to provide mock sign-up data.
    """
    return UserSignup(
        username="testuser",
        email="test@gmail.com",
        phone_number="1234567890",
        password="Password1!",
        confirm_password="Password1!",
        gender="male",
        date_of_birth="01/01/2000",
    )


# Mock user model
@pytest.fixture
def test_user(sign_up_data: UserSignup) -> User:
    """
    Fixture to provide a mock user model.
    """
    return User(
        user_id=uuid.uuid4().int % (10**11),
        username=sign_up_data.username,
        email=sign_up_data.email,
        phone_number=sign_up_data.phone_number,
        password=get_hash_password("Password1!"),
        date_of_birth=datetime.strptime(sign_up_data.date_of_birth, "%d/%m/%Y").date(),
        gender=sign_up_data.gender,
    )


@pytest.fixture
def token_data(test_user) -> dict[str, str]:
    """
    Fixture to provide mock token data.
    """
    return {"user_id": test_user.user_id, "email": test_user.email}


@pytest.fixture(scope="function")
def test_engine() -> Generator[Engine, None, None]:
    """
    Using sqlite in-memory database instead of PostgreSQL for testing.
    Because it is faster and does not require a separate database server.
    Also, the operations are similar to PostgreSQL.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_db_and_tables(engine)

    yield engine

    engine.dispose()


@pytest.fixture(scope="function")
def session(test_engine) -> Generator[Session, None, None]:
    """
    Fixture to provide a new database session for each test function.
    Ensures each test runs in isolation with a clean database state.
    """
    with get_session(test_engine) as session:
        yield session
