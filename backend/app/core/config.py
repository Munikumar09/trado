
from pathlib import Path
from typing import List
from urllib.parse import quote_plus

import sys
import os
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings,SettingsConfigDict
from registrable import Registrable

from app.utils.common.logger import get_logger

logger = get_logger(Path(__file__).name)

def is_test_env() -> bool:
    # Explicit flag wins
    if os.getenv("TEST_ENV", "").lower() in {"1", "true", "yes"}:
        return True
    # Fallback: running under pytest
    if "pytest" in sys.modules:
        return True
    return False

IS_TEST_ENV = is_test_env()

class AppBaseSetting(BaseSettings, Registrable):
    """
    Base class for all application settings combining Pydantic BaseSettings with Registrable.

    Attributes
    ________
    Config: ``type``
        Pydantic configuration that sets env file, ignores extra fields, and excludes
        internal registration details from dumps.
    """
    model_config=SettingsConfigDict(
        env_file= ".test.env" if IS_TEST_ENV else ".env",
        extra="ignore",
        
    )


@Registrable.register("brevo_settings")
class BrevoSettings(AppBaseSetting):
    """
    Settings for Brevo (email) integration.

    Attributes
    ________
    sender_name: ``str``
        Display name used as the sender in outgoing emails.
    sender_email: ``str``
        Email address used as the sender in outgoing emails.
    api_key: ``str``
        API key to authenticate with Brevo.
    """

    sender_name: str = Field(alias="BREVO_SENDER_NAME")
    sender_email: str = Field(alias="BREVO_SENDER_EMAIL")
    api_key: str = Field(alias="BREVO_API_KEY")


class PostgresSettings(AppBaseSetting):
    """
    Settings for the PostgreSQL database connection.

    Attributes
    ________
    user: ``str``
        Database user name.
    password: ``str``
        Database user password.
    host: ``str``
        Database host name or IP address.
    port: ``str``
        Database port as a string (e.g., "5432").
    db: ``str``
        Database name.
    """

    user: str = Field(alias="POSTGRES_USER")
    password: str = Field(alias="POSTGRES_PASSWORD")
    host: str = Field(alias="POSTGRES_HOST")
    port: str = Field(alias="POSTGRES_PORT")
    db: str = Field(alias="POSTGRES_DB")

    @property
    def url(self) -> str:
        """
        Get the PostgreSQL database connection URL.
        """
        return f"postgresql://{quote_plus(self.user)}:{quote_plus(self.password)}@{self.host}:{self.port}/{self.db}"


class JWTSettings(AppBaseSetting):
    """
    Settings for JWT token generation and validation.

    Attributes
    ________
    secret_key: ``str``
        Secret used to sign access tokens.
    refresh_secret_key: ``str``
        Secret used to sign refresh tokens.
    hashing_algo: ``str``
        Hashing algorithm identifier (e.g., HS256).
    machine_id: ``int``
        Numeric machine identifier to include in tokens or for sharding.
    """

    secret_key: str = Field(alias="JWT_SECRET_KEY")
    refresh_secret_key: str = Field(alias="JWT_REFRESH_SECRET_KEY")
    hashing_algo: str = Field(alias="JWT_HASHING_ALGO")
    machine_id: int = Field(alias="MACHINE_ID")


class SmartAPISettings(AppBaseSetting):
    """
    Settings for SmartAPI integration.

    Attributes
    ________
    api_key: ``str``
        API key for SmartAPI.
    client_id: ``str``
        Client identifier for SmartAPI.
    password: ``str``
        Password associated with the SmartAPI client.
    token: ``str``
        Session or access token for SmartAPI.
    connection_num: ``int``
        Optional connection slot or worker number.
    """

    api_key: str = Field(alias="SMARTAPI_API_KEY")
    client_id: str = Field(alias="SMARTAPI_CLIENT_ID")
    password: str = Field(alias="SMARTAPI_PWD")
    token: str = Field(alias="SMARTAPI_TOKEN")
    connection_num: int = -1


class UplinkSettings(AppBaseSetting):
    """
    Settings for Uplink integration.

    Attributes
    ________
    api_key: ``str``
        API key for Uplink.
    secret_key: ``str``
        Secret key for Uplink.
    totp_key: ``str``
        TOTP seed used for generating one-time passwords.
    pin: ``str``
        User PIN associated with the Uplink account.
    mobile_no: ``str``
        Registered mobile number for Uplink.
    connection_num: ``int``
        Optional connection slot or worker number.
    """

    api_key: str = Field(alias="UPLINK_API_KEY")
    secret_key: str = Field(alias="UPLINK_SECRET_KEY")
    totp_key: str = Field(alias="UPLINK_TOTP_KEY")
    pin: str = Field(alias="UPLINK_PIN")
    mobile_no: str = Field(alias="UPLINK_MOBILE_NO")
    connection_num: int = -1


class RedisSettings(AppBaseSetting):
    """
    Settings for Redis connection.

    Attributes
    ________
    host: ``str``
        Redis server host name or IP address.
    port: ``int``
        Redis server port.
    db: ``int``
        Logical Redis database index.
    """

    host: str = Field(alias="REDIS_HOST")
    port: int = Field(alias="REDIS_PORT")
    db: int = Field(alias="REDIS_DB")


@Registrable.register("kafka_settings")
class KafkaSettings(AppBaseSetting):
    """
    Settings for Kafka producer/consumer configuration.

    Attributes
    ________
    topic: ``str``
        Topic used for instrument messages.
    brokers: ``str``
        Comma-separated list of Kafka broker URLs.
    group_id: ``str``
        Consumer group identifier.
    """

    topic: str = Field(alias="KAFKA_TOPIC_INSTRUMENT")
    brokers: str = Field(alias="KAFKA_BROKER_URL")
    group_id: str = Field(alias="KAFKA_CONSUMER_GROUP_ID")


class WebSocketSettings(AppBaseSetting):
    """
    Settings for the WebSocket server.

    Attributes
    ________
    host: ``str``
        WebSocket server host name or IP address.
    port: ``int``
        WebSocket server port.
    """

    host: str = Field(alias="WEBSOCKET_HOST")
    port: int = Field(alias="WEBSOCKET_PORT")


class MiddlewareSettings(AppBaseSetting):
    """
    Settings for API middleware.

    Attributes
    ________
    cors_origins: ``List[str]``
        Allowed origins for CORS middleware.
    """

    cors_origins: List[str] = Field(alias="CORS_ORIGINS")


class Settings(AppBaseSetting):
    """
    Aggregated application settings composed of sub-settings.

    Attributes
    ________
    root_path: ``str``
        Filesystem or URL root path for the application.
    brevo_config: ``BrevoSettings``
        Email (Brevo) configuration.
    postgres_config: ``PostgresSettings``
        PostgreSQL database configuration.
    jwt_config: ``JWTSettings``
        JWT token configuration.
    smartapi_config: ``SmartAPISettings``
        SmartAPI integration configuration.
    uplink_config: ``UplinkSettings``
        Uplink integration configuration.
    redis_config: ``RedisSettings``
        Redis cache/store configuration.
    kafka_config: ``KafkaSettings``
        Kafka messaging configuration.
    websocket_config: ``WebSocketSettings``
        WebSocket server configuration.
    middleware_config: ``MiddlewareSettings``
        Middleware (e.g., CORS) configuration.
    """

    root_path: str = Field(alias="ROOT_PATH")
    brevo_config: BrevoSettings = BrevoSettings()

    postgres_config: PostgresSettings = PostgresSettings()

    jwt_config: JWTSettings = JWTSettings()
    smartapi_config: SmartAPISettings = SmartAPISettings()
    uplink_config: UplinkSettings = UplinkSettings()

    redis_config: RedisSettings = RedisSettings()
    kafka_config: KafkaSettings = KafkaSettings()
    websocket_config: WebSocketSettings = WebSocketSettings()
    middleware_config: MiddlewareSettings = MiddlewareSettings()


try:
    settings = Settings()
except ValidationError as e:
    logger.error("❌ Environment variable validation failed: %s", e)
    raise e
