from app.utils.fetch_data import get_env_var

# Constants for the application
SERVICE_NAME = "Market Data Service"
API_VERSION = "1.0.0"


# Authentication Constants
EMAIL = "email"
USER_ID = "user_id"


# Secret keys for JWT tokens
JWT_SECRET = get_env_var("JWT_SECRET_KEY")
JWT_REFRESH_SECRET = get_env_var("JWT_REFRESH_SECRET_KEY")
JWT_HASHING_ALGO = get_env_var("JWT_HASHING_ALGO")

# Define token expiration times
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_MINUTES = 7 * 24 * 60

# Database Constants
INSERTION_BATCH_SIZE = 1000

try:
    MACHINE_ID = int(get_env_var("MACHINE_ID"))
except ValueError as e:
    raise ValueError("MACHINE_ID must be an integer") from e

# Rate Limiter Constants
TIMES = 1
SECONDS = 120


# UPLINK CONSTANTS
UPLINK_ACCESS_TOKEN = "uplink_access_token"

# Kafka Constants
KAFKA_BROKER_URL = "KAFKA_BROKER_URL"
KAFKA_TOPIC_INSTRUMENT = "KAFKA_TOPIC_INSTRUMENT"
KAFKA_CONSUMER_GROUP_ID = "KAFKA_CONSUMER_GROUP_ID"

# Websocket Server
CHANNEL_PREFIX = "stock:"

KAFKA_CONSUMER_DEFAULT_CONFIG = {
    "auto.offset.reset": "earliest",
    "enable.auto.commit": True,
    "broker.address.family": "v4",
    "session.timeout.ms": 30000,  # 30 seconds
    "fetch.min.bytes": 1,
    "fetch.wait.max.ms": 500,
}
