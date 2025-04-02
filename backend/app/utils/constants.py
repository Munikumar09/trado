from app.utils.fetch_data import get_required_env_var

# Authentication Constants
EMAIL = "email"
USER_ID = "user_id"


# Secret keys for JWT tokens
JWT_SECRET = get_required_env_var("JWT_SECRET_KEY")
JWT_REFRESH_SECRET = get_required_env_var("JWT_REFRESH_SECRET_KEY")
JWT_HASHING_ALGO = get_required_env_var("JWT_HASHING_ALGO")

# Define token expiration times
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_MINUTES = 7 * 24 * 60

# Database Constants
INSERTION_BATCH_SIZE = 1000

try:
    MACHINE_ID = int(get_required_env_var("MACHINE_ID"))
except ValueError as e:
    raise ValueError("MACHINE_ID must be an integer") from e

# Rate Limiter Constants
TIMES = 1
SECONDS = 120


# UPLINK CONSTANTS
UPLINK_ACCESS_TOKEN = "uplink_access_token"
