"""
This module provides the SmartapiCredentialManager class, a credential management utility
for securely handling authentication and session management with the SmartAPI platform
(Angel Broking). It supports credential caching, session renewal, and header generation
for API requests, leveraging Redis for efficient credential reuse and expiry management.

Features:
- Securely stores and manages SmartAPI credentials (API key, client ID, password, token, etc.).
- Caches credentials in Redis to minimize redundant authentication and improve performance.
- Automatically handles credential expiry and renewal based on SmartAPI's session policies
  (e.g., daily expiry at 5am IST).
- Generates appropriate headers for SmartAPI requests with proper authentication tokens.
- Integrates with a base credential manager for extensibility and code reuse.
- Supports multi-connection scenarios for load balancing or parallel API usage.
- Uses TOTP (Time-based One-Time Password) for secure authentication.
"""

import threading
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pyotp
import redis
from omegaconf import DictConfig
from redis import Redis
from SmartApi import SmartConnect

from app.data_layer.data_models.credential_model import (
    SmartAPICredentialInput,
    SmartAPICredentialOutput,
)
from app.utils.common.logger import get_logger
from app.utils.credentials.base_credential_manager import CredentialManager
from app.utils.fetch_data import get_env_var
from app.utils.redis_utils import RedisSyncConnection

logger = get_logger(Path(__file__).name)


@CredentialManager.register("smartapi_credential_manager")
class SmartapiCredentialManager(
    CredentialManager[SmartAPICredentialInput, SmartAPICredentialOutput]
):
    """
    Credentials class to store the credentials required to authenticate the SmartAPI connection.

    Attributes:
    -----------
    credential_input: ``SmartAPICredentialInput``
        The input credentials required to generate the output credentials
    credentials: ``SmartAPICredentialOutput``
        The output credentials generated from the input credentials
    """

    _connection_lock = threading.Lock()
    total_connections = 3
    current_connection = 0

    def __init__(
        self,
        credential_input: SmartAPICredentialInput,
        credentials: SmartAPICredentialOutput,
    ) -> None:
        self.credential_input = credential_input
        self.credentials = credentials

    def check_cache(
        self, redis_client: Redis, key: str
    ) -> SmartAPICredentialOutput | None:
        """
        Check if the credentials are present in the cache.

        Parameters:
        -----------
        redis_client: ``Redis``
            The Redis client used to interact with redis for checking the cache
        key: ``str``
            The key to check in the cache

        Returns:
        --------
        ``SmartAPICredentialOutput | None``
            The credentials if present in the cache, else None
        """
        try:
            pipe = redis_client.pipeline()
            pipe.type(key)
            pipe.hgetall(key)

            results = pipe.execute()
            redis_type, hash_data = results

            if redis_type == "hash":
                return SmartAPICredentialOutput(**hash_data)

            return None
        except redis.RedisError as e:
            logger.error("Redis error while checking cache for key %s: %s", key, e)
            return None

    def _generate_new_credentials(
        self, credential_input: SmartAPICredentialInput
    ) -> SmartAPICredentialOutput:
        """
        Generate new credentials using the SmartAPI credentials input.

        Parameters:
        -----------
        credential_input: ``SmartAPICredentialInput``
            The input credentials required to generate the output credentials

        Returns:
        --------
        ``SmartAPICredentialOutput``
            The output credentials generated from the input credentials
        """
        try:
            smart_connect = SmartConnect(credential_input.api_key)
            totp = pyotp.TOTP(credential_input.token).now()
            data = smart_connect.generateSession(
                credential_input.client_id, credential_input.pwd, totp
            )

            if "data" not in data:
                raise ValueError("Invalid response from SmartAPI: missing 'data' field")

            session_data = data["data"]
            required_fields = ["jwtToken", "refreshToken", "feedToken", "clientcode"]
            missing_fields = [
                field for field in required_fields if field not in session_data
            ]

            if missing_fields:
                raise ValueError(
                    f"Missing fields in SmartAPI response: {missing_fields}"
                )

            return SmartAPICredentialOutput(
                access_token=session_data["jwtToken"],
                refresh_token=session_data["refreshToken"],
                feed_token=session_data["feedToken"],
                user_id=session_data["clientcode"],
            )
        except Exception as e:
            logger.error("Failed to generate SmartAPI credentials: %s", e)
            raise

    def get_next_expiry_time(self) -> int:
        """
        The credentials will expire 5 am IST every day, so it will return the next expiry time.

        Returns:
        --------
        ``int``
            The next expiry time in seconds from now
        """
        ist = ZoneInfo("Asia/Kolkata")
        now = datetime.now(ist)
        next_expiry = now.replace(hour=5, minute=0, second=0, microsecond=0)

        if now > next_expiry:
            next_expiry += timedelta(days=1)

        return int((next_expiry - now).total_seconds())

    def cache_credentials(
        self, redis_client: Redis, key: str, credentials: SmartAPICredentialOutput
    ):
        """
        Cache the credentials in Redis.

        Parameters:
        -----------
        redis_client: ``Redis``
            The Redis client used to interact with redis to store the credentials
        key: ``str``
            The key to store the credentials in the cache
        credentials: ``SmartAPICredentialOutput``
            The credentials to store in the cache
        """
        try:
            pipe = redis_client.pipeline()
            pipe.hset(key, mapping=credentials.to_dict())
            ttl = self.get_next_expiry_time()
            pipe.expire(key, ttl)
            pipe.execute()
            logger.info(
                "Credentials cached successfully for key: %s for next %d seconds",
                key,
                ttl,
            )
        except redis.RedisError as e:
            logger.error("Error caching credentials for key %s: %s", key, e)

    def get_headers(self) -> dict[str, str]:
        """
        The method to get the headers for the API request.

        Returns:
        --------
        ``dict[str, str]``
            The headers for the API request for the SmartAPI
        """
        auth_token = self.credentials.access_token
        headers = {
            "Authorization": auth_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-UserType": "USER",
            "X-SourceID": "WEB",
            "X-ClientLocalIP": "CLIENT_LOCAL_IP",
            "X-ClientPublicIP": "CLIENT_PUBLIC_IP",
            "X-MACAddress": "MAC_ADDRESS",
            "X-PrivateKey": self.credential_input.api_key,
        }

        return headers

    @classmethod
    def generate_credentials(
        cls, credential_input: SmartAPICredentialInput
    ) -> SmartAPICredentialOutput:
        """
        Generate the credentials for the SmartAPI.

        Parameters:
        -----------
        credential_input: ``SmartAPICredentialInput``
            The input credentials required to generate the output credentials

        Returns:
        --------
        ``SmartAPICredentialOutput``
            The output credentials generated from the input credentials
        """
        try:
            redis_connection = RedisSyncConnection()
            client = redis_connection.get_connection()
            key = f"smartapi_credentials:{credential_input.client_id}:{credential_input.connection_num}"

            temp_instance = cls(
                credential_input,
                SmartAPICredentialOutput(
                    access_token="", refresh_token="", feed_token="", user_id=""
                ),
            )
            credentials = temp_instance.check_cache(client, key)

            if credentials:
                logger.info("Credentials found in cache for key: %s", key)
                return credentials

            logger.info("Generating new credentials for key: %s", key)
            credentials = temp_instance._generate_new_credentials(credential_input)
            temp_instance.cache_credentials(client, key, credentials)

            return credentials

        finally:
            redis_connection.close_connection()

    @classmethod
    def from_cfg(cls, cfg: DictConfig | None = None) -> "SmartapiCredentialManager":
        """
        Create a Credentials object from the credentials file.

        Parameters:
        -----------
        cfg: ``DictConfig``
            The configuration object containing the credentials

        Returns:
        --------
        ``SmartapiCredentialManager``
            The Credentials object with the credentials
        """

        if cfg is None:
            cfg = DictConfig({})

        connection_num = cfg.get("connection_num")

        if connection_num is None:
            connection_num = cls.current_connection % cls.total_connections
            logger.info(
                "Connection number is not provided, using default connection number: %d",
                connection_num,
            )
        if connection_num >= cls.total_connections:
            original_connection_num = connection_num
            connection_num = connection_num % cls.total_connections
            logger.info(
                "Connection number %d exceeds total connections %d, using modulo: %d",
                original_connection_num,
                cls.total_connections,
                connection_num,
            )

        api_key = get_env_var("SMARTAPI_API_KEY")
        client_id = get_env_var("SMARTAPI_CLIENT_ID")
        pwd = get_env_var("SMARTAPI_PWD")
        token = get_env_var("SMARTAPI_TOKEN")

        smart_credential_input = SmartAPICredentialInput(
            api_key=api_key,
            client_id=client_id,
            pwd=pwd,
            token=token,
            connection_num=connection_num,
        )
        smart_credentials = cls.generate_credentials(smart_credential_input)

        with cls._connection_lock:
            cls.current_connection = (connection_num + 1) % cls.total_connections

        return cls(
            smart_credential_input,
            smart_credentials,
        )
