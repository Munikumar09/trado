# --- Standard Library Imports ---
import asyncio
import sys
from pathlib import Path

# --- Third-Party Imports ---
from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_limiter import FastAPILimiter

from app.core.app_state import AppState
from app.core.singleton import Singleton

# --- Project Imports ---
from app.data_layer.streaming.consumers.kafka_consumer import KafkaConsumer
from app.sockets.websocket_server_manager import ConnectionManager, RedisPubSubManager
from app.sockets.websocket_server_protocol import StockTickerServerFactory
from app.utils.asyncio_utils.asyncio_support import (
    AsyncioLoop,
    install_twisted_reactor,
    register_shutdown_handler,
    register_task_for_cleanup,
)
from app.utils.asyncio_utils.coro_utils import fire_and_forgot
from app.utils.common.logger import get_logger
from app.utils.constants import API_VERSION, SERVICE_NAME
from app.utils.fetch_data import get_env_var
from app.utils.redis_utils import RedisAsyncConnection
from app.utils.startup_utils import create_tokens_db

logger = get_logger(Path(__file__).name)

# --- Environment Variables ---
load_dotenv()

# --- Install Twisted Reactor Safely ---
install_twisted_reactor()
LOOP = AsyncioLoop.get_loop()

# --- Try Import Twisted ---
try:
    from twisted.internet import reactor

    TWISTED_AVAILABLE = True
except ImportError as e:
    logger.error("Failed to import Twisted components: %s", e)
    TWISTED_AVAILABLE = False
    reactor = None  # type: ignore[assignment]


class FastAPIApp(metaclass=Singleton):
    """
    FastAPI application instance.

    This class serves as a singleton for the FastAPI application, ensuring that
    only one instance of the application is created and used throughout the
    application lifecycle. It provides a convenient way to access the FastAPI
    app and manage its state.

    Attributes:
    -----------
    app: ``FastAPI``
        The FastAPI application instance.
    """

    _app: FastAPI | None = None

    @classmethod
    def get_fast_api_app(cls) -> FastAPI:
        """
        Get the FastAPI application instance.

        Returns
        -------
        ``FastAPI``
            The FastAPI application instance.
        """
        if cls._app is None:
            cls._app = FastAPI(
                title=SERVICE_NAME,
                description="Market data API with real-time WebSocket updates",
                version=API_VERSION,
            )

            # Add CORS middleware - only once when app is created
            cors_origins = get_env_var("CORS_ORIGINS", "*")
            if cors_origins is None:
                cors_origins = "*"

            cls._app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_origins.split(","),
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        return cls._app


app = FastAPIApp.get_fast_api_app()  # pylint: disable=invalid-name


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for all unhandled exceptions in the application.

    Parameters
    ----------
    request: ``Request``
        The incoming request that caused the exception.
    exc: ``Exception``
        The exception that was raised.
    """
    logger.exception("Unhandled exception in request %s: %s", request.url, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


@app.on_event("startup")
async def startup_event():
    """
    Application startup event to initialize services.

    Initializes:
    - Database for tokens
    - Redis connection pool
    - FastAPI rate limiter
    - Kafka consumer task
    - WebSocket server (if Twisted is available)

    Handles cleanup on startup failure.
    """
    logger.info("Starting %s v%s...", SERVICE_NAME, API_VERSION)

    try:
        # Initialize DB
        create_tokens_db()
        logger.info("Tokens database initialized")

        # Initialize Redis connection
        redis_connection = RedisAsyncConnection()
        AppState.redis_client = await redis_connection.get_connection()

        # Initialize rate limiter
        await FastAPILimiter.init(AppState.redis_client)
        logger.info("Request rate limiter initialized")

        kafka_consumer = KafkaConsumer()
        AppState.kafka_consumer_task = fire_and_forgot(
            kafka_consumer.consume_messages()
        )
        register_task_for_cleanup(AppState.kafka_consumer_task)
        logger.info("Kafka consumer started")

        # Start WebSocket Server if Twisted is available
        if TWISTED_AVAILABLE and reactor:
            try:
                ws_host = get_env_var("WEBSOCKET_HOST")
                ws_port = int(get_env_var("WEBSOCKET_PORT"))

                if ws_port == 8000:
                    raise ValueError(
                        "WebSocket port cannot be the same as FastAPI port (8000)"
                    )

                # Initialize managers
                pubsub_manager = RedisPubSubManager(AppState.redis_client)
                AppState.connection_manager = ConnectionManager(pubsub_manager)

                # Register cleanup handlers
                register_shutdown_handler(pubsub_manager.close)
                register_shutdown_handler(AppState.connection_manager.close)

                factory = StockTickerServerFactory(
                    f"ws://{ws_host}:{ws_port}",
                    AppState.connection_manager,
                )

                # Start listening
                listen_tcp = getattr(reactor, "listenTCP", None)
                if listen_tcp is not None:
                    AppState.websocket_server_port = listen_tcp(
                        ws_port, factory, interface=ws_host
                    )
                    AppState.websocket_server_running = True
                    logger.info(
                        "WebSocket server started on ws://%s:%d", ws_host, ws_port
                    )
                else:
                    logger.error(
                        "Twisted reactor does not support listenTCP; WebSocket server not started."
                    )

            except Exception as e:
                logger.error("Failed to start WebSocket server: %s", e, exc_info=True)
        else:
            logger.warning("Twisted unavailable. WebSocket server not started.")

        AppState.startup_complete = True
        logger.info("%s startup completed successfully", SERVICE_NAME)

    except Exception as e:
        logger.critical("Fatal error during startup: %s", e, exc_info=True)
        sys.exit(1)


@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown event to clean up resources.

    Performs cleanup:
    - Stops WebSocket server if running
    - Cancels Kafka consumer task
    - Closes Redis connections and all managed resources
    """
    if not AppState.startup_complete:
        logger.info("Shutdown called before startup completed")

    logger.info("Application shutting down, cleaning up resources...")

    # Stop WebSocket Server
    if (
        TWISTED_AVAILABLE
        and AppState.websocket_server_port
        and AppState.websocket_server_running
    ):
        try:
            logger.info("Stopping WebSocket server...")
            stop_deferred = AppState.websocket_server_port.stopListening()
            if stop_deferred:
                # Wait for up to 5 seconds for the server to stop
                try:
                    await asyncio.wait_for(
                        asyncio.ensure_future(stop_deferred), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Timed out waiting for WebSocket server to stop")
            logger.info("WebSocket server stopped")
        except Exception as e:
            logger.error("Error stopping WebSocket server: %s", e, exc_info=True)
        finally:
            AppState.websocket_server_port = None
            AppState.websocket_server_running = False

    # Stop Kafka Consumer
    if AppState.kafka_consumer_task and not AppState.kafka_consumer_task.done():
        logger.info("Stopping Kafka consumer...")
        try:
            # First, try to wait for graceful completion
            await asyncio.wait_for(AppState.kafka_consumer_task, timeout=5.0)
            logger.info("Kafka consumer stopped")
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for Kafka consumer to stop")

            # Cancel the task if it didn't stop gracefully
            AppState.kafka_consumer_task.cancel()
            try:
                await AppState.kafka_consumer_task
            except asyncio.CancelledError:
                logger.info("Kafka consumer task cancelled")
        except asyncio.CancelledError:
            logger.info("Kafka consumer task cancelled")
        except Exception as e:
            logger.error("Error during Kafka task shutdown: %s", e, exc_info=True)
        finally:
            AppState.kafka_consumer_task = None

    logger.info("%s shutdown complete", SERVICE_NAME)
