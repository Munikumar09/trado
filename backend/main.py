# --- Standard Library Imports ---
import os
from pathlib import Path

import uvicorn

# --- Third-Party Imports ---
from fastapi.responses import JSONResponse
from twisted.internet import reactor

from app.core.app_state import AppState
from app.core.application import TWISTED_AVAILABLE, app

# --- Project Imports ---
from app.routers.authentication import authentication
from app.routers.nse.derivatives import derivatives
from app.routers.nse.equity import equity
from app.routers.smartapi.smartapi import smartapi
from app.utils.asyncio_utils.asyncio_support import AsyncioLoop
from app.utils.common.logger import get_logger
from app.utils.constants import API_VERSION, SERVICE_NAME

logger = get_logger(Path(__file__).name)

# --- Routers ---
app.include_router(derivatives.router)
app.include_router(equity.router)
app.include_router(smartapi.router)
app.include_router(authentication.router)

LOOP = AsyncioLoop.get_loop()


# --- API Endpoints ---
@app.get("/", summary="Root endpoint", tags=["General"])
async def index():
    """
    Root endpoint to verify API is running
    """
    return {
        "service": SERVICE_NAME,
        "version": API_VERSION,
        "status": "running",
    }


@app.get("/health", summary="Health Check", tags=["General"])
async def health_check():
    """
    Comprehensive health check for all system components.
    Returns detailed status of each component.
    """
    # Check Kafka status
    kafka_status = "unknown"
    if AppState.kafka_consumer_task:
        kafka_consumer_task = AppState.kafka_consumer_task
        if kafka_consumer_task.cancelled():
            kafka_status = "cancelled"
        elif kafka_consumer_task.done():
            if kafka_consumer_task.exception():
                kafka_status = f"error: {str(kafka_consumer_task.exception())[:100]}"
            else:
                kafka_status = "done"
        else:
            kafka_status = "running"

    # Check WebSocket server status
    ws_status = "stopped"
    ws_connections = 0
    if AppState.websocket_server_running:
        ws_status = "listening"

    # Check Redis status
    redis_status = "error"
    redis_latency_ms = None
    try:
        if AppState.redis_client:
            start_time = AsyncioLoop.get_loop().time()
            await AppState.redis_client.ping()
            end_time = AsyncioLoop.get_loop().time()
            redis_latency_ms = round((end_time - start_time) * 1000, 2)
            redis_status = "ok"
    except RuntimeError:
        redis_status = "not_initialized"
    except Exception as e:
        logger.warning("Redis health check failed: %s", e)

    status_code = (
        200
        if kafka_status in ("running", "unknown")
        and ws_status in ("listening", "disabled", "stopped")
        and redis_status == "ok"
        else 503
    )

    # Build response with detailed component status
    health_data = {
        "status": "healthy" if status_code == 200 else "unhealthy",
        "timestamp": AsyncioLoop.get_loop().time(),
        "components": {
            "kafka_consumer": {
                "status": kafka_status,
            },
            "websocket_server": {
                "status": ws_status,
                "connections": ws_connections,
            },
            "redis": {
                "status": redis_status,
            },
        },
        "version": API_VERSION,
    }

    # Add latency if available
    if redis_latency_ms is not None:
        health_data["components"]["redis"]["latency_ms"] = redis_latency_ms

    return JSONResponse(content=health_data, status_code=status_code)


# --- Main Entry Point ---
async def start_uvicorn_server():
    """
    Start the Uvicorn server with the specified configuration.
    Handle signals gracefully.
    """
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))

    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        loop="asyncio",
        reload=os.getenv("DEBUG", "false").lower() == "true",
    )
    server = uvicorn.Server(config)

    logger.info("Starting Uvicorn server on %s:%d", host, port)
    await server.serve()


if __name__ == "__main__":
    try:
        logger.info("Initializing %s...", SERVICE_NAME)
        LOOP.run_until_complete(start_uvicorn_server())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, initiating shutdown...")
    except Exception as e:
        logger.critical("Fatal error: %s", e, exc_info=True)
    finally:
        logger.info("Stopping event loop...")
        # Only call reactor methods if Twisted is available and reactor is imported
        if TWISTED_AVAILABLE and reactor is not None:
            # Some reactors may not have 'running' or 'stop' attributes, so check first
            STOP_FN = getattr(reactor, "stop", None)
            if callable(STOP_FN) and getattr(reactor, "running", False):
                STOP_FN()  # pylint: disable=not-callable

        if not LOOP.is_closed():
            LOOP.close()
        logger.info("Application terminated")
