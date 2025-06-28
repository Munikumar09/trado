# pylint: disable= protected-access, too-many-lines, import-outside-toplevel

"""
Comprehensive tests for the FastAPI application module.

This test suite covers all aspects of the application.py module including:
- FastAPIApp singleton class initialization
- FastAPI application configuration and middleware setup
- Global exception handler functionality
- Startup event handling and service initialization
- Shutdown event handling and resource cleanup
- Error handling during startup and shutdown
- Integration with external services (Redis, Kafka, WebSocket)
- Edge cases and failure scenarios

The tests are organized into logical sections:
1. FastAPIApp Singleton Tests
2. Application Configuration Tests
3. Exception Handler Tests
4. Startup Event Tests
5. Shutdown Event Tests
6. Integration Tests
7. Error Handling and Edge Cases
"""

import asyncio
from asyncio import Future
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.application import (
    FastAPIApp,
    app,
    global_exception_handler,
    shutdown_event,
    startup_event,
)
from app.utils.constants import SERVICE_NAME

# =============================================================================
# SHARED FIXTURES
# =============================================================================


@pytest.fixture
def mock_logger():
    """
    Mock logger fixture for testing log output. Provides a mock logger to verify logging
    behavior throughout the test suite.
    """
    with patch("app.core.application.logger") as mock:
        yield mock


@pytest.fixture
def mock_app_state():
    """
    Mock AppState fixture for testing application state management. Provides a mock AppState
    to isolate application state during testing.
    """
    # Store original values
    original_redis_client = None
    original_kafka_consumer_task = None
    original_connection_manager = None
    original_websocket_server_port = None
    original_websocket_server_running = None
    original_startup_complete = None

    # Import the actual AppState class
    from app.core.app_state import AppState

    # Store original values if they exist
    if hasattr(AppState, "redis_client"):
        original_redis_client = AppState.redis_client
    if hasattr(AppState, "kafka_consumer_task"):
        original_kafka_consumer_task = AppState.kafka_consumer_task
    if hasattr(AppState, "connection_manager"):
        original_connection_manager = AppState.connection_manager
    if hasattr(AppState, "websocket_server_port"):
        original_websocket_server_port = AppState.websocket_server_port
    if hasattr(AppState, "websocket_server_running"):
        original_websocket_server_running = AppState.websocket_server_running
    if hasattr(AppState, "startup_complete"):
        original_startup_complete = AppState.startup_complete

    # Set default test values
    AppState.redis_client = None
    AppState.kafka_consumer_task = None
    AppState.connection_manager = None
    AppState.websocket_server_port = None
    AppState.websocket_server_running = False
    AppState.startup_complete = False

    yield AppState

    # Restore original values
    AppState.redis_client = original_redis_client
    AppState.kafka_consumer_task = original_kafka_consumer_task
    AppState.connection_manager = original_connection_manager
    AppState.websocket_server_port = original_websocket_server_port
    AppState.websocket_server_running = original_websocket_server_running
    AppState.startup_complete = original_startup_complete


@pytest.fixture(autouse=True)
def mock_env_vars():
    """
    Mock environment variables for testing. Provides consistent environment variable
    values for testing.
    """
    env_vars = {
        "CORS_ORIGINS": "http://localhost:3000,http://localhost:8080",
        "WEBSOCKET_HOST": "localhost",
        "WEBSOCKET_PORT": "8001",
        "KAFKA_BROKER_URL": "localhost:9092",
        "REDIS_URL": "redis://localhost:6379",
    }

    with patch("app.core.application.get_env_var") as mock_get_env:
        mock_get_env.side_effect = lambda key, default=None: env_vars.get(key, default)

        yield mock_get_env


def _create_mock_redis_client():
    """
    Helper function to create consistent mock Redis client.
    """
    mock_client = AsyncMock()
    mock_client.ping = AsyncMock(return_value=True)
    mock_client.publish = AsyncMock(return_value=None)
    mock_client.close = AsyncMock(return_value=None)
    mock_client.flushdb = AsyncMock(return_value=None)

    return mock_client


@pytest.fixture
def mock_redis_connection():
    """
    Mock Redis connection fixture for testing Redis integration. Provides a mock Redis
    connection and client for isolated testing.
    """
    mock_redis_client = _create_mock_redis_client()

    with patch("app.core.application.RedisAsyncConnection") as mock_conn_class:
        mock_instance = AsyncMock()
        mock_instance.get_connection = AsyncMock(return_value=mock_redis_client)
        mock_conn_class.return_value = mock_instance

        yield {"client": mock_redis_client, "connection": mock_instance}


@pytest.fixture
def mock_kafka_consumer():
    """
    Mock Kafka consumer fixture for testing Kafka integration. Provides a mock KafkaConsumer
    for isolated testing.
    """

    async def mock_consume_messages():
        """
        Mock async function that doesn't create unawaited coroutines.
        """

    with patch("app.core.application.KafkaConsumer") as mock_consumer_class:
        mock_instance = MagicMock()
        mock_instance.consume_messages = mock_consume_messages
        mock_consumer_class.return_value = mock_instance

        yield {"class": mock_consumer_class, "instance": mock_instance}


@pytest.fixture
def mock_twisted_components():
    """
    Mock Twisted components fixture for testing WebSocket server integration. Provides mock
    Twisted reactor and related components.
    """
    mock_reactor = MagicMock()
    mock_reactor.listenTCP = MagicMock()

    mock_port = MagicMock()
    mock_port.stopListening = MagicMock()
    mock_reactor.listenTCP.return_value = mock_port

    with patch("app.core.application.reactor", mock_reactor):
        with patch("app.core.application.TWISTED_AVAILABLE", True):

            yield {"reactor": mock_reactor, "port": mock_port}


@pytest.fixture
def mock_external_dependencies(
    mock_redis_connection,
    mock_kafka_consumer,
    mock_app_state,
):
    """
    Combined fixture that mocks all external dependencies. Provides a comprehensive mock
    setup for testing application functionality without external service dependencies.
    """
    with patch("app.core.application.create_tokens_db") as mock_create_db:
        with patch("app.core.application.FastAPILimiter") as mock_limiter:
            with patch("app.core.application.fire_and_forgot") as mock_fire_and_forgot:
                with patch(
                    "app.core.application.register_task_for_cleanup"
                ) as mock_register_task:
                    with patch(
                        "app.core.application.register_shutdown_handler"
                    ) as mock_register_shutdown:
                        # Configure async mocks to return simple values
                        mock_limiter.init = AsyncMock(return_value=None)

                        # fire_and_forgot should consume the coroutine to prevent warnings
                        def mock_fire_and_forgot_func(coro):
                            # Close the coroutine to prevent RuntimeWarning
                            if hasattr(coro, "close"):
                                coro.close()

                            # Return a mock Task
                            mock_task = MagicMock()
                            mock_task.done = MagicMock(return_value=False)
                            mock_task.cancel = MagicMock()

                            return mock_task

                        mock_fire_and_forgot.side_effect = mock_fire_and_forgot_func

                        yield {
                            "create_db": mock_create_db,
                            "limiter": mock_limiter,
                            "fire_and_forgot": mock_fire_and_forgot,
                            "register_task": mock_register_task,
                            "register_shutdown": mock_register_shutdown,
                            "redis": mock_redis_connection,
                            "kafka": mock_kafka_consumer,
                            "app_state": mock_app_state,
                        }


# =============================================================================
# FASTAPI APP SINGLETON TESTS
# =============================================================================


def test_fastapi_app_singleton_behavior():
    """
    Test that FastAPIApp follows singleton pattern. Verifies that multiple calls to
    get_fast_api_app() return the same instance and that the singleton pattern is
    properly implemented.
    """
    # Clear any existing app instance
    FastAPIApp._app = None

    # Get first instance
    app1 = FastAPIApp.get_fast_api_app()

    # Get second instance
    app2 = FastAPIApp.get_fast_api_app()

    # Should be the same instance
    assert app1 is app2
    assert isinstance(app1, FastAPI)

    # Clean up
    FastAPIApp._app = None


def test_fastapi_app_initialization_with_defaults():
    """
    Test FastAPI app initialization with default configuration. Verifies that the FastAPI
    application is properly initialized with correct title, description, and version.
    """
    # Clear any existing app instance
    FastAPIApp._app = None

    with patch("app.core.application.SERVICE_NAME", "TestService"):
        with patch("app.core.application.API_VERSION", "1.0.0"):
            app_instance = FastAPIApp.get_fast_api_app()

            assert app_instance.title == "TestService"
            assert (
                app_instance.description
                == "Market data API with real-time WebSocket updates"
            )
            assert app_instance.version == "1.0.0"

    # Clean up
    FastAPIApp._app = None


def test_fastapi_app_cors_middleware_configuration():
    """
    Test CORS middleware configuration. Verifies that CORS middleware is properly configured
    with correct origins and settings from environment variables, including wildcard origins.
    """
    # Clear any existing app instance
    FastAPIApp._app = None

    app_instance = FastAPIApp.get_fast_api_app()

    # Check that CORS middleware was added
    middleware_found = False
    for middleware in app_instance.user_middleware:
        if middleware.cls == CORSMiddleware:
            middleware_found = True

            # Verify CORS configuration
            options = middleware.options
            assert options["allow_credentials"] is True
            assert options["allow_methods"] == ["*"]
            assert options["allow_headers"] == ["*"]

            # Origins should be split from environment variable
            expected_origins = ["http://localhost:3000", "http://localhost:8080"]
            assert options["allow_origins"] == expected_origins
            break

    assert middleware_found, "CORS middleware not found"

    # Test wildcard origins scenario
    with patch("app.core.application.get_env_var") as mock_get_env:
        mock_get_env.return_value = "*"
        FastAPIApp._app = None  # Reset for new configuration

        app_wildcard = FastAPIApp.get_fast_api_app()
        for middleware in app_wildcard.user_middleware:
            if middleware.cls == CORSMiddleware:
                options = middleware.options
                assert options["allow_origins"] == ["*"]
                break

    # Clean up
    FastAPIApp._app = None


# =============================================================================
# APPLICATION CONFIGURATION TESTS
# =============================================================================


def test_module_level_app_instance():
    """
    Test that the module-level app instance is correctly created. Verifies that the app variable
    at module level is a FastAPI instance created through the singleton pattern.
    """
    assert isinstance(app, FastAPI)
    assert app.title is not None
    assert app.version is not None


def test_app_exception_handler_registration():
    """
    Test that the global exception handler is properly registered. Verifies that the global
    exception handler is registered on the FastAPI application instance.
    """
    # Check that exception handler is registered
    exception_handlers = app.exception_handlers
    assert Exception in exception_handlers
    assert exception_handlers[Exception] is global_exception_handler


# =============================================================================
# EXCEPTION HANDLER TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_global_exception_handler_comprehensive(mock_logger):
    """
    Test global exception handler with different exception types and scenarios.
    Verifies that the handler works correctly with various exception types and
    returns proper JSON response format.
    """
    mock_request = MagicMock(spec=Request)
    mock_request.url = "http://localhost:8000/test"

    test_cases = [
        Exception("Test error"),
        ValueError("Invalid value"),
        KeyError("Missing key"),
        RuntimeError("Runtime error"),
        ConnectionError("Connection failed"),
    ]

    for exception in test_cases:
        mock_logger.reset_mock()

        response = await global_exception_handler(mock_request, exception)

        # Verify response format
        assert isinstance(response, JSONResponse)
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert response.body == b'{"detail":"Internal server error"}'

        # Verify logging
        mock_logger.exception.assert_called_once_with(
            "Unhandled exception in request %s: %s", mock_request.url, exception
        )


# =============================================================================
# STARTUP EVENT TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_startup_event_successful_initialization(
    mock_external_dependencies,
    mock_logger,
):
    """
    Test successful startup event execution. Verifies that all services are properly
    initialized during startup and that the startup completes successfully.
    """
    deps = mock_external_dependencies

    # Mock WebSocket components
    with patch("app.core.application.RedisPubSubManager") as mock_pubsub:
        with patch("app.core.application.ConnectionManager") as mock_conn_mgr:
            with patch("app.core.application.StockTickerServerFactory") as mock_factory:

                await startup_event()

                # Verify database initialization
                deps["create_db"].assert_called_once()

                # Verify Redis initialization
                deps["redis"]["connection"].get_connection.assert_called_once()

                # Verify rate limiter initialization
                deps["limiter"].init.assert_called_once_with(deps["redis"]["client"])

                # Verify Kafka consumer initialization
                deps["kafka"]["class"].assert_called_once()
                deps["fire_and_forgot"].assert_called_once()
                deps["register_task"].assert_called_once()

                # Verify WebSocket server initialization
                mock_pubsub.assert_called_once_with(deps["redis"]["client"])
                mock_conn_mgr.assert_called_once()
                mock_factory.assert_called_once()

                # Verify startup completion
                assert deps["app_state"].startup_complete is True

                # Verify logging
                mock_logger.info.assert_any_call("Tokens database initialized")
                mock_logger.info.assert_any_call("Request rate limiter initialized")
                mock_logger.info.assert_any_call("Kafka consumer started")
                mock_logger.info.assert_any_call(
                    "%s startup completed successfully", SERVICE_NAME
                )


@pytest.mark.asyncio
async def test_startup_event_without_twisted(mock_external_dependencies, mock_logger):
    """
    Test startup event when Twisted is not available. Verifies that startup continues
    successfully even when Twisted is not available for WebSocket server.
    """
    deps = mock_external_dependencies

    with patch("app.core.application.TWISTED_AVAILABLE", False):
        await startup_event()

        # Verify that basic services are still initialized
        deps["create_db"].assert_called_once()
        deps["redis"]["connection"].get_connection.assert_called_once()
        deps["limiter"].init.assert_called_once()
        deps["kafka"]["class"].assert_called_once()

        # Verify warning about Twisted
        mock_logger.warning.assert_called_with(
            "Twisted unavailable. WebSocket server not started."
        )

        # Verify startup completion
        assert deps["app_state"].startup_complete is True


@pytest.mark.asyncio
async def test_startup_event_websocket_port_conflict(
    mock_external_dependencies,
    mock_logger,
):
    """
    Test startup event with WebSocket port conflict. Verifies that startup handles WebSocket
    port conflicts gracefully.
    """
    deps = mock_external_dependencies

    # Mock environment to return port 8000 (same as FastAPI)
    with patch("app.core.application.get_env_var") as mock_get_env:

        def mock_env_side_effect(key, default=None):
            if key == "WEBSOCKET_PORT":
                return "8000"
            if key == "WEBSOCKET_HOST":
                return "localhost"
            return default

        mock_get_env.side_effect = mock_env_side_effect

        with patch("app.core.application.RedisPubSubManager"):
            with patch("app.core.application.ConnectionManager"):
                with patch("app.core.application.StockTickerServerFactory"):
                    await startup_event()

                    # Should log error about port conflict
                    mock_logger.error.assert_any_call(
                        "Failed to start WebSocket server: %s", ANY, exc_info=True
                    )
    assert not deps["app_state"].websocket_server_running
    assert deps["app_state"].websocket_server_port is None


@pytest.mark.asyncio
async def test_startup_event_critical_failures(mock_external_dependencies, mock_logger):
    """
    Test startup event with critical service failures that should cause system exit.
    Tests Redis connection, database initialization, and Kafka consumer failures.
    """
    deps = mock_external_dependencies

    # Test Redis connection failure
    deps["redis"]["connection"].get_connection.side_effect = ConnectionError(
        "Redis connection failed"
    )

    with patch("app.core.application.sys.exit") as mock_exit:
        await startup_event()
        mock_logger.critical.assert_called_with(
            "Fatal error during startup: %s",
            deps["redis"]["connection"].get_connection.side_effect,
            exc_info=True,
        )
        mock_exit.assert_called_once_with(1)

    # Reset mocks and test database failure
    mock_logger.reset_mock()
    mock_exit.reset_mock()
    deps["redis"]["connection"].get_connection.side_effect = None
    deps["create_db"].side_effect = RuntimeError("Database initialization failed")

    with patch("app.core.application.sys.exit") as mock_exit:
        await startup_event()
        mock_logger.critical.assert_called_with(
            "Fatal error during startup: %s",
            deps["create_db"].side_effect,
            exc_info=True,
        )
        mock_exit.assert_called_once_with(1)

    # Reset mocks and test Kafka failure
    mock_logger.reset_mock()
    mock_exit.reset_mock()
    deps["create_db"].side_effect = None
    deps["kafka"]["class"].side_effect = RuntimeError("Kafka consumer failed")

    with patch("app.core.application.sys.exit") as mock_exit:
        await startup_event()
        mock_logger.critical.assert_called_with(
            "Fatal error during startup: %s",
            deps["kafka"]["class"].side_effect,
            exc_info=True,
        )
        mock_exit.assert_called_once_with(1)


# =============================================================================
# SHUTDOWN EVENT TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_shutdown_event_successful_cleanup(
    mock_external_dependencies, mock_logger, mock_twisted_components
):
    """
    Test successful shutdown event execution. Verifies that all resources are properly cleaned
    up during shutdown and that the shutdown completes successfully.
    """
    deps = mock_external_dependencies
    twisted = mock_twisted_components

    # Set up app state as if startup completed successfully
    deps["app_state"].startup_complete = True
    deps["app_state"].websocket_server_port = twisted["port"]
    deps["app_state"].websocket_server_running = True

    # Mock Kafka consumer task that completes successfully
    async def successful_task():
        pass  # Completes successfully

    # Create a real task that will complete successfully when awaited
    mock_kafka_task = asyncio.create_task(successful_task())

    # Override done() to return False initially so the shutdown logic runs
    mock_kafka_task.done = MagicMock(return_value=False)

    # Mock cancel to track calls
    mock_kafka_task.cancel = MagicMock(side_effect=mock_kafka_task.cancel)
    deps["app_state"].kafka_consumer_task = mock_kafka_task

    mock_deferred = Future()
    mock_deferred.set_result(None)  # Complete the future immediately
    twisted["port"].stopListening.return_value = mock_deferred

    await shutdown_event()

    # Verify WebSocket server shutdown
    twisted["port"].stopListening.assert_called_once()
    assert deps["app_state"].websocket_server_port is None
    assert deps["app_state"].websocket_server_running is False

    # Verify Kafka consumer shutdown - in successful cleanup, task completes gracefully
    # so cancel() should not be called
    mock_kafka_task.cancel.assert_not_called()
    assert deps["app_state"].kafka_consumer_task is None

    # Verify logging
    mock_logger.info.assert_any_call(
        "Application shutting down, cleaning up resources..."
    )
    mock_logger.info.assert_any_call("Stopping WebSocket server...")
    mock_logger.info.assert_any_call("WebSocket server stopped")
    mock_logger.info.assert_any_call("Stopping Kafka consumer...")


@pytest.mark.asyncio
async def test_shutdown_event_before_startup_complete(
    mock_external_dependencies, mock_logger
):
    """
    Test shutdown event called before startup completion. Verifies that shutdown handles cases
    where it's called before startup has completed.
    """
    deps = mock_external_dependencies
    deps["app_state"].startup_complete = False

    await shutdown_event()

    # Should log that shutdown was called before startup
    mock_logger.info.assert_any_call("Shutdown called before startup completed")


@pytest.mark.asyncio
async def test_shutdown_event_timeout_scenarios(
    mock_external_dependencies, mock_logger, mock_twisted_components
):
    """
    Test shutdown event with timeout scenarios for WebSocket server and Kafka consumer.
    Verifies that timeouts are handled gracefully with appropriate warnings.
    """
    deps = mock_external_dependencies
    twisted = mock_twisted_components

    # Test WebSocket server stop timeout
    deps["app_state"].startup_complete = True
    deps["app_state"].websocket_server_port = twisted["port"]
    deps["app_state"].websocket_server_running = True

    async def timeout_deferred():
        await asyncio.sleep(10)  # Longer than timeout

    twisted["port"].stopListening.return_value = timeout_deferred()

    await shutdown_event()

    mock_logger.warning.assert_called_with(
        "Timed out waiting for WebSocket server to stop"
    )

    # Reset for Kafka timeout test
    mock_logger.reset_mock()
    deps["app_state"].websocket_server_port = None
    deps["app_state"].websocket_server_running = False

    # Test Kafka consumer timeout
    async def timeout_task():
        await asyncio.sleep(100)  # Longer than timeout

    mock_kafka_task = asyncio.create_task(timeout_task())
    mock_kafka_task.done = MagicMock(return_value=False)
    mock_kafka_task.cancel = MagicMock(side_effect=mock_kafka_task.cancel)
    deps["app_state"].kafka_consumer_task = mock_kafka_task

    await shutdown_event()

    mock_logger.warning.assert_called_with(
        "Timed out waiting for Kafka consumer to stop"
    )


@pytest.mark.asyncio
async def test_shutdown_event_error_scenarios(
    mock_external_dependencies, mock_logger, mock_twisted_components
):
    """
    Test shutdown event with various error scenarios including WebSocket server errors
    and Kafka consumer errors. Verifies graceful error handling during shutdown.
    """
    deps = mock_external_dependencies
    twisted = mock_twisted_components

    # Test WebSocket server stop error
    deps["app_state"].startup_complete = True
    deps["app_state"].websocket_server_port = twisted["port"]
    deps["app_state"].websocket_server_running = True

    twisted["port"].stopListening.side_effect = RuntimeError("Stop error")

    await shutdown_event()

    mock_logger.error.assert_called_with(
        "Error stopping WebSocket server: %s",
        twisted["port"].stopListening.side_effect,
        exc_info=True,
    )

    # Reset for Kafka error test
    mock_logger.reset_mock()
    deps["app_state"].websocket_server_port = None
    deps["app_state"].websocket_server_running = False

    # Test Kafka consumer error
    async def error_task():
        raise RuntimeError("Kafka shutdown error")

    mock_kafka_task = asyncio.create_task(error_task())
    mock_kafka_task.done = MagicMock(return_value=False)
    mock_kafka_task.cancel = MagicMock(side_effect=mock_kafka_task.cancel)
    deps["app_state"].kafka_consumer_task = mock_kafka_task

    await shutdown_event()

    mock_logger.error.assert_called_with(
        "Error during Kafka task shutdown: %s", ANY, exc_info=True
    )


@pytest.mark.asyncio
async def test_shutdown_event_no_websocket_server(
    mock_external_dependencies, mock_logger
):
    """
    Test shutdown event when no WebSocket server is running. Verifies that shutdown handles
    cases where no WebSocket server was started.
    """
    deps = mock_external_dependencies

    # Set up app state without WebSocket server
    deps["app_state"].startup_complete = True
    deps["app_state"].websocket_server_port = None
    deps["app_state"].websocket_server_running = False

    await shutdown_event()

    # Should not attempt to stop WebSocket server
    mock_logger.info.assert_any_call(
        "Application shutting down, cleaning up resources..."
    )

    for call in mock_logger.info.mock_calls:
        assert " ".join(call.args) != "Stopping WebSocket server..."


@pytest.mark.asyncio
async def test_shutdown_event_no_kafka_consumer(
    mock_external_dependencies, mock_logger
):
    """
    Test shutdown event when no Kafka consumer is running. Verifies that shutdown handles
    cases where no Kafka consumer was started.
    """
    deps = mock_external_dependencies

    # Set up app state without Kafka consumer
    deps["app_state"].startup_complete = True
    deps["app_state"].kafka_consumer_task = None

    await shutdown_event()

    # Should not attempt to stop Kafka consumer
    mock_logger.info.assert_any_call(
        "Application shutting down, cleaning up resources..."
    )

    for call in mock_logger.info.mock_calls:
        assert " ".join(call.args) != "Stopping Kafka consumer..."


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_full_startup_shutdown_cycle(
    mock_external_dependencies, mock_logger, mock_twisted_components
):
    """
    Test complete startup and shutdown cycle. Verifies that a complete startup followed by
    shutdown works correctly and all resources are properly managed.
    """
    deps = mock_external_dependencies
    twisted = mock_twisted_components

    # Mock WebSocket components
    with patch("app.core.application.RedisPubSubManager") as mock_pubsub:
        with patch("app.core.application.ConnectionManager") as mock_conn_mgr:
            with patch("app.core.application.StockTickerServerFactory") as mock_factory:

                # Perform startup
                await startup_event()

                mock_pubsub.assert_called_once_with(deps["redis"]["client"])
                mock_conn_mgr.assert_called_once()
                mock_factory.assert_called_once()

                mock_logger.info.assert_any_call(
                    "%s startup completed successfully", SERVICE_NAME
                )

                # Verify startup completed
                assert deps["app_state"].startup_complete is True
                assert deps["app_state"].websocket_server_port is not None
                assert deps["app_state"].websocket_server_running is True

                # Mock Kafka task for shutdown
                async def test_task():
                    pass  # Completes successfully

                # Create a real task for testing
                mock_kafka_task = asyncio.create_task(test_task())

                # done() should be a regular method that returns a boolean, not a coroutine
                mock_kafka_task.done = MagicMock(return_value=False)
                original_cancel = mock_kafka_task.cancel
                mock_kafka_task.cancel = MagicMock(side_effect=original_cancel)
                deps["app_state"].kafka_consumer_task = mock_kafka_task

                # Mock WebSocket server stop
                mock_deferred = Future()
                mock_deferred.set_result(None)  # Complete immediately
                twisted["port"].stopListening.return_value = mock_deferred

                # Perform shutdown
                await shutdown_event()

                mock_logger.info.assert_any_call(
                    "Application shutting down, cleaning up resources..."
                )

                # Verify shutdown completed
                assert deps["app_state"].websocket_server_port is None
                assert deps["app_state"].websocket_server_running is False
                assert deps["app_state"].kafka_consumer_task is None


@pytest.mark.asyncio
async def test_startup_with_partial_failures(mock_external_dependencies, mock_logger):
    """
    Test startup with partial service failures. Verifies that startup handles cases where
    some services fail but others succeed.
    """
    deps = mock_external_dependencies

    # Mock WebSocket factory to raise exception
    with patch("app.core.application.RedisPubSubManager"):
        with patch("app.core.application.ConnectionManager"):
            with patch("app.core.application.StockTickerServerFactory") as mock_factory:
                mock_factory.side_effect = RuntimeError("WebSocket factory error")

                await startup_event()

                # Should log error about WebSocket server failure
                mock_logger.error.assert_any_call(
                    "Failed to start WebSocket server: %s", ANY, exc_info=True
                )

                # But startup should still complete for other services
                assert deps["app_state"].startup_complete is True


@pytest.mark.asyncio
async def test_environment_variable_integration():
    """
    Test environment variable integration. Verifies that environment variables are properly
    loaded and used throughout the application startup process.
    """
    test_env_vars = {
        "CORS_ORIGINS": "http://test1.com,http://test2.com",
        "WEBSOCKET_HOST": "0.0.0.0",
        "WEBSOCKET_PORT": "8001",
        "KAFKA_BROKER_URL": "test-kafka:9092",
        "REDIS_URL": "redis://test-redis:6379",
    }

    with patch("app.core.application.get_env_var") as mock_get_env:
        mock_get_env.side_effect = lambda key, default=None: test_env_vars.get(
            key, default
        )

        # Clear any existing app instance
        FastAPIApp._app = None

        # Get new app instance with test environment
        app_instance = FastAPIApp.get_fast_api_app()

        # Verify CORS configuration uses environment variables
        cors_middleware = None
        for middleware in app_instance.user_middleware:
            if middleware.cls == CORSMiddleware:
                cors_middleware = middleware
                break

        assert cors_middleware is not None
        expected_origins = ["http://test1.com", "http://test2.com"]
        assert cors_middleware.options["allow_origins"] == expected_origins

        # Clean up
        FastAPIApp._app = None


# =============================================================================
# ERROR HANDLING AND EDGE CASES
# =============================================================================


@pytest.mark.asyncio
async def test_multiple_startup_shutdown_calls_and_edge_cases(
    mock_external_dependencies,
):
    """
    Test multiple startup/shutdown calls and various edge cases including missing environment variables.
    Verifies that multiple calls are handled gracefully and defaults work correctly.
    """
    deps = mock_external_dependencies

    # Test multiple startup calls
    await startup_event()
    await startup_event()
    await startup_event()
    assert deps["app_state"].startup_complete is True

    # Test multiple shutdown calls
    await shutdown_event()
    await shutdown_event()
    await shutdown_event()

    # Test startup with missing environment variables
    with patch("app.core.application.get_env_var") as mock_get_env:
        mock_get_env.return_value = None
        FastAPIApp._app = None
        app_instance = FastAPIApp.get_fast_api_app()
        assert isinstance(app_instance, FastAPI)
        FastAPIApp._app = None


@pytest.mark.asyncio
async def test_reactor_missing_listen_tcp(mock_external_dependencies, mock_logger):
    """
    Test startup when reactor doesn't have listenTCP method. Verifies that startup handles
    cases where Twisted reactor doesn't support the expected interface.
    """

    # Mock reactor without listenTCP
    mock_reactor = MagicMock()
    del mock_reactor.listenTCP  # Remove the method

    with patch("app.core.application.reactor", mock_reactor):
        with patch("app.core.application.TWISTED_AVAILABLE", True):
            with patch("app.core.application.RedisPubSubManager"):
                with patch("app.core.application.ConnectionManager"):
                    with patch("app.core.application.StockTickerServerFactory"):
                        await startup_event()

                        # Should log error about missing listenTCP
                        mock_logger.error.assert_any_call(
                            "Twisted reactor does not support listenTCP; WebSocket server not started."
                        )
    assert mock_external_dependencies["app_state"].websocket_server_running is False
    assert mock_external_dependencies["app_state"].websocket_server_port is None


def test_fastapi_app_multiple_cors_configurations():
    """
    Test FastAPIApp with multiple CORS configurations and singleton behavior.
    Verifies that CORS middleware is only added once and service configuration works correctly.
    """
    # Clear any existing app instance
    FastAPIApp._app = None

    with patch("app.core.application.get_env_var") as mock_get_env:
        mock_get_env.return_value = "http://localhost:3000"

        # Get app instance multiple times
        app1 = FastAPIApp.get_fast_api_app()
        app2 = FastAPIApp.get_fast_api_app()
        app3 = FastAPIApp.get_fast_api_app()

        # Should be same instance
        assert app1 is app2 is app3

        # Should have only one CORS middleware
        cors_count = sum(
            1 for middleware in app1.user_middleware if middleware.cls == CORSMiddleware
        )
        assert cors_count == 1

    # Test service name and version configuration
    with patch("app.core.application.SERVICE_NAME", "TestTradoService"):
        with patch("app.core.application.API_VERSION", "2.0.0"):
            FastAPIApp._app = None
            app_instance = FastAPIApp.get_fast_api_app()
            assert app_instance.title == "TestTradoService"
            assert app_instance.version == "2.0.0"
            assert (
                app_instance.description
                == "Market data API with real-time WebSocket updates"
            )

    # Clean up
    FastAPIApp._app = None


def test_module_imports_and_structure():
    """
    Test module imports and overall structure. Verifies that all necessary imports
    are present and the module structure is correct.
    """
    import app.core.application as app_module

    # Verify key components are imported/defined
    assert hasattr(app_module, "FastAPIApp")
    assert hasattr(app_module, "app")
    assert hasattr(app_module, "global_exception_handler")
    assert hasattr(app_module, "startup_event")
    assert hasattr(app_module, "shutdown_event")
    assert hasattr(app_module, "logger")
    assert hasattr(app_module, "LOOP")

    # Verify environment loading and feature flags
    assert hasattr(app_module, "load_dotenv")
    assert hasattr(app_module, "TWISTED_AVAILABLE")
    assert isinstance(app_module.TWISTED_AVAILABLE, bool)


# =============================================================================
# PERFORMANCE AND RESOURCE TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_memory_cleanup_and_resource_management(mock_external_dependencies):
    """
    Test memory cleanup and resource management during shutdown and startup timeout handling.
    Verifies that all references are properly cleaned up and timeouts are handled gracefully.
    """
    deps = mock_external_dependencies

    # Test memory cleanup during shutdown
    deps["app_state"].startup_complete = True
    deps["app_state"].redis_client = _create_mock_redis_client()

    async def cleanup_task():
        pass

    mock_kafka_task = asyncio.create_task(cleanup_task())
    mock_kafka_task.done = MagicMock(return_value=False)
    original_cancel = mock_kafka_task.cancel
    mock_kafka_task.cancel = MagicMock(side_effect=original_cancel)
    deps["app_state"].kafka_consumer_task = mock_kafka_task

    await shutdown_event()

    # Verify cleanup
    assert deps["app_state"].kafka_consumer_task is None

    # Test startup timeout handling
    async def slow_connection():
        await asyncio.sleep(10)  # Simulate slow connection
        return _create_mock_redis_client()

    deps["redis"]["connection"].get_connection = slow_connection

    # Use asyncio.wait_for to test timeout behavior
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(startup_event(), timeout=1.0)
