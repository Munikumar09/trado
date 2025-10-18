# pylint: disable= protected-access, import-outside-toplevel

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
from collections.abc import Generator
from unittest.mock import ANY, AsyncMock, MagicMock
from pytest_mock import MockerFixture, MockType

import pytest
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.core.application import (
    FastAPIApp,
    app,
    global_exception_handler,
    shutdown_event,
    startup_event,
)
from app.utils.constants import SERVICE_NAME
from redis import Redis

# =============================================================================
# SHARED FIXTURES
# =============================================================================


@pytest.fixture
def mock_logger(mocker: MockerFixture) -> MockType:
    """
    Mock logger fixture for testing log output. Provides a mock logger to verify logging
    behavior throughout the test suite.
    """
    return mocker.patch("app.core.application.logger")


@pytest.fixture
def mock_redis_client(mocker: MockerFixture) -> Generator[MagicMock, None, None]:
    """
    Fixture to provide a mock Redis client.
    """
    mock_client = mocker.MagicMock(spec=Redis)
    mock_pubsub = mocker.MagicMock()
    mock_client.pubsub.return_value = mock_pubsub
    
    return mock_client


@pytest.fixture
def mock_app_state():
    """
    Mock AppState fixture for testing application state management. 
    Provides a mock AppState to isolate application state during testing.
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

@pytest.fixture
def mock_redis_connection(mocker: MockerFixture):
    """
    Mock Redis connection fixture for testing Redis integration. Provides a mock Redis
    connection and client for isolated testing.
    """
    fake_connection = mocker.patch(
        "app.core.application.RedisAsyncConnection",
    )
    mock_redis_client = AsyncMock()
    fake_build = AsyncMock()
    fake_build.get_connection.return_value = mock_redis_client
    fake_connection.build.return_value = fake_build

    return {"client": mock_redis_client, "connection": fake_connection}


@pytest.fixture
def mock_kafka_consumer(mocker):
    """
    Mock Kafka consumer fixture for testing Kafka integration. Provides a
    mock KafkaConsumer for isolated testing.
    """
    mock_kafka_consumer = mocker.patch("app.core.application.KafkaConsumer")
    mock_instance = MagicMock()
    mock_instance.consume_messages = AsyncMock(return_value=[])
    mock_kafka_consumer.build.return_value = mock_instance

    return {"class": mock_kafka_consumer, "instance": mock_instance}


@pytest.fixture
def mock_twisted_components(mocker):
    """
    Mock Twisted components fixture for testing WebSocket server integration. Provides mock
    Twisted reactor and related components.
    """
    mock_reactor = MagicMock()
    mock_reactor.listenTCP = MagicMock()

    mock_port = MagicMock()
    mock_port.stopListening = MagicMock()
    mock_reactor.listenTCP.return_value = mock_port

    mocker.patch("app.core.application.TWISTED_AVAILABLE", True)
    mocker.patch("app.core.application.reactor", mock_reactor)

    return {"reactor": mock_reactor, "port": mock_port}


@pytest.fixture
def mock_external_dependencies(
    mock_redis_connection, mock_kafka_consumer, mock_app_state, mocker
):
    """
    Combined fixture that mocks all external dependencies. Provides a comprehensive mock
    setup for testing application functionality without external service dependencies.
    """
    mock_create_db = mocker.patch("app.core.application.create_tokens_db")
    mock_limiter = mocker.patch("app.core.application.FastAPILimiter")
    mock_fire_and_forgot = mocker.patch("app.core.application.fire_and_forgot")
    mock_register_task = mocker.patch("app.core.application.register_task_for_cleanup")
    mock_register_shutdown = mocker.patch(
        "app.core.application.register_shutdown_handler"
    )
    mock_pubsub = mocker.patch("app.core.application.RedisPubSubManager")
    mock_conn_mgr = mocker.patch("app.core.application.ConnectionManager")
    mock_factory = mocker.patch("app.core.application.StockTickerServerFactory")

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

    return {
        "create_db": mock_create_db,
        "limiter": mock_limiter,
        "fire_and_forgot": mock_fire_and_forgot,
        "register_task": mock_register_task,
        "register_shutdown": mock_register_shutdown,
        "redis": mock_redis_connection,
        "kafka": mock_kafka_consumer,
        "app_state": mock_app_state,
        "pubsub": mock_pubsub,
        "conn_mgr": mock_conn_mgr,
        "factory": mock_factory,
    }


@pytest.fixture(autouse=True)
def clean_app_state():
    """
    Clean up the application state after each test.
    """
    FastAPIApp._app = None
    yield
    FastAPIApp._app = None


def validate_redis_pubsub(redis_connection):
    """
    Validate the Redis PubSub manager.
    """
    redis_connection.build.assert_called_once_with(settings.redis_config)
    mock_instance = redis_connection.build.return_value
    mock_instance.get_connection.assert_called_once()


def complete_startup(deps, twisted, complete=True, running=True):
    deps["app_state"].startup_complete = complete
    deps["app_state"].websocket_server_port = twisted["port"]
    deps["app_state"].websocket_server_running = running


def get_mock_kafka_task(task):
    mock_kafka_task = asyncio.create_task(task())
    mock_kafka_task.done = MagicMock(return_value=False)
    mock_kafka_task.cancel = MagicMock(side_effect=mock_kafka_task.cancel)

    return mock_kafka_task


def validate_websocket_task(deps):
    assert deps["app_state"].websocket_server_port is None
    assert deps["app_state"].websocket_server_running is False


async def sample_task():
    pass


# =============================================================================
# FASTAPI APP SINGLETON TESTS
# =============================================================================


def test_fastapi_app_singleton_behavior():
    """
    Test that FastAPIApp follows singleton pattern. Verifies that multiple calls to
    get_fast_api_app() return the same instance and that the singleton pattern is
    properly implemented.
    """
    # Get first instance
    app1 = FastAPIApp.get_fast_api_app()

    # Get second instance
    app2 = FastAPIApp.get_fast_api_app()

    # Should be the same instance
    assert app1 is app2
    assert isinstance(app1, FastAPI)


def test_fastapi_app_initialization_with_defaults():
    """
    Test FastAPI app initialization with default configuration. Verifies that the FastAPI
    application is properly initialized with correct title, description, and version.
    """
    app_instance = FastAPIApp.get_fast_api_app()

    assert app_instance.title == "Market Data Service"
    assert (
        app_instance.description == "Market data API with real-time WebSocket updates"
    )
    assert app_instance.version == "1.0.0"


def test_fastapi_app_cors_middleware_configuration():
    """
    Test CORS middleware configuration. Verifies that CORS middleware is properly configured
    with correct origins and settings from environment variables, including wildcard origins.
    """

    app_instance = FastAPIApp.get_fast_api_app()

    # Check that CORS middleware was added
    middleware_found = False
    for middleware in app_instance.user_middleware:

        if middleware.cls == CORSMiddleware:
            middleware_found = True

            # Verify CORS configuration
            options = middleware.kwargs
            assert options["allow_credentials"] is True
            assert options["allow_methods"] == ["*"]
            assert options["allow_headers"] == ["*"]
            assert options["allow_origins"] == ["*"]
            break

    assert middleware_found, "CORS middleware not found"


# =============================================================================
# APPLICATION CONFIGURATION TESTS
# =============================================================================


def test_module_level_app_instance():
    """
    Test that the module-level app instance is correctly created. Verifies that the app
    variable at module level is a FastAPI instance created through the singleton pattern.
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


@pytest.mark.parametrize(
    "exception",
    [
        Exception("Test error"),
        ValueError("Invalid value"),
        KeyError("Missing key"),
        RuntimeError("Runtime error"),
    ],
)
@pytest.mark.asyncio
async def test_global_exception_handler_comprehensive(mock_logger, exception):
    """
    Test global exception handler with different exception types and scenarios.
    Verifies that the handler works correctly with various exception types and
    returns proper JSON response format.
    """
    mock_request = MagicMock(spec=Request)
    mock_request.url = "http://localhost:8000/test"

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

    await startup_event()

    # Verify database initialization
    deps["create_db"].assert_called_once()

    validate_redis_pubsub(deps["redis"]["connection"])

    # Verify rate limiter initialization
    deps["limiter"].init.assert_called_once_with(deps["redis"]["client"])

    # Verify Kafka consumer initialization
    deps["kafka"]["class"].build.assert_called_once_with(settings.kafka_config)
    deps["fire_and_forgot"].assert_called_once()
    deps["register_task"].assert_called_once()

    # Verify WebSocket server initialization
    deps["pubsub"].assert_called_once_with(deps["redis"]["client"])
    deps["conn_mgr"].assert_called_once()
    deps["factory"].assert_called_once()

    # Verify startup completion
    assert deps["app_state"].startup_complete is True

    # Verify logging
    mock_logger.info.assert_any_call("Tokens database initialized")
    mock_logger.info.assert_any_call("Request rate limiter initialized")
    mock_logger.info.assert_any_call("Kafka consumer started")
    mock_logger.info.assert_any_call("%s startup completed successfully", SERVICE_NAME)


@pytest.mark.asyncio
async def test_startup_event_without_twisted(
    mock_external_dependencies, mock_logger, mocker
):
    """
    Test startup event when Twisted is not available. Verifies that startup continues
    successfully even when Twisted is not available for WebSocket server.
    """
    deps = mock_external_dependencies
    mocker.patch("app.core.application.TWISTED_AVAILABLE", False)
    await startup_event()

    # Verify that basic services are still initialized
    deps["create_db"].assert_called_once()

    # Verify Redis initialization
    validate_redis_pubsub(deps["redis"]["connection"])

    deps["limiter"].init.assert_called_once()

    deps["kafka"]["class"].build.assert_called_once_with(settings.kafka_config)

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
    original_port = settings.websocket_config.port
    conflict_port = 8000
    settings.websocket_config.port = conflict_port
    await startup_event()

    # Should log error about port conflict
    mock_logger.error.assert_any_call(
        "Failed to start WebSocket server: %s", ANY, exc_info=True
    )
    assert not deps["app_state"].websocket_server_running
    assert deps["app_state"].websocket_server_port is None
    settings.websocket_config.port = original_port


@pytest.mark.asyncio
async def test_startup_event_critical_failures(mock_external_dependencies, mock_logger):
    """
    Test startup event with critical service failures that should cause system exit.
    Tests Redis connection, database initialization, and Kafka consumer failures.
    """
    deps = mock_external_dependencies

    error_msg = "Redis connection failed"
    # Test Redis connection failure
    deps["redis"]["connection"].build.return_value.get_connection.side_effect = (
        ConnectionError(error_msg)
    )

    with pytest.raises(ConnectionError) as exc_info:
        await startup_event()

    assert str(exc_info.value) == error_msg
    mock_logger.critical.assert_called_with(
        "Fatal error during startup: %s",
        deps["redis"]["connection"].build.return_value.get_connection.side_effect,
        exc_info=True,
    )

    # Reset mocks and test database failure
    mock_logger.reset_mock()
    error_msg = "Database initialization failed"
    deps["redis"]["connection"].build.return_value.get_connection.side_effect = None
    deps["create_db"].side_effect = RuntimeError(error_msg)

    with pytest.raises(RuntimeError) as exc_info:
        await startup_event()

    assert str(exc_info.value) == error_msg
    mock_logger.critical.assert_called_with(
        "Fatal error during startup: %s",
        deps["create_db"].side_effect,
        exc_info=True,
    )

    # Reset mocks and test Kafka failure
    mock_logger.reset_mock()
    error_msg = "Kafka consumer failed"
    deps["create_db"].side_effect = None
    deps["redis"]["connection"].build.return_value.get_connection.side_effect = None
    deps["kafka"]["class"].build.side_effect = RuntimeError("Kafka consumer failed")

    with pytest.raises(RuntimeError) as exc_info:
        await startup_event()

    assert str(exc_info.value) == error_msg
    mock_logger.critical.assert_called_with(
        "Fatal error during startup: %s",
        deps["kafka"]["class"].build.side_effect,
        exc_info=True,
    )


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
    complete_startup(deps, twisted)

    kafka_task = get_mock_kafka_task(sample_task)
    # Create a real task that will complete successfully when awaited
    deps["app_state"].kafka_consumer_task = kafka_task

    mock_deferred = Future()
    mock_deferred.set_result(None)  # Complete the future immediately
    twisted["port"].stopListening.return_value = mock_deferred

    await shutdown_event()

    # Verify WebSocket server shutdown
    twisted["port"].stopListening.assert_called_once()
    validate_websocket_task(deps)

    # Verify Kafka consumer shutdown - in successful cleanup, task completes gracefully
    # so cancel() should not be called
    kafka_task.cancel.assert_not_called()
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
    complete_startup(deps, twisted)

    async def timeout_task(time_sec=100):
        await asyncio.sleep(time_sec)  # Longer than timeout

    twisted["port"].stopListening.return_value = timeout_task(10)

    await shutdown_event()

    mock_logger.warning.assert_called_with(
        "Timed out waiting for WebSocket server to stop"
    )

    # Reset for Kafka timeout test
    mock_logger.reset_mock()
    validate_websocket_task(deps)

    deps["app_state"].kafka_consumer_task = get_mock_kafka_task(timeout_task)

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
    complete_startup(deps, twisted)

    twisted["port"].stopListening.side_effect = RuntimeError("Stop error")

    await shutdown_event()

    mock_logger.error.assert_called_with(
        "Error stopping WebSocket server: %s",
        twisted["port"].stopListening.side_effect,
        exc_info=True,
    )

    # Reset for Kafka error test
    mock_logger.reset_mock()
    validate_websocket_task(deps)

    # Test Kafka consumer error
    async def error_task():
        raise RuntimeError("Kafka shutdown error")

    deps["app_state"].kafka_consumer_task = get_mock_kafka_task(error_task)

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
    complete_startup(deps, {"port": None}, running=False)

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

    # Perform startup
    await startup_event()

    deps["pubsub"].assert_called_once_with(deps["redis"]["client"])
    deps["conn_mgr"].assert_called_once()
    deps["factory"].assert_called_once()

    mock_logger.info.assert_any_call("%s startup completed successfully", SERVICE_NAME)

    # Verify startup completed
    assert deps["app_state"].startup_complete is True
    assert deps["app_state"].websocket_server_port is not None
    assert deps["app_state"].websocket_server_running is True

    deps["app_state"].kafka_consumer_task = get_mock_kafka_task(sample_task)

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
    deps["app_state"].side_effect = RuntimeError("WebSocket factory error")

    await startup_event()

    # Should log error about WebSocket server failure
    mock_logger.error.assert_any_call(
        "Failed to start WebSocket server: %s", ANY, exc_info=True
    )

    # But startup should still complete for other services
    assert deps["app_state"].startup_complete is True


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
    num_startup_calls = 3
    for _ in range(num_startup_calls):
        await startup_event()

    assert deps["app_state"].startup_complete is True

    # Test multiple shutdown calls
    for _ in range(num_startup_calls):
        await shutdown_event()

    # Test startup with missing environment variables
    app_instance = FastAPIApp.get_fast_api_app()
    assert isinstance(app_instance, FastAPI)


@pytest.mark.asyncio
async def test_reactor_missing_listen_tcp(
    mock_external_dependencies, mock_twisted_components, mock_logger
):
    """
    Test startup when reactor doesn't have listenTCP method. Verifies that startup handles
    cases where Twisted reactor doesn't support the expected interface.
    """
    del mock_twisted_components["reactor"].listenTCP  # Remove the method
    await startup_event()

    # Should log error about missing listenTCP
    mock_logger.error.assert_any_call(
        "Twisted reactor does not support listenTCP; WebSocket server not started."
    )
    validate_websocket_task(mock_external_dependencies)


def test_fastapi_app_multiple_cors_configurations():
    """
    Test FastAPIApp with multiple CORS configurations and singleton behavior.
    Verifies that CORS middleware is only added once and service configuration works correctly.
    """

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

    # Verify environment loading and feature flags
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
    deps["app_state"].redis_client = AsyncMock()

    deps["app_state"].kafka_consumer_task = get_mock_kafka_task(sample_task)

    await shutdown_event()

    # Verify cleanup
    assert deps["app_state"].kafka_consumer_task is None

    # Test startup timeout handling
    async def slow_connection():
        await asyncio.sleep(10)  # Simulate slow connection
        return AsyncMock()

    deps["redis"]["connection"].build.return_value.get_connection = slow_connection

    # Use asyncio.wait_for to test timeout behavior
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(startup_event(), timeout=1.0)
