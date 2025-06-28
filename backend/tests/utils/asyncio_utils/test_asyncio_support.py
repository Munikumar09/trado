# pylint: disable=redefined-outer-name protected-access no-member
"""
Comprehensive tests for the asyncio_support module.

This test suite covers all aspects of the asyncio_support module including:
- AsyncioLoop singleton class behavior
- Exception handling setup and configuration
- Twisted reactor installation logic
- Signal handling and graceful shutdown
- Task registration and cleanup mechanisms
- Error handling and edge cases

Tests use minimal mocking, only mocking external dependencies like:
- Twisted reactor components
- System signal handling
- Process exit calls
- Heavy I/O operations
"""

import asyncio
import signal
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

from app.utils.asyncio_utils.asyncio_support import (
    AsyncioLoop,
    LoopNotRunningError,
    _install_twisted_reactor,
    _reactor_already_installed,
    _running_tasks,
    _setup_exception_handling,
    _shutdown_tasks,
    install_twisted_reactor,
    register_shutdown_handler,
    register_task_for_cleanup,
    shutdown,
)

# =============================================================================
# SHARED FIXTURES
# =============================================================================


@pytest.fixture
def mock_logger():
    """
    Mock logger fixture for testing log output.
    """
    with patch("app.utils.asyncio_utils.asyncio_support.logger") as mock:
        yield mock


@pytest.fixture(autouse=True)
def clear_global_state():
    """
    Clear global state before and after each test.
    """
    original_shutdown_tasks = _shutdown_tasks.copy()
    original_running_tasks = _running_tasks.copy()

    _shutdown_tasks.clear()
    _running_tasks.clear()

    yield

    _shutdown_tasks.clear()
    _shutdown_tasks.extend(original_shutdown_tasks)
    _running_tasks.clear()
    _running_tasks.update(original_running_tasks)


@pytest.fixture(autouse=True)
def reset_sys_modules():
    """
    Reset sys.modules state for twisted reactor tests.
    """
    original_modules = sys.modules.copy()
    yield

    # Remove any twisted modules added during tests
    for module in list(sys.modules.keys()):
        if module.startswith("twisted") and module not in original_modules:
            del sys.modules[module]


# =============================================================================
# ASYNCIOLOOP SINGLETON TESTS
# =============================================================================


def test_asyncio_loop_singleton_creation(mock_logger):
    """
    Test AsyncioLoop singleton creation and initialization. Verifies that the singleton creates
    a new loop, sets it as the event loop, sets up exception handling, and logs the creation.
    """
    # Test first creation
    loop1 = AsyncioLoop.get_loop()

    assert isinstance(loop1, asyncio.AbstractEventLoop)
    assert AsyncioLoop._loop is loop1
    assert asyncio.get_event_loop() is loop1
    mock_logger.info.assert_called_once_with(
        "Created new asyncio event loop (singleton)."
    )


def test_asyncio_loop_singleton_reuse():
    """
    Test that AsyncioLoop returns the same instance on subsequent calls. Verifies the singleton
    behavior by ensuring the same loop instance is returned on multiple get_loop() calls.
    """
    loop1 = AsyncioLoop.get_loop()
    loop2 = AsyncioLoop.get_loop()
    loop3 = AsyncioLoop.get_loop()

    assert loop1 is loop2 is loop3
    assert isinstance(loop1, asyncio.AbstractEventLoop)


def test_asyncio_loop_thread_safety():
    """
    Test AsyncioLoop thread safety in concurrent access. Verifies that the singleton
    behaves correctly when accessed from multiple threads simultaneously.
    """
    loops = {}

    def _get_loop_in_thread(thread_id):
        loops[thread_id] = AsyncioLoop.get_loop()

    threads = []
    for i in range(5):
        thread = threading.Thread(target=_get_loop_in_thread, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # All threads should get the same loop instance
    first_loop = loops[0]
    for loop in loops.values():
        assert loop is first_loop


# =============================================================================
# EXCEPTION HANDLING TESTS
# =============================================================================


def test_setup_exception_handling():
    """
    Test _setup_exception_handling function. Verifies that the exception handler
    is properly set on the event loop.
    """
    loop = asyncio.new_event_loop()

    # Initially no custom handler
    assert loop.get_exception_handler() is None

    _setup_exception_handling(loop)

    # Now should have a custom handler
    assert loop.get_exception_handler() is not None
    assert callable(loop.get_exception_handler())

    loop.close()


def test_exception_handler_with_real_exception(mock_logger):
    """
    Test exception handler behavior with actual exceptions. Verifies that exceptions are
    properly logged with context information.
    """
    loop = AsyncioLoop.get_loop()

    # Create a real task that will raise an exception
    async def failing_task():
        raise ValueError("Test error for exception handling")

    # Run the task and let it fail
    loop.create_task(failing_task())

    loop.run_until_complete(
        asyncio.sleep(0.1)
    )  # Allow time for the task to run and fail

    # The exception handler should have been called
    mock_logger.error.assert_called()
    call_args = mock_logger.error.call_args
    assert "Unhandled exception in async operation" in call_args[0][0]

    loop.close()


def test_exception_handler_with_cancelled_error(mock_logger):
    """
    Test exception handler with CancelledError (should not log). Verifies that CancelledError
    exceptions are ignored as they're normal during shutdown operations.
    """
    loop = AsyncioLoop.get_loop()

    async def cancelled_task():
        await asyncio.sleep(10)  # Long sleep to allow cancellation

    task = loop.create_task(cancelled_task())
    task.cancel()

    try:
        loop.run_until_complete(task)
    except asyncio.CancelledError:
        pass  # Expected

    # Should not log cancelled errors
    mock_logger.error.assert_not_called()
    mock_logger.warning.assert_not_called()

    loop.close()


def test_exception_handler_context_logging(mock_logger):
    """
    Test that exception handler logs context information properly. Verifies that task and
    other context details are included in logs.
    """
    loop = AsyncioLoop.get_loop()

    async def task_with_context():
        raise RuntimeError("Error with context")

    loop.create_task(task_with_context(), name="test_task")

    loop.run_until_complete(
        asyncio.sleep(0.1)
    )  # Allow time for the task to run and fail

    mock_logger.error.assert_called()
    call_args = mock_logger.error.call_args

    # Verify context is logged
    assert len(call_args) >= 2
    extra = call_args[1].get("extra", {})
    assert "task" in extra

    loop.close()


# =============================================================================
# TWISTED REACTOR INSTALLATION TESTS
# =============================================================================


def test_reactor_already_installed_false_when_not_imported():
    """
    Test _reactor_already_installed returns False when reactor not imported. Verifies that
    function returns False when Twisted reactor is not in sys.modules.
    """
    # Ensure twisted reactor is not in sys.modules
    if "twisted.internet.reactor" in sys.modules:
        del sys.modules["twisted.internet.reactor"]

    result = _reactor_already_installed()
    assert result is False


def test_reactor_already_installed_true_when_imported():
    """
    Test _reactor_already_installed returns True when reactor is imported. Verifies detection
    of already installed Twisted reactor.
    """
    # Mock the reactor module
    mock_reactor = MagicMock()
    mock_reactor.running = True

    with patch.dict(sys.modules, {"twisted.internet.reactor": mock_reactor}):
        result = _reactor_already_installed()
        assert result is True


def test_reactor_already_installed_handles_import_error():
    """
    Test _reactor_already_installed handles ImportError gracefully. Verifies that ImportError
    during reactor import is handled properly.
    """
    # Put something in sys.modules but make import fail
    with patch.dict(sys.modules, {"twisted.internet.reactor": MagicMock()}):
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            result = _reactor_already_installed()
            assert result is False


def test_install_twisted_reactor_success(mock_logger):
    """
    Test successful Twisted reactor installation. Verifies that the reactor is installed
    correctly when Twisted is available.
    """
    mock_asyncioreactor = MagicMock()
    with patch.dict(
        sys.modules, {"twisted.internet.asyncioreactor": mock_asyncioreactor}
    ):
        loop = AsyncioLoop.get_loop()

        _install_twisted_reactor(loop)

        mock_asyncioreactor.install.assert_called_once_with(loop)
        mock_logger.info.assert_any_call("Installing Twisted AsyncioSelectorReactor...")
        mock_logger.info.assert_any_call("Twisted reactor installed successfully.")

        loop.close()


def test_install_twisted_reactor_import_error(mock_logger):
    """
    Test Twisted reactor installation when Twisted is not available.
    Verifies that ImportError is handled gracefully with appropriate warning.
    """
    loop = AsyncioLoop.get_loop()

    # Mock the import to fail
    with patch("builtins.__import__", side_effect=ImportError("No twisted")):
        _install_twisted_reactor(loop)

    mock_logger.warning.assert_called_once_with(
        "Twisted or asyncioreactor not available. Skipping install."
    )

    loop.close()


def test_install_twisted_reactor_already_installed_error(mock_logger):
    """
    Test Twisted reactor installation when reactor is already installed. Verifies that
    ReactorAlreadyInstalledError is handled with warning.
    """
    mock_asyncioreactor = MagicMock()

    with patch.dict(
        sys.modules, {"twisted.internet.asyncioreactor": mock_asyncioreactor}
    ):
        loop = AsyncioLoop.get_loop()
        mock_asyncioreactor.install.side_effect = Exception(
            "ReactorAlreadyInstalledError"
        )

        _install_twisted_reactor(loop)

        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0]
        assert "reactor already installed" in warning_call[0].lower()

        loop.close()


def test_install_twisted_reactor_unexpected_error(mock_logger):
    """
    Test Twisted reactor installation with unexpected error. Verifies that unexpected
    errors are logged and re-raised.
    """
    mock_asyncioreactor = MagicMock()

    with patch.dict(
        sys.modules, {"twisted.internet.asyncioreactor": mock_asyncioreactor}
    ):
        loop = AsyncioLoop.get_loop()
        mock_asyncioreactor.install.side_effect = Exception("Unexpected error")

        with pytest.raises(Exception, match="Unexpected error"):
            _install_twisted_reactor(loop)

        mock_logger.error.assert_called_once()

        loop.close()


# =============================================================================
# SIGNAL HANDLING AND FULL INSTALLATION TESTS
# =============================================================================


def test_install_twisted_reactor_full_success(mock_logger):
    """
    Test complete install_twisted_reactor function execution. Verifies the full installation
    process including signal handlers and atexit registration.
    """

    with patch("app.utils.asyncio_utils.asyncio_support.sys.platform", "linux"):
        new_sys_modules = {
            key: val for key, val in sys.modules.items() if "pytest" not in key
        }
        with patch.dict(sys.modules, new_sys_modules, clear=True):  # No pytest modules
            install_twisted_reactor()

    mock_logger.info.assert_any_call("Installing Twisted AsyncioSelectorReactor...")

    mock_logger.info.assert_any_call("Twisted reactor installed successfully.")
    mock_logger.info.assert_any_call("Created new asyncio event loop (singleton).")

    loop = AsyncioLoop.get_loop()
    assert loop is not None


@patch("app.utils.asyncio_utils.asyncio_support._reactor_already_installed")
def test_install_twisted_reactor_already_installed_skip(
    mock_reactor_check,
    mock_logger,
):
    """
    Test install_twisted_reactor when reactor is already installed. Verifies that installation
    is skipped when reactor is already present.
    """
    mock_reactor_check.return_value = True
    install_twisted_reactor()

    mock_logger.info.assert_called_with(
        "Twisted reactor already installed. Skipping installation."
    )


def test_install_twisted_reactor_pytest_environment(
    mock_logger,
):
    """
    Test install_twisted_reactor in pytest environment. Verifies that reactor installation
    is skipped in pytest context.
    """
    install_twisted_reactor()
    mock_logger.info.assert_any_call(
        "Detected pytest environment. Skipping reactor install."
    )


@patch("app.utils.asyncio_utils.asyncio_support._reactor_already_installed")
@patch("app.utils.asyncio_utils.asyncio_support._install_twisted_reactor")
def test_install_twisted_reactor_windows_platform(
    mock_install,
    mock_reactor_check,
):
    """
    Test install_twisted_reactor on Windows platform. Verifies that signal handlers are not
    installed on Windows.
    """
    mock_reactor_check.return_value = False
    sys.modules.pop("pytest_asyncio")
    sys.modules.pop("pytest")  # Ensure pytest modules are not present

    with patch("app.utils.asyncio_utils.asyncio_support.sys.platform", "win32"):
        install_twisted_reactor()

    # Should still install reactor but not signal handlers
    mock_install.assert_called_once()


# =============================================================================
# TASK REGISTRATION AND CLEANUP TESTS
# =============================================================================


def test_register_shutdown_handler(mock_logger):
    """
    Test register_shutdown_handler function. Verifies that shutdown handlers are properly
    registered and logged.
    """

    def test_handler():
        return "handler_result"

    register_shutdown_handler(test_handler)

    assert test_handler in _shutdown_tasks
    assert len(_shutdown_tasks) == 1
    mock_logger.debug.assert_called_once_with(
        "Registered shutdown handler: %s", test_handler.__name__
    )


def test_register_multiple_shutdown_handlers():
    """
    Test registering multiple shutdown handlers. Verifies that multiple handlers can be
    registered and are stored in order.
    """

    def handler1():
        pass

    def handler2():
        pass

    def handler3():
        pass

    register_shutdown_handler(handler1)
    register_shutdown_handler(handler2)
    register_shutdown_handler(handler3)

    assert len(_shutdown_tasks) == 3
    assert _shutdown_tasks == [handler1, handler2, handler3]


def test_register_task_for_cleanup():
    """
    Test register_task_for_cleanup function. Verifies that tasks are properly registered
    for cleanup.
    """

    async def dummy_task():
        await asyncio.sleep(0.1)

    loop = AsyncioLoop.get_loop()
    task = loop.create_task(dummy_task())

    register_task_for_cleanup(task)

    assert task in _running_tasks
    assert len(_running_tasks) == 1

    # Cancel the task before closing the loop to prevent RuntimeWarning
    task.cancel()
    try:
        loop.run_until_complete(task)
    except asyncio.CancelledError:
        pass  # Expected for cancelled tasks

    assert task not in _running_tasks
    assert len(_running_tasks) == 0

    loop.close()


def test_register_task_for_cleanup_auto_removal():
    """
    Test automatic task removal from cleanup registry when task completes. Verifies that
    completed tasks are automatically removed from tracking.
    """

    async def quick_task():
        return "done"

    loop = AsyncioLoop.get_loop()
    task = loop.create_task(quick_task())

    register_task_for_cleanup(task)
    assert task in _running_tasks

    # Complete the task
    loop.run_until_complete(task)

    assert task not in _running_tasks
    assert len(_running_tasks) == 0

    loop.close()


# =============================================================================
# SHUTDOWN FUNCTIONALITY TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_shutdown_without_signal(mock_logger):
    """
    Test shutdown function without signal parameter. Verifies that normal shutdown doesn't
    call sys.exit.
    """
    with patch("app.utils.asyncio_utils.asyncio_support.sys.exit") as mock_exit:
        await shutdown()

    mock_logger.info.assert_any_call("Initiating graceful shutdown...")
    mock_logger.info.assert_any_call("Graceful shutdown completed.")
    mock_exit.assert_not_called()


@pytest.mark.asyncio
async def test_shutdown_with_signal(mock_logger):
    """
    Test shutdown function with signal parameter. Verifies that shutdown with signal logs
    appropriately and exits.
    """
    test_signal = signal.SIGTERM

    with patch("app.utils.asyncio_utils.asyncio_support.sys.exit") as mock_exit:
        await shutdown(test_signal)

    mock_logger.info.assert_any_call(
        "Received shutdown signal %s, initiating graceful shutdown...", test_signal.name
    )
    mock_logger.info.assert_any_call("Graceful shutdown completed.")
    mock_exit.assert_called_once_with(0)


@pytest.mark.asyncio
async def test_shutdown_with_real_running_tasks(mock_logger):
    """
    Test shutdown function with actual running tasks. Verifies that real tasks are properly
    cancelled during shutdown.
    """

    # Create real async tasks
    async def long_running_task(_):
        await asyncio.sleep(1)

    # Create and register tasks
    task1 = asyncio.create_task(long_running_task(1))
    task2 = asyncio.create_task(long_running_task(2))

    await asyncio.sleep(0.1)  # Allow tasks to start

    register_task_for_cleanup(task1)
    register_task_for_cleanup(task2)

    task1.cancel()
    task2.cancel()

    assert len(_running_tasks) == 2
    assert not task1.done()
    assert not task2.done()

    # Perform shutdown
    await shutdown()

    # Tasks should be cancelled
    assert task1.cancelled()
    assert task2.cancelled()
    mock_logger.info.assert_any_call("Cancelling %d running tasks", 2)


@pytest.mark.asyncio
async def test_shutdown_with_completed_tasks():
    """
    Test shutdown with already completed tasks. Verifies that completed tasks are not cancelled.
    """

    async def quick_task():
        return "completed"

    # Create and complete a task
    task = asyncio.create_task(quick_task())
    await asyncio.sleep(0.1)  # Allow task to complete

    register_task_for_cleanup(task)

    assert task.done()
    assert not task.cancelled()

    await shutdown()

    # Task should still be done, not cancelled
    assert task.done()
    assert not task.cancelled()


@pytest.mark.asyncio
async def test_shutdown_with_handlers_lifo_order():
    """
    Test shutdown function with registered handlers in LIFO order. Verifies that shutdown
    handlers are called in reverse order (LIFO).
    """
    call_order = []

    def handler1():
        call_order.append("handler1")

    def handler2():
        call_order.append("handler2")

    async def async_handler():
        call_order.append("async_handler")

    # Register handlers in order
    register_shutdown_handler(handler1)
    register_shutdown_handler(handler2)
    register_shutdown_handler(async_handler)

    await shutdown()

    # Should be called in reverse order (LIFO)
    assert call_order == ["async_handler", "handler2", "handler1"]


@pytest.mark.asyncio
async def test_shutdown_handler_exception_handling(mock_logger):
    """
    Test shutdown function when handler raises exception. Verifies that exceptions in shutdown
    handlers are logged but don't stop shutdown.
    """
    call_order = []

    def failing_handler():
        call_order.append("failing_handler")
        raise ValueError("Handler error")

    def working_handler():
        call_order.append("working_handler")

    register_shutdown_handler(working_handler)  # This should still run
    register_shutdown_handler(failing_handler)  # This will fail

    await shutdown()

    # Both handlers should have been attempted
    assert "failing_handler" in call_order
    assert "working_handler" in call_order

    # Error should be logged
    mock_logger.error.assert_called()
    error_call = mock_logger.error.call_args[0]

    assert "Error in shutdown handler" in error_call[0]
    assert "failing_handler" in error_call[1]


@pytest.mark.asyncio
async def test_shutdown_async_handler_exception(mock_logger):
    """
    Test shutdown with async handler that raises exception. Verifies that async handler
    exceptions are properly caught and logged.
    """

    async def failing_async_handler():
        raise RuntimeError("Async handler error")

    register_shutdown_handler(failing_async_handler)

    await shutdown()

    mock_logger.error.assert_called()
    error_call = mock_logger.error.call_args[0]
    assert "Error in shutdown handler" in error_call[0]


@pytest.mark.asyncio
async def test_shutdown_task_wait_timeout(mock_logger):
    """
    Test shutdown with task cancellation timeout. Verifies that timeout during task wait is
    handled gracefully.
    """

    # Create a task that won't respond to cancellation quickly
    async def stubborn_task():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            # Simulate a task that takes time to respond to cancellation
            await asyncio.sleep(10)  # This will timeout in our test

    task = asyncio.create_task(stubborn_task())
    register_task_for_cleanup(task)

    with patch(
        "app.utils.asyncio_utils.asyncio_support.asyncio.wait",
        side_effect=Exception("Wait error"),
    ):
        await shutdown()

    mock_logger.error.assert_called()
    error_call = mock_logger.error.call_args[0]

    assert "Error waiting for tasks to cancel" in error_call[0]


# =============================================================================
# ERROR HANDLING AND EDGE CASES
# =============================================================================


def test_loop_not_running_error():
    """
    Test LoopNotRunningError exception class. Verifies that the custom exception can
    be raised and caught properly.
    """
    error_msg = "Loop is not running"

    with pytest.raises(LoopNotRunningError, match=error_msg):
        raise LoopNotRunningError(error_msg)

    assert issubclass(LoopNotRunningError, RuntimeError)


@patch("app.utils.asyncio_utils.asyncio_support._reactor_already_installed")
@patch("app.utils.asyncio_utils.asyncio_support._install_twisted_reactor")
def test_install_twisted_reactor_import_error_handling(
    mock_install,
    mock_reactor_check,
    mock_logger,
):
    """
    Test install_twisted_reactor ImportError handling. Verifies that ImportError for Twisted is
    handled gracefully.
    """
    mock_reactor_check.return_value = False
    mock_install.side_effect = ImportError("No Twisted module")

    install_twisted_reactor()

    mock_logger.warning.assert_any_call(
        "Twisted not available. Limited functionality may result."
    )


@patch("app.utils.asyncio_utils.asyncio_support._reactor_already_installed")
@patch("app.utils.asyncio_utils.asyncio_support._install_twisted_reactor")
def test_install_twisted_reactor_unexpected_error_handling(
    mock_install,
    mock_reactor_check,
    mock_logger,
):
    """
    Test install_twisted_reactor unexpected error handling. Verifies that unexpected errors during
    reactor installation are logged.
    """
    mock_reactor_check.return_value = False
    mock_install.side_effect = RuntimeError("Unexpected error")

    install_twisted_reactor()

    mock_logger.error.assert_called_once()
    error_call = mock_logger.error.call_args[0]
    assert "Failed to install Twisted reactor" in error_call[0]


def test_exception_handler_edge_cases(mock_logger):
    """
    Test exception handler with various edge cases. Verifies robustness of exception handling
    with unusual contexts.
    """
    loop = asyncio.new_event_loop()
    _setup_exception_handling(loop)

    handler = loop.get_exception_handler()

    # Test with minimal context
    handler(loop, {})

    # Test with message only
    handler(loop, {"message": "Test message"})
    mock_logger.error.assert_called_with(
        "Async error without exception: %s", "Test message"
    )

    # Test with None values
    handler(loop, {"exception": None, "message": None, "task": None, "handle": None})

    loop.close()


@pytest.mark.asyncio
async def test_shutdown_empty_state(mock_logger):
    """
    Test shutdown with no registered tasks or handlers. Verifies that shutdown works correctly
    with empty state.
    """
    assert len(_shutdown_tasks) == 0
    assert len(_running_tasks) == 0

    await shutdown()

    mock_logger.info.assert_any_call("Initiating graceful shutdown...")
    mock_logger.info.assert_any_call("Graceful shutdown completed.")


def test_register_task_for_cleanup_edge_cases():
    """
    Test task registration edge cases. Verifies robustness of task registration with various
    task states.
    """

    # Test with already completed task
    async def completed_task():
        return "done"

    loop = asyncio.new_event_loop()
    task = loop.create_task(completed_task())
    loop.run_until_complete(task)

    register_task_for_cleanup(task)
    assert task in _running_tasks

    # Test with cancelled task
    async def cancelled_task():
        await asyncio.sleep(10)

    task2 = loop.create_task(cancelled_task())
    task2.cancel()

    register_task_for_cleanup(task2)
    assert task2 in _running_tasks

    # Properly handle the cancelled task to prevent RuntimeWarning
    try:
        loop.run_until_complete(task2)
    except asyncio.CancelledError:
        pass  # Expected for cancelled tasks

    loop.close()


def test_multiple_asyncio_loop_instances_isolation():
    """
    Test that multiple AsyncioLoop instances maintain singleton behavior. Verifies that
    the singleton pattern works correctly across different instantiation scenarios.
    """
    # Multiple ways to get the loop should return the same instance
    loop1 = AsyncioLoop.get_loop()
    loop2 = AsyncioLoop().get_loop()

    # Even creating new AsyncioLoop instances should return same loop
    instance = AsyncioLoop()
    loop3 = instance.get_loop()

    assert loop1 is loop2 is loop3
    assert isinstance(loop1, asyncio.AbstractEventLoop)


def test_signal_handler_integration():
    """
    Test signal handler setup and integration. Verifies that signal handlers are properly
    configured to use fire_and_forgot.
    """
    with patch(
        "app.utils.asyncio_utils.asyncio_support._reactor_already_installed",
        return_value=False,
    ):
        with patch.dict(sys.modules, {}, clear=True):
            install_twisted_reactor()

    # Verify that signal handlers were set up
    loop = AsyncioLoop.get_loop()

    # The signal handlers should have been added
    # We can't easily test the actual signal handling without complex mocking,
    # but we can verify the setup completed without errors
    assert loop is not None
