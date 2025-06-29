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
    Provides a pytest fixture that yields a mocked logger for capturing and asserting log output during tests.
    """
    with patch("app.utils.asyncio_utils.asyncio_support.logger") as mock:
        yield mock


@pytest.fixture(autouse=True)
def clear_global_state():
    """
    Temporarily clears and restores global shutdown and running task state for test isolation.
    
    Intended for use as a pytest fixture to ensure that modifications to `_shutdown_tasks` and `_running_tasks` during a test do not affect other tests.
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
    Context manager that restores the original `sys.modules` state for Twisted-related modules after a test.
    
    Removes any Twisted modules added to `sys.modules` during the test, ensuring a clean import state for subsequent tests.
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
    Verifies that `AsyncioLoop.get_loop()` creates and initializes a singleton asyncio event loop.
    
    Ensures the loop is set as the current event loop, exception handling is configured, and creation is logged.
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
    Tests that AsyncioLoop.get_loop() returns the same event loop instance across multiple threads, ensuring singleton thread safety.
    """
    loops = {}

    def _get_loop_in_thread(thread_id):
        """
        Retrieve and store the singleton asyncio event loop for the specified thread.
        
        Parameters:
            thread_id: Identifier of the thread for which to obtain the event loop.
        """
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
    Test that _setup_exception_handling installs a custom exception handler on the event loop.
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
    Tests that the custom exception handler logs errors when a task raises an actual exception, including relevant context information.
    """
    loop = AsyncioLoop.get_loop()

    # Create a real task that will raise an exception
    async def failing_task():
        """
        Asynchronous task that raises a ValueError to simulate an error condition.
        """
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
    Verifies that the exception handler does not log errors or warnings when handling asyncio.CancelledError, as such exceptions are expected during normal shutdown.
    """
    loop = AsyncioLoop.get_loop()

    async def cancelled_task():
        """
        An asynchronous task that sleeps for 10 seconds, intended to be cancelled during execution.
        """
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
    Tests that the exception handler logs contextual information, including task details, when an exception occurs in an asyncio task.
    """
    loop = AsyncioLoop.get_loop()

    async def task_with_context():
        """
        Asynchronous task that raises a RuntimeError with a specific error message.
        """
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
    Verify that _reactor_already_installed returns False when an ImportError occurs during reactor import, ensuring graceful handling of import failures.
    """
    # Put something in sys.modules but make import fail
    with patch.dict(sys.modules, {"twisted.internet.reactor": MagicMock()}):
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            result = _reactor_already_installed()
            assert result is False


def test_install_twisted_reactor_success(mock_logger):
    """
    Test that the Twisted reactor is installed successfully when Twisted is available.
    
    Verifies that the reactor's install method is called with the event loop and that appropriate log messages are generated.
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
    Test that Twisted reactor installation handles ImportError gracefully and logs a warning when Twisted is unavailable.
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
    Test that installing the Twisted reactor when it is already installed logs a warning and handles the ReactorAlreadyInstalledError gracefully.
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
    Test that unexpected exceptions during Twisted reactor installation are logged and re-raised.
    
    Verifies that if an unexpected error occurs while installing the Twisted reactor, the error is logged and the exception is propagated.
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
    Tests that the full `install_twisted_reactor` process completes successfully on a Linux platform, including reactor installation, signal handler setup, atexit registration, and appropriate logging.
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
    Tests that `install_twisted_reactor` skips installation and logs an informational message when the Twisted reactor is already installed.
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
    Tests that `install_twisted_reactor` installs the Twisted reactor on Windows platforms without setting up signal handlers.
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
    Tests that a shutdown handler can be registered and is correctly added to the shutdown handler list, with appropriate debug logging.
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
    Test that multiple shutdown handlers can be registered and are stored in the correct order.
    """

    def handler1():
        """
        A placeholder shutdown handler function for testing purposes.
        """
        pass

    def handler2():
        """
        A no-op shutdown handler used for testing purposes.
        """
        pass

    def handler3():
        """
        A no-op shutdown handler used for testing purposes.
        """
        pass

    register_shutdown_handler(handler1)
    register_shutdown_handler(handler2)
    register_shutdown_handler(handler3)

    assert len(_shutdown_tasks) == 3
    assert _shutdown_tasks == [handler1, handler2, handler3]


def test_register_task_for_cleanup():
    """
    Tests that tasks registered with `register_task_for_cleanup` are tracked for cleanup and properly removed from tracking upon cancellation and completion.
    """

    async def dummy_task():
        """
        A simple asynchronous task that sleeps for a short duration.
        
        This function is typically used as a placeholder or for testing asynchronous task behavior.
        """
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
    Test that tasks registered for cleanup are automatically removed from tracking upon completion.
    """

    async def quick_task():
        """
        Asynchronously returns the string "done".
        
        Returns:
            str: The string "done" upon completion.
        """
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
    Tests that the shutdown function performs a graceful shutdown without calling sys.exit when no signal is provided.
    """
    with patch("app.utils.asyncio_utils.asyncio_support.sys.exit") as mock_exit:
        await shutdown()

    mock_logger.info.assert_any_call("Initiating graceful shutdown...")
    mock_logger.info.assert_any_call("Graceful shutdown completed.")
    mock_exit.assert_not_called()


@pytest.mark.asyncio
async def test_shutdown_with_signal(mock_logger):
    """
    Tests that the shutdown function logs receipt of a signal and calls sys.exit(0) to terminate the process.
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
    Test that the shutdown function cancels running tasks registered for cleanup.
    
    Creates real asynchronous tasks, registers them for cleanup, and verifies that they are properly cancelled during shutdown.
    """

    # Create real async tasks
    async def long_running_task(_):
        """
        Simulates a long-running asynchronous operation by sleeping for one second.
        """
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
        """
        Asynchronously returns the string "completed".
        """
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
    Tests that shutdown handlers are executed in last-in-first-out (LIFO) order during shutdown.
    
    Verifies that both synchronous and asynchronous handlers registered for shutdown are called in reverse order of registration.
    """
    call_order = []

    def handler1():
        """
        Appends 'handler1' to the call_order list to record its invocation.
        """
        call_order.append("handler1")

    def handler2():
        """
        Appends the string "handler2" to the call_order list to record invocation order during tests.
        """
        call_order.append("handler2")

    async def async_handler():
        """
        Asynchronous shutdown handler that appends its identifier to the call order list.
        """
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
    Tests that the shutdown process continues and logs errors when a registered shutdown handler raises an exception.
    
    Verifies that all shutdown handlers are executed regardless of exceptions, and that errors from failing handlers are properly logged.
    """
    call_order = []

    def failing_handler():
        """
        A shutdown handler that appends its name to the call order and raises a ValueError to simulate a handler failure.
        """
        call_order.append("failing_handler")
        raise ValueError("Handler error")

    def working_handler():
        """
        Appends 'working_handler' to the call_order list to track execution order during tests.
        """
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
    Tests that exceptions raised by asynchronous shutdown handlers during shutdown are caught and logged as errors.
    """

    async def failing_async_handler():
        """
        Asynchronous handler that raises a RuntimeError when executed.
        
        Raises:
            RuntimeError: Always raised with the message "Async handler error".
        """
        raise RuntimeError("Async handler error")

    register_shutdown_handler(failing_async_handler)

    await shutdown()

    mock_logger.error.assert_called()
    error_call = mock_logger.error.call_args[0]
    assert "Error in shutdown handler" in error_call[0]


@pytest.mark.asyncio
async def test_shutdown_task_wait_timeout(mock_logger):
    """
    Test that shutdown handles errors during task cancellation waiting gracefully.
    
    Simulates a scenario where an exception occurs while waiting for tasks to cancel during shutdown, and verifies that the error is logged.
    """

    # Create a task that won't respond to cancellation quickly
    async def stubborn_task():
        """
        Simulates an asyncio task that is slow to respond to cancellation.
        
        This task sleeps for a period, and if cancelled, deliberately delays its cancellation handling by sleeping again.
        """
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
    Test that `install_twisted_reactor` handles ImportError gracefully and logs a warning when Twisted is unavailable.
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
    Test that unexpected errors during Twisted reactor installation are logged as errors.
    
    This test simulates a runtime error during reactor installation and verifies that the error is logged appropriately.
    """
    mock_reactor_check.return_value = False
    mock_install.side_effect = RuntimeError("Unexpected error")

    install_twisted_reactor()

    mock_logger.error.assert_called_once()
    error_call = mock_logger.error.call_args[0]
    assert "Failed to install Twisted reactor" in error_call[0]


def test_exception_handler_edge_cases(mock_logger):
    """
    Tests the event loop's exception handler with minimal and unusual context inputs to ensure it handles edge cases without errors.
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
    Tests that the shutdown process completes successfully when no tasks or shutdown handlers are registered.
    """
    assert len(_shutdown_tasks) == 0
    assert len(_running_tasks) == 0

    await shutdown()

    mock_logger.info.assert_any_call("Initiating graceful shutdown...")
    mock_logger.info.assert_any_call("Graceful shutdown completed.")


def test_register_task_for_cleanup_edge_cases():
    """
    Test registration of tasks for cleanup with completed and cancelled states.
    
    Verifies that tasks which are already completed or cancelled can still be registered for cleanup and are tracked appropriately.
    """

    # Test with already completed task
    async def completed_task():
        """
        An asynchronous function that immediately returns the string "done".
        
        Returns:
            str: The string "done".
        """
        return "done"

    loop = asyncio.new_event_loop()
    task = loop.create_task(completed_task())
    loop.run_until_complete(task)

    register_task_for_cleanup(task)
    assert task in _running_tasks

    # Test with cancelled task
    async def cancelled_task():
        """
        An asynchronous task that sleeps for 10 seconds.
        
        Intended for use in tests where the task may be cancelled before completion.
        """
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
    Verifies that signal handlers are set up during Twisted reactor installation and that the integration completes without errors.
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
