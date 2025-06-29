# pylint: disable=import-outside-toplevel
import asyncio
import atexit
import signal
import sys
from asyncio import AbstractEventLoop, CancelledError, Task
from pathlib import Path
from typing import Any, Callable, List, Optional, Set

from app.core.singleton import Singleton
from app.utils.asyncio_utils.coro_utils import fire_and_forgot
from app.utils.common.logger import get_logger

logger = get_logger(Path(__file__).name)


# Track tasks registered for graceful shutdown
_shutdown_tasks: List[Callable[[], Any]] = []

# Tasks to be cancelled during cleanup
_running_tasks: Set[Task] = set()


class AsyncioLoop(metaclass=Singleton):
    """
    Singleton class to manage the asyncio event loop. This class ensures that only one
    instance of the event loop is created and used throughout the application
    """

    _loop: Optional[AbstractEventLoop] = None

    @classmethod
    def get_loop(cls) -> AbstractEventLoop:
        """
        Returns the singleton asyncio event loop instance, creating and configuring a new one if necessary.
        
        Ensures that a single event loop is used throughout the application, setting up exception handling on creation.
        
        Returns:
            AbstractEventLoop: The singleton event loop instance.
        """
        if cls._loop is None or cls._loop.is_closed():
            cls._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(cls._loop)
            _setup_exception_handling(cls._loop)
            logger.info("Created new asyncio event loop (singleton).")

        return cls._loop


class LoopNotRunningError(RuntimeError):
    """Exception raised when event loop access is attempted but no loop is running."""


def _setup_exception_handling(loop: AbstractEventLoop) -> None:
    """
    Sets a custom exception handler on the given event loop to log unhandled exceptions, excluding cancellations during shutdown.
    
    Parameters:
        loop (AbstractEventLoop): The event loop on which to configure exception handling.
    """

    def exception_handler(
        loop: AbstractEventLoop, context: dict  # pylint: disable=unused-argument
    ) -> None:
        """
        Handles unhandled exceptions in the asyncio event loop, logging errors with contextual information except for normal task cancellations.
        """
        exception = context.get("exception")
        message = context.get("message")

        if exception is None:
            if message:
                logger.error("Async error without exception: %s", message)
            return

        if isinstance(exception, CancelledError):
            # Don't log cancelled tasks, they're normal during shutdown
            return

        # Get relevant context
        task = context.get("task")
        handle = context.get("handle")
        protocol = context.get("protocol")
        transport = context.get("transport")

        log_context = {
            "task": repr(task) if task else "N/A",
            "handle": repr(handle) if handle else "N/A",
            "protocol": repr(protocol) if protocol else "N/A",
            "transport": repr(transport) if transport else "N/A",
        }

        logger.error(
            "Unhandled exception in async operation: %s",
            str(exception),
            exc_info=exception,
            extra=log_context,
        )

    loop.set_exception_handler(exception_handler)


def _install_twisted_reactor(loop: AbstractEventLoop) -> None:
    """
    Installs Twisted's AsyncioSelectorReactor using the specified asyncio event loop.
    
    This function must be called before importing `twisted.internet.reactor`. If Twisted or its asyncioreactor is unavailable, the installation is skipped. If the reactor is already installed, a warning is logged and the function continues without error. Unexpected exceptions during installation are re-raised.
    """
    try:
        from twisted.internet import asyncioreactor
    except ImportError:
        logger.warning("Twisted or asyncioreactor not available. Skipping install.")
        return

    try:

        logger.info("Installing Twisted AsyncioSelectorReactor...")
        asyncioreactor.install(loop)
        logger.info("Twisted reactor installed successfully.")
    except Exception as e:
        if (
            "ReactorAlreadyInstalledError" in str(e)
            or "reactor already installed" in str(e).lower()
        ):
            logger.warning("Twisted reactor already installed: %s. Continuing...", e)
        else:
            logger.error("Unexpected error installing Twisted reactor: %s", e)
            raise


def _reactor_already_installed() -> bool:
    """
    Determine whether the Twisted reactor is already imported and installed.
    
    Returns:
        bool: True if the Twisted reactor is present and appears installed, False otherwise.
    """
    if "twisted.internet.reactor" in sys.modules:
        try:
            from twisted.internet import reactor

            # Avoid direct access to __class__ attribute for linting compatibility
            return hasattr(reactor, "running") or type(reactor).__name__ == "Reactor"
        except ImportError:
            pass

    return False


def install_twisted_reactor() -> None:
    """
    Initializes the singleton asyncio event loop, installs the Twisted reactor for asyncio integration if not already present, and sets up graceful shutdown handlers.
    
    This function should be called at application startup to ensure proper event loop and reactor setup. It installs signal handlers for SIGTERM and SIGINT on non-Windows platforms to trigger orderly shutdown, and registers an atexit handler to perform cleanup when the interpreter exits. Reactor installation is skipped if already installed or if running under pytest.
    """

    # Create the singleton loop
    loop = AsyncioLoop.get_loop()

    # Install Twisted reactor if not already installed
    if _reactor_already_installed():
        logger.info("Twisted reactor already installed. Skipping installation.")
        return

    # Install twisted reactor
    try:
        # Skip reactor install if we're in a pytest context
        # pytest-asyncio will handle its own event loop
        if "pytest_asyncio" not in sys.modules and "pytest" not in sys.modules:
            _install_twisted_reactor(loop)
        else:
            logger.info("Detected pytest environment. Skipping reactor install.")
    except ImportError:
        logger.warning("Twisted not available. Limited functionality may result.")
    except Exception as e:
        logger.error("Failed to install Twisted reactor: %s", e)

    # Set up signal handlers for graceful shutdown
    if not sys.platform.startswith("win"):
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig,
                    lambda sig=sig: fire_and_forgot(
                        shutdown(sig=sig),
                        done_callback=lambda t: logger.info(
                            "Shutdown task completed: %s", t
                        ),
                        error_callback=lambda e: logger.error(
                            "Error in shutdown task: %s", e
                        ),
                    ),
                )
            logger.info("Installed signal handlers for graceful shutdown")
        except Exception as e:
            logger.warning("Failed to set up signal handlers: %s", e)

    # Register atexit handler for when the interpreter is shutting down
    atexit.register(
        lambda: loop.run_until_complete(shutdown()) if loop.is_running() else None
    )


def register_shutdown_handler(handler: Callable[[], Any]) -> None:
    """
    Registers a callable to be executed during application shutdown.
    
    Shutdown handlers are executed in reverse order of registration (LIFO) to ensure proper cleanup of dependent resources. Both synchronous and asynchronous callables are supported.
    """
    _shutdown_tasks.append(handler)
    logger.debug("Registered shutdown handler: %s", handler.__name__)


def register_task_for_cleanup(task: Task) -> None:
    """
    Registers an asyncio Task to be cancelled during application shutdown.
    
    The task is tracked for cleanup and will be automatically removed from tracking once it completes.
    """
    _running_tasks.add(task)

    # Auto-remove task when done
    task.add_done_callback(
        lambda t: _running_tasks.discard(t) if t in _running_tasks else None
    )


async def shutdown(sig: signal.Signals | None = None) -> None:
    """
    Performs a graceful shutdown by cancelling tracked asyncio tasks and executing registered shutdown handlers.
    
    If a termination signal is provided, logs the signal and exits the process after shutdown. Cancels all tasks registered for cleanup, waits for their cancellation, and runs all shutdown handlers in reverse order. Handles both synchronous and asynchronous shutdown handlers.
    """
    if sig:
        logger.info(
            "Received shutdown signal %s, initiating graceful shutdown...", sig.name
        )
    else:
        logger.info("Initiating graceful shutdown...")

    # Cancel all tracked tasks
    if _running_tasks:
        logger.info("Cancelling %d running tasks", len(_running_tasks))
        for task in _running_tasks:
            if not task.done() and not task.cancelled():
                task.cancel()

        # Wait for tasks to finish cancellation (with timeout)
        pending = [task for task in _running_tasks if not task.done()]
        if pending:
            try:
                await asyncio.wait(pending, timeout=5.0)
            except Exception as e:
                logger.error("Error waiting for tasks to cancel: %s", e)

    # Call shutdown handlers in reverse order (LIFO)
    for handler in reversed(_shutdown_tasks):
        try:
            logger.debug("Running shutdown handler: %s", handler.__name__)
            result = handler()

            # Handle coroutines
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error("Error in shutdown handler %s: %s", handler.__name__, e)

    logger.info("Graceful shutdown completed.")

    # If we received a termination signal, exit the process
    if sig:
        sys.exit(0)
