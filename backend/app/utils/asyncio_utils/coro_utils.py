import asyncio
import logging
import traceback
import weakref
from asyncio.tasks import Task
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# Keep track of active tasks to prevent them from being garbage collected
# Use a weak reference set to avoid memory leaks
_active_tasks: weakref.WeakSet[Task[Any]] = weakref.WeakSet()


def _done_callback(
    task: Task, error_callback: Callable[[BaseException | None], Any] | None
) -> None:
    """
    This function is called when a task is done. It checks for exceptions and handles them
    accordingly. If an error callback is provided, it will be called with the exception.
    Otherwise, the exception will be logged. This function is added as a done callback to
    the task when it is created.

    Parameters
    ----------
    task : ``Task``
        The task that has completed.
    error_callback : ``Callable[[Exception], Any] | None``
        Optional callback function to handle exceptions. If not provided, exceptions will be logged.
    """
    try:
        # Check if the task has an exception
        if task.exception():
            if error_callback:
                # Use custom error handler if provided
                error_callback(task.exception())
            else:
                # Default error logging
                exception = task.exception()
                logger.error("Unhandled exception in background task: %s", exception)

                # Get the task's stack trace for better debugging
                # type(exception) is always a type, but exception could be None
                # So, guard against None before calling format_exception
                if exception is not None:
                    task_tb = traceback.format_exception(
                        type(exception), exception, exception.__traceback__
                    )
                    logger.error("".join(task_tb))
    except asyncio.CancelledError:
        # Task was cancelled, which is normal
        pass
    except Exception as e:
        # Handle errors in the done callback itself
        logger.error("Error in task completion handler: %s", e)
    finally:
        # Remove from active tasks set if present
        if task in _active_tasks:
            _active_tasks.discard(task)


def fire_and_forgot(
    coro: Coroutine[Any, Any, Any],
    done_callback: Callable[[Task], Any] | None = None,
    error_callback: Callable[[BaseException | None], Any] | None = None,
) -> Task:
    """
    Fire and forget a coroutine while ensuring proper error handling.

    This function is used to run a coroutine without awaiting its result,
    but still provides error handling and task tracking. The task is added
    to a global set to prevent garbage collection until complete.

    Parameters
    ----------
    coro: ``Coroutine``
        The coroutine to be executed in the current event loop in the background.
    done_callback: ``Callable[[Task], Any] | None``, ( default = None )
        Optional callback function to be called when the task is done.
        This callback will be called with the task as an argument.
    error_callback: ``Callable[[Exception], Any] | None``, ( default = None )
        Optional callback function to handle exceptions. If not provided,
        exceptions will be logged. This callback will be called with the
        exception as an argument.


    Returns
    -------
    ``Task``
        The task object created from the coroutine.

    Raises
    ------
    ``RuntimeError``
        If no running event loop is found.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError as e:
        raise RuntimeError(
            "No running event loop found. Make sure you are calling this function "
            "from within an async context or event loop."
        ) from e
    # Create the task
    task = loop.create_task(coro)

    # Add it to our tracking set
    _active_tasks.add(task)

    if done_callback:
        # If a done callback is provided, add it to the task
        task.add_done_callback(done_callback)
    else:
        # Otherwise, add the default done callback
        task.add_done_callback(lambda t: _done_callback(t, error_callback))

    return task


def get_active_tasks_count() -> int:
    """
    Get the current number of active background tasks.

    Returns
    -------
    ``int``
        The number of active background tasks.
    """
    return len(_active_tasks)


async def cancel_all_tasks() -> int:
    """
    Cancel all tracked background tasks. This function is useful during shutdown to ensure
    clean termination.

    Returns
    -------
    ``int``
        The number of tasks that were cancelled.
    """
    cancelled = 0
    tasks = list(_active_tasks)  # Make a copy to avoid modification during iteration

    for task in tasks:
        if not task.done() and not task.cancelled():
            task.cancel()
            cancelled += 1

    # Wait briefly for tasks to respond to cancellation
    await asyncio.sleep(0.1)

    return cancelled
