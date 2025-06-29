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
    Handles completion of a background asyncio task, invoking an error callback or logging exceptions if they occurred.
    
    If the task raised an exception, calls the provided error callback with the exception, or logs the exception and its traceback if no callback is given. Silently ignores task cancellations. Ensures the completed task is removed from the active tasks set.
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
    Schedules a coroutine to run in the background with error handling and lifecycle tracking.
    
    Creates an asyncio Task from the given coroutine and adds it to a global set to prevent premature garbage collection. Attaches a completion callback for error reporting or custom handling. Raises a RuntimeError if called outside an active event loop.
    
    Parameters:
        coro (Coroutine): The coroutine to execute as a background task.
        done_callback (Callable[[Task], Any], optional): Callback invoked when the task completes.
        error_callback (Callable[[BaseException | None], Any], optional): Callback invoked if the task raises an exception.
    
    Returns:
        Task: The asyncio Task object representing the scheduled coroutine.
    
    Raises:
        RuntimeError: If no running event loop is found.
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
    Return the number of currently tracked active background asyncio tasks.
    
    Returns:
        int: The count of active tasks being managed.
    """
    return len(_active_tasks)


async def cancel_all_tasks() -> int:
    """
    Cancel all currently tracked background asyncio tasks.
    
    This coroutine issues cancellation requests to all active tasks that are not already completed or cancelled, then briefly awaits to allow tasks to process the cancellation.
    
    Returns:
        int: The number of tasks that were cancelled.
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
