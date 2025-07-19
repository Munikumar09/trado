import threading
from typing import Any, Type, TypeVar

T = TypeVar("T")


class Singleton(type):
    """
    Thread-safe implementation of the Singleton pattern as a metaclass.

    This metaclass ensures that only one instance of a class is created,
    regardless of how many times the class is instantiated. It is thread-safe
    and handles the case where multiple threads might try to create an instance
    simultaneously.

    Usage:
    ------
    ```python
    class MyClass(metaclass=Singleton):
        def __init__(self, arg1=None):
            self.arg1 = arg1

    # Both variables will reference the same instance
    instance1 = MyClass("first")
    instance2 = MyClass("second")  # arg1 will still be "first"
    assert instance1 is instance2  # True
    ```
    """

    _instances: dict[Type, Any] = {}
    _lock: threading.RLock = threading.RLock()

    def __call__(cls, *args, **kwargs):
        """
        Create a new instance of a class only if it doesn't already exist.
        Otherwise, return the existing instance.
        """
        # Fast path for when instance already exists
        if cls in cls._instances:
            return cls._instances[cls]

        # Slow path with lock when we need to check again or create
        with cls._lock:
            # Check again in case another thread created the instance
            # while we were waiting for the lock
            if cls not in cls._instances:
                cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

            return cls._instances[cls]

    @classmethod
    def clear_instance(mcs, target_cls: Type) -> None:
        """
        Clear the singleton instance for the given class.
        Useful for testing purposes.
        """
        with mcs._lock:
            if target_cls in mcs._instances:
                del mcs._instances[target_cls]
