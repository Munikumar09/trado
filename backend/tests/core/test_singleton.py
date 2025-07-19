# pylint: disable=protected-access
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pytest

from app.core.singleton import Singleton


class MySingletonClass(metaclass=Singleton):
    """
    Test singleton class with initialization tracking.
    """

    def __init__(self, value: str = "default"):
        self.value = value
        self.initialized_at = time.time()

        # Track if __init__ was called (for testing purposes)
        if not hasattr(self, "_init_called"):
            self._init_called = True


class AnotherSingletonClass(metaclass=Singleton):
    """
    Another test singleton class to test isolation between different singleton classes.
    """

    def __init__(self, name: str = "another"):
        self.name = name


class InitCounterSingleton(metaclass=Singleton):
    """
    Singleton class that counts how many times __init__ is called.
    """

    _init_count = 0

    def __init__(self, data: Any = None):
        InitCounterSingleton._init_count += 1
        self.data = data
        self.init_number = InitCounterSingleton._init_count


@pytest.fixture(autouse=True)
def clear_singleton_instances():
    """
    Clear all singleton instances before each test to ensure test isolation.
    """
    yield
    # Clean up after each test
    Singleton.clear_instance(MySingletonClass)
    Singleton.clear_instance(AnotherSingletonClass)
    Singleton.clear_instance(InitCounterSingleton)
    InitCounterSingleton._init_count = 0


def test_basic_singleton_instance():
    """
    Test that the same instance is returned for multiple instantiations.
    """
    instance1 = MySingletonClass("first")
    instance2 = MySingletonClass("second")

    # Should be the same instance
    assert instance1 is instance2

    # Value should be from the first instantiation
    assert instance1.value == "first"
    assert instance2.value == "first"


def test_singleton_with_different_classes():
    """
    Test that different singleton classes maintain separate instances.
    """
    my_instance1 = MySingletonClass("test1")
    my_instance2 = MySingletonClass("test2")

    another_instance1 = AnotherSingletonClass("another1")
    another_instance2 = AnotherSingletonClass("another2")

    # Same class instances should be identical
    assert my_instance1 is my_instance2
    assert another_instance1 is another_instance2

    # Different class instances should be different
    assert my_instance1 is not another_instance1

    # Values should be from first instantiation of each class
    assert my_instance1.value == "test1"
    assert another_instance1.name == "another1"


def test_clear_instance():
    """
    Test the clear_instance method functionality.
    """
    # Create an instance
    instance1 = MySingletonClass("original")
    original_id = id(instance1)

    # Clear the instance
    Singleton.clear_instance(MySingletonClass)

    # Create a new instance
    instance2 = MySingletonClass("new")

    # Should be a different instance
    assert id(instance2) != original_id
    assert instance2.value == "new"


def test_clear_instance_for_specific_class_only():
    """
    Test that clearing an instance only affects the specified class.
    """
    # Create instances of both classes
    my_instance = MySingletonClass("my_value")
    another_instance = AnotherSingletonClass("another_value")

    # Store original IDs
    my_original_id = id(my_instance)
    another_original_id = id(another_instance)

    # Clear only MySingletonClass
    Singleton.clear_instance(MySingletonClass)

    # Create new instances
    my_new_instance = MySingletonClass("new_my_value")
    another_new_instance = AnotherSingletonClass("new_another_value")

    # MySingletonClass should be a new instance
    assert id(my_new_instance) != my_original_id
    assert my_new_instance.value == "new_my_value"

    # AnotherSingletonClass should be the same instance
    assert id(another_new_instance) == another_original_id
    assert another_new_instance.name == "another_value"


def test_clear_non_existent_instance():
    """
    Test that clearing a non-existent instance doesn't raise an error.
    """
    # This should not raise any exception
    Singleton.clear_instance(MySingletonClass)

    # Should be able to create an instance normally after
    instance = MySingletonClass("test")
    assert instance.value == "test"


def test_thread_safety_of_singleton():
    """
    Test that singleton creation is thread-safe.
    """
    instances = []
    exceptions = []

    def create_instance(value: str):
        """
        Function to create a singleton instance in a thread.
        """
        try:
            instance = MySingletonClass(value)
            instances.append(instance)
        except Exception as e:
            exceptions.append(e)

    # Create multiple threads that try to create instances simultaneously
    threads = []
    for i in range(20):
        thread = threading.Thread(target=create_instance, args=(f"value_{i}",))
        threads.append(thread)

    # Start all threads simultaneously
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Should have no exceptions
    assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"

    # Should have exactly 20 instances (all the same)
    assert len(instances) == 20

    # All instances should be the same object
    first_instance = instances[0]
    for instance in instances[1:]:
        assert instance is first_instance

    # The value should be from whichever thread got there first
    assert first_instance.value.startswith("value_")


def test_singleton_instance_persists_attributes():
    """
    Test that attributes set on singleton instances persist across instantiations.
    """
    # Create first instance and set an attribute
    instance1 = MySingletonClass("test")
    instance1.custom_attribute = "custom_value"

    # Create second instance
    instance2 = MySingletonClass("ignored")

    # Should be the same instance with the custom attribute
    assert instance1 is instance2
    assert hasattr(instance2, "custom_attribute")
    assert instance2.custom_attribute == "custom_value"


def test_init_called_effectively_once():
    """
    Test that __init__ is effectively called only once per singleton class.
    """
    # Create multiple instances
    instance1 = InitCounterSingleton("data1")
    instance2 = InitCounterSingleton("data2")
    instance3 = InitCounterSingleton("data3")

    # Should all be the same instance
    assert instance1 is instance2 is instance3

    # __init__ should have been called only once
    assert InitCounterSingleton._init_count == 1
    assert instance1.init_number == 1

    # Data should be from first instantiation
    assert instance1.data == "data1"
    assert instance2.data == "data1"
    assert instance3.data == "data1"


def test_thread_safety_with_thread_pool_executor():
    """
    Test thread safety using ThreadPoolExecutor for more controlled concurrency.
    """

    def create_singleton(thread_id: int) -> tuple[int, MySingletonClass]:
        """
        Create singleton and return thread_id and instance.
        """
        instance = MySingletonClass(f"thread_{thread_id}")
        return thread_id, instance

    # Use ThreadPoolExecutor to create instances concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(create_singleton, i) for i in range(50)]
        results = [future.result() for future in as_completed(futures)]

    # Extract instances
    instances = [result[1] for result in results]

    # All instances should be the same
    first_instance = instances[0]
    for instance in instances[1:]:
        assert instance is first_instance

    # Should have the value from the first thread that completed
    assert first_instance.value.startswith("thread_")


def test_singleton_with_no_init_args():
    """
    Test singleton behavior when no initialization arguments are provided.
    """
    instance1 = MySingletonClass()
    instance2 = MySingletonClass()

    assert instance1 is instance2
    assert instance1.value == "default"


def test_singleton_memory_cleanup_after_clear():
    """
    Test that memory is properly cleaned up after clearing an instance.
    """
    # Create an instance
    instance = MySingletonClass("test")
    instance_id = id(instance)

    # Clear the singleton
    Singleton.clear_instance(MySingletonClass)

    # The original instance should no longer be in the instances dict
    assert MySingletonClass not in Singleton._instances

    # Create a new instance
    new_instance = MySingletonClass("new_test")

    # Should be a completely different instance
    assert id(new_instance) != instance_id
    assert new_instance.value == "new_test"


def test_multiple_clear_operations():
    """
    Test multiple consecutive clear operations.
    """
    # Create instance
    instance1 = MySingletonClass("first")

    # Clear multiple times (should not cause issues)
    Singleton.clear_instance(MySingletonClass)
    Singleton.clear_instance(MySingletonClass)
    Singleton.clear_instance(MySingletonClass)

    # Create new instance
    instance2 = MySingletonClass("second")

    assert instance1 is not instance2
    assert instance2.value == "second"


def test_singleton_inheritance_isolation():
    """
    Test that inherited singleton classes maintain separate instances.
    """

    class ParentSingleton(metaclass=Singleton):
        """
        Parent singleton class to test inheritance isolation.
        """

        def __init__(self, value="parent"):
            self.value = value

    class ChildSingleton(ParentSingleton):
        """
        Child singleton class to test inheritance isolation.
        """

        def __init__(self, value="child"):
            super().__init__(value)

    try:
        parent1 = ParentSingleton("parent1")
        parent2 = ParentSingleton("parent2")

        child1 = ChildSingleton("child1")
        child2 = ChildSingleton("child2")

        # Parents should be the same instance
        assert parent1 is parent2
        assert parent1.value == "parent1"

        # Children should be the same instance
        assert child1 is child2
        assert child1.value == "child1"

        # Parent and child should be different instances
        assert parent1 is not child1

    finally:
        Singleton.clear_instance(ParentSingleton)
        Singleton.clear_instance(ChildSingleton)


def test_singleton_with_exception_in_init():
    """
    Test singleton behavior when __init__ raises an exception.
    """

    class ExceptionSingleton(metaclass=Singleton):
        """
        Singleton class to test exception handling in __init__.
        """

        def __init__(self, should_fail=False):
            if should_fail:
                raise ValueError("Initialization failed")
            self.value = "success"

    try:
        # First attempt should fail
        with pytest.raises(ValueError, match="Initialization failed"):
            ExceptionSingleton(should_fail=True)

        # Class should not be in instances after failed initialization
        assert ExceptionSingleton not in Singleton._instances

        # Second attempt should succeed
        instance = ExceptionSingleton(should_fail=False)
        assert instance.value == "success"

        # Third attempt should return the same successful instance
        instance2 = ExceptionSingleton(should_fail=True)
        assert instance is instance2
        assert instance2.value == "success"

    finally:
        Singleton.clear_instance(ExceptionSingleton)


def test_singleton_with_kwargs():
    """
    Test singleton with keyword arguments.
    """

    class KwargsSingleton(metaclass=Singleton):
        """
        Singleton class to test keyword arguments.
        """

        def __init__(self, name="default", age=0, **kwargs):
            self.name = name
            self.age = age
            self.extra = kwargs

    try:
        instance1 = KwargsSingleton(
            name="Alice", age=30, city="New York", country="USA"
        )
        instance2 = KwargsSingleton(
            name="Bob", age=25, city="London"
        )  # These will be ignored

        assert instance1 is instance2
        assert instance1.name == "Alice"
        assert instance1.age == 30
        assert instance1.extra == {"city": "New York", "country": "USA"}

    finally:
        Singleton.clear_instance(KwargsSingleton)


def test_singleton_stress_test_rapid_creation():
    """
    Stress test with rapid singleton creation and clearing.
    """

    class StressSingleton(metaclass=Singleton):
        """
        Singleton class to test rapid creation and clearing.
        """

        def __init__(self, value="stress"):
            self.value = value
            self.creation_time = time.time()

    try:
        expected_value = "iteration_0"

        for i in range(100):
            # Create instance
            instance = StressSingleton(f"iteration_{i}")
            assert (
                instance.value == expected_value
            )  # Should be the value from current cycle's first creation

            # Every 10 iterations, clear and recreate
            if i % 10 == 9:
                Singleton.clear_instance(StressSingleton)
                new_instance = StressSingleton(f"cleared_{i}")
                assert new_instance.value == f"cleared_{i}"
                Singleton.clear_instance(StressSingleton)

                # Update expected value for next cycle
                if i + 1 < 100:
                    expected_value = f"iteration_{i + 1}"

    finally:
        Singleton.clear_instance(StressSingleton)


def test_singleton_with_property_and_methods():
    """
    Test singleton with properties and methods.
    """

    class ComplexSingleton(metaclass=Singleton):
        """
        Complex singleton class to test properties and methods.
        """

        def __init__(self, initial_value=0):
            self._value = initial_value
            self._call_count = 0

        @property
        def value(self):
            """
            Property to get the value of the singleton.
            """
            return self._value

        @value.setter
        def value(self, new_value):
            self._value = new_value

        def increment(self):
            """
            Increment the call count.
            """
            self._call_count += 1
            return self._call_count

    try:
        instance1 = ComplexSingleton(42)
        instance2 = ComplexSingleton(100)  # Ignored

        assert instance1 is instance2
        assert instance1.value == 42

        # Test property setter
        instance1.value = 99
        assert instance2.value == 99

        # Test method calls
        count1 = instance1.increment()
        count2 = instance2.increment()

        assert count1 == 1
        assert count2 == 2
        assert instance1._call_count == 2

    finally:
        Singleton.clear_instance(ComplexSingleton)


def test_singleton_instances_dict_thread_safety():
    """
    Test that the _instances dictionary itself is thread-safe.
    """
    results = {}
    exceptions = []

    def create_and_clear_singleton(thread_id: int):
        """
        Create singleton, do operations, then clear it.
        """
        try:

            class ThreadSingleton(metaclass=Singleton):
                """
                Singleton class to test thread safety.
                """

                def __init__(self, tid):
                    self.thread_id = tid

            # Create instance
            instance = ThreadSingleton(thread_id)
            results[thread_id] = instance

            # Small delay to increase chance of race conditions
            time.sleep(0.001)

            # Clear instance
            Singleton.clear_instance(ThreadSingleton)

        except Exception as e:
            exceptions.append((thread_id, e))

    # Run multiple threads
    threads = []
    for i in range(50):
        thread = threading.Thread(target=create_and_clear_singleton, args=(i,))
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # Should have no exceptions
    assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"

    # Should have results from all threads
    assert len(results) == 50
