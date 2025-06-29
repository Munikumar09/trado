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
        """
        Initialize the singleton instance with a value and timestamp.
        
        Parameters:
            value (str): The value to assign to the instance. Defaults to "default".
        """
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
        """
        Initialize the singleton instance with a name attribute.
        
        Parameters:
            name (str): The name to assign to the instance. Defaults to "another".
        """
        self.name = name


class InitCounterSingleton(metaclass=Singleton):
    """
    Singleton class that counts how many times __init__ is called.
    """

    _init_count = 0

    def __init__(self, data: Any = None):
        """
        Initialize the singleton instance, incrementing the class-level initialization counter and storing the provided data.
        
        Parameters:
            data (Any, optional): Data to associate with the instance. Defaults to None.
        """
        InitCounterSingleton._init_count += 1
        self.data = data
        self.init_number = InitCounterSingleton._init_count


@pytest.fixture(autouse=True)
def clear_singleton_instances():
    """
    Pytest fixture that clears singleton instances and resets counters before and after each test to ensure test isolation.
    """
    yield
    # Clean up after each test
    Singleton.clear_instance(MySingletonClass)
    Singleton.clear_instance(AnotherSingletonClass)
    Singleton.clear_instance(InitCounterSingleton)
    InitCounterSingleton._init_count = 0


def test_basic_singleton_instance():
    """
    Verify that multiple instantiations of the same singleton class return the identical instance and preserve the initial initialization arguments.
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
    Verify that singleton instances are unique per class, ensuring different singleton classes do not share instances and that each preserves its own initialization arguments from the first instantiation.
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
    Verify that clearing a singleton instance removes it, allowing a new instance to be created with different initialization arguments.
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
    Verify that clearing a singleton instance for one class does not affect singleton instances of other classes.
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
    Verify that clearing a singleton instance that does not exist does not raise an error and allows normal instantiation afterward.
    """
    # This should not raise any exception
    Singleton.clear_instance(MySingletonClass)

    # Should be able to create an instance normally after
    instance = MySingletonClass("test")
    assert instance.value == "test"


def test_thread_safety_of_singleton():
    """
    Verifies that singleton instance creation is thread-safe by concurrently instantiating the singleton in multiple threads and ensuring all threads receive the same instance without exceptions.
    """
    instances = []
    exceptions = []

    def create_instance(value: str):
        """
        Creates a `MySingletonClass` instance with the given value and appends it to the shared instances list, capturing any exceptions that occur during instantiation.
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
    Verify that attributes added to a singleton instance remain accessible across subsequent instantiations of the same class.
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
    Verify that the singleton's `__init__` method is called only once, regardless of multiple instantiations, and that only the first initialization arguments are used.
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
    Verifies that singleton instance creation remains thread-safe when using ThreadPoolExecutor, ensuring all concurrent threads receive the same instance and that initialization arguments from the first completed thread are retained.
    """

    def create_singleton(thread_id: int) -> tuple[int, MySingletonClass]:
        """
        Creates a `MySingletonClass` singleton instance using the thread ID and returns a tuple of the thread ID and the singleton instance.
        
        Parameters:
            thread_id (int): Identifier for the thread, used to initialize the singleton.
        
        Returns:
            tuple[int, MySingletonClass]: A tuple containing the thread ID and the singleton instance.
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
    Verify that a singleton class returns the same instance when instantiated without arguments and uses default initialization values.
    """
    instance1 = MySingletonClass()
    instance2 = MySingletonClass()

    assert instance1 is instance2
    assert instance1.value == "default"


def test_singleton_memory_cleanup_after_clear():
    """
    Verify that clearing a singleton instance removes it from memory, allowing a new, distinct instance to be created with subsequent initialization arguments.
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
    Verify that performing multiple consecutive clear operations on a singleton class does not cause errors and allows creation of a new, distinct instance afterward.
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
    Verifies that singleton instances of a parent class and its child class are distinct, ensuring inheritance does not share singleton instances.
    """

    class ParentSingleton(metaclass=Singleton):
        """
        Parent singleton class to test inheritance isolation.
        """

        def __init__(self, value="parent"):
            """
            Initialize the singleton instance with a specified value.
            
            Parameters:
                value (str): The value to assign to the instance. Defaults to "parent".
            """
            self.value = value

    class ChildSingleton(ParentSingleton):
        """
        Child singleton class to test inheritance isolation.
        """

        def __init__(self, value="child"):
            """
            Initialize the child singleton class with a specified value, defaulting to "child".
            """
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
    Tests that if a singleton's `__init__` raises an exception, the instance is not cached, allowing subsequent successful initialization and ensuring later instantiations return the same instance.
    """

    class ExceptionSingleton(metaclass=Singleton):
        """
        Singleton class to test exception handling in __init__.
        """

        def __init__(self, should_fail=False):
            """
            Initialize the singleton instance, optionally raising an exception to simulate initialization failure.
            
            Parameters:
                should_fail (bool): If True, raises a ValueError to simulate a failed initialization.
            """
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
    Verifies that a singleton class initialized with keyword arguments only uses the arguments from the first instantiation, and subsequent instantiations return the same instance with the original values.
    """

    class KwargsSingleton(metaclass=Singleton):
        """
        Singleton class to test keyword arguments.
        """

        def __init__(self, name="default", age=0, **kwargs):
            """
            Initialize the singleton instance with a name, age, and any additional keyword arguments.
            
            Parameters:
                name (str): The name to assign to the instance. Defaults to "default".
                age (int): The age to assign to the instance. Defaults to 0.
                **kwargs: Additional attributes to store in the instance.
            """
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
    Performs a stress test by rapidly creating and clearing a singleton instance in a loop.
    
    This test verifies that the singleton implementation maintains correct instance value retention and allows proper recreation after clearing, even under repeated rapid operations.
    """

    class StressSingleton(metaclass=Singleton):
        """
        Singleton class to test rapid creation and clearing.
        """

        def __init__(self, value="stress"):
            """
            Initialize the singleton instance with a value and record its creation time.
            
            Parameters:
                value (str): The value to assign to the instance. Defaults to "stress".
            """
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
    Tests that a singleton class with properties, setters, and methods maintains consistent state and behavior across multiple instantiations, verifying property persistence and method call effects.
    """

    class ComplexSingleton(metaclass=Singleton):
        """
        Complex singleton class to test properties and methods.
        """

        def __init__(self, initial_value=0):
            """
            Initialize the singleton instance with an initial value and reset the call count.
            
            Parameters:
            	initial_value (int, optional): The starting value for the instance. Defaults to 0.
            """
            self._value = initial_value
            self._call_count = 0

        @property
        def value(self):
            """
            Returns the value stored in the singleton instance.
            """
            return self._value

        @value.setter
        def value(self, new_value):
            """
            Set the value of the singleton instance.
            
            Parameters:
            	new_value: The new value to assign to the instance.
            """
            self._value = new_value

        def increment(self):
            """
            Increments and returns the current call count.
            
            Returns:
                int: The updated call count after incrementing.
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
    Verifies that the internal singleton instances dictionary remains thread-safe when multiple threads concurrently create and clear singleton instances.
    
    This test launches multiple threads, each creating and clearing a singleton instance, and asserts that no exceptions occur and all threads complete their operations successfully.
    """
    results = {}
    exceptions = []

    def create_and_clear_singleton(thread_id: int):
        """
        Creates a singleton instance in a thread, stores it, introduces a brief delay, and then clears the instance.
        
        Intended for use in multithreaded tests to verify thread safety of singleton creation and clearing. Any exceptions encountered are recorded with the thread ID.
        """
        try:

            class ThreadSingleton(metaclass=Singleton):
                """
                Singleton class to test thread safety.
                """

                def __init__(self, tid):
                    """
                    Initialize the singleton instance with the given thread identifier.
                    
                    Parameters:
                        tid: The thread identifier to associate with this instance.
                    """
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
