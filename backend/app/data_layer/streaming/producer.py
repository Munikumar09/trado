"""
This module contains the base class for all Producer classes.
"""

from abc import ABC, abstractmethod
from typing import Optional

from omegaconf import DictConfig
from registrable import Registrable


class Producer(ABC, Registrable):
    """
    Abstract base class for all message producers in the streaming system.

    Defines the common interface that all concrete producer implementations
    must follow. Implements the Registrable pattern to allow registration
    and instantiation by name for a factory pattern approach.

    Example:
    --------
    ```python
    @Producer.register("my_producer")
    class MyProducer(Producer):
        def __init__(self, topic: str, server: str):
            self.topic = topic
            self.server = server
            # Setup connection, etc.

        def __call__(self, data: str) -> bool:
            # Send data to the streaming server
            # Return True if successful, False otherwise
            return True

        def close(self) -> None:
            # Cleanup resources
            pass

        @classmethod
        def from_cfg(cls, cfg: DictConfig) -> Optional["MyProducer"]:
            return cls(
                topic=cfg.get("topic", "default_topic"),
                server=cfg.get("server", "localhost:9092")
            )
    ```
    """

    @abstractmethod
    def __call__(self, data: str) -> bool:
        """
        Send a data message to the streaming system.
        
        Parameters:
            data (str): The serialized data to send, such as a JSON string.
        
        Returns:
            bool: True if the data was successfully sent; False otherwise.
        
        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """
        Closes the producer and releases all associated resources.
        
        This method should be called when the producer is no longer needed to ensure that buffered messages are flushed and resources such as connections and threads are properly released.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_cfg(cls, cfg: DictConfig) -> Optional["Producer"]:
        """
        Instantiate a concrete producer from a configuration object.
        
        This factory method creates and returns a producer instance using parameters provided in the configuration. Returns None if instantiation is not possible with the given configuration.
        
        Returns:
            Producer or None: An initialized producer instance, or None if creation fails.
        
        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError
