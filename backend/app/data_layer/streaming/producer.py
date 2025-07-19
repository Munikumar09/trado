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
        Send data to the streaming system.

        This callable method provides a convenient interface for sending data
        to the underlying streaming platform. Concrete implementations should
        handle serialization, batching, and error handling as appropriate.

        Parameters
        ----------
        data: ``str``
            The data to send to the streaming system, typically a JSON string
            or other serialized message format

        Returns
        -------
        success: ``bool``
            True if the data was successfully sent to the streaming system,
            False if the send operation failed

        Raises
        ------
        ``NotImplementedError``
            If the method is not implemented by a subclass
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """
        Release resources and close connections.

        Properly shuts down the producer, ensuring that any buffered messages
        are flushed and all resources (connections, threads, etc.) are properly
        released. This method should be called when the producer is no longer needed.

        Raises
        ------
        ``NotImplementedError``
            If the method is not implemented by a subclass
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_cfg(cls, cfg: DictConfig) -> Optional["Producer"]:
        """
        Create a producer instance from configuration.

        Factory method to instantiate a concrete producer implementation
        using parameters specified in a configuration object.

        Parameters
        ----------
        cfg: ``DictConfig``
            Configuration object containing parameters required to
            initialize the producer, such as:

        Returns
        -------
        ``Producer | None``
            An initialized concrete producer instance ready to send messages,
            or None if the producer cannot be created with the given configuration

        Raises
        ------
        ``NotImplementedError``
            If the method is not implemented by a subclass
        """
        raise NotImplementedError
