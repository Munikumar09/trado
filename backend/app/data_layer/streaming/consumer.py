""" 
This module contains the base class for all Consumer classes.
"""

from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Optional

from omegaconf import DictConfig
from registrable import Registrable

from app.core.singleton import Singleton


class ConsumerMeta(ABCMeta, Singleton, Registrable):
    """
    Custom metaclass to resolve metaclass conflicts between ABCMeta, Singleton, and Registrable.

    This metaclass combines the functionality of:
    - ABCMeta: To enforce abstract method implementation
    - Singleton: To ensure only one instance of each consumer type
    - Registrable: To enable registration and lookup by name for factory pattern
    """


class Consumer(ABC, Registrable, metaclass=ConsumerMeta):
    """
    Abstract base class for all message consumers in the streaming system.

    Defines the common interface and functionality that all concrete consumer
    implementations must follow. Uses the Singleton and Registrable patterns
    to ensure only one instance of each consumer type exists and to allow
    registration and instantiation by name.

    Example:
    --------
    ```python
    @Consumer.register("my_consumer")
    class MyConsumer(Consumer):
        def __init__(self, topic: str, group_id: str):
            self.topic = topic
            self.group_id = group_id

        async def consume_messages(self) -> None:
            # Implementation specific to MyConsumer
            pass

        @classmethod
        def from_cfg(cls, cfg: DictConfig) -> Optional["MyConsumer"]:
            return cls(
                topic=cfg.get("topic", "default_topic"),
                group_id=cfg.get("group_id", "default_group")
            )
    ```
    """

    @abstractmethod
    async def consume_messages(self) -> Any:
        """
        Consume and process messages from a data source.
        
        This abstract method must be implemented by subclasses to define the logic for message consumption, including connection management and error handling.
        
        Returns:
            The result of the message consumption process, which may vary by implementation.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_cfg(cls, cfg: DictConfig) -> Optional["Consumer"]:
        """
        Create a consumer instance from a configuration object.
        
        This abstract factory method must be implemented by subclasses to instantiate a concrete consumer using parameters from the provided configuration. Returns an instance of the consumer or None if instantiation is not possible.
        """
        raise NotImplementedError
