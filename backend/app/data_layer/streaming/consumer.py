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
        Consumes and processes messages from a data source.

        This abstract method must be implemented by all concrete consumer classes.
        Implementations should handle the complete message consumption lifecycle
        including connection management, message processing, and error handling.

        Returns
        -------
        ``Any``
            The result of the message consumption process, which can vary
            depending on the implementation. This could be a list of messages,
            a status report, or any other relevant information.

        Raises
        ------
        ``NotImplementedError``
            If the method is not implemented by a subclass
        """

    @classmethod
    @abstractmethod
    def from_cfg(cls, cfg: DictConfig) -> Optional["Consumer"]:
        """
        Creates a consumer instance from configuration.

        Factory method to instantiate a concrete consumer implementation
        using parameters specified in a configuration object.

        Parameters
        ----------
        cfg: ``DictConfig``
            Configuration object containing parameters required to
            initialize the consumer, such as:

        Returns
        -------
        consumer: ``Consumer | None``
            An initialized concrete consumer instance ready to consume messages,
            or None if the consumer cannot be created with the given configuration

        Raises
        ------
        ``NotImplementedError``
            If the method is not implemented by a subclass
        """
