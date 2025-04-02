""" 
This module contains the base class for all streaming classes.
"""

from abc import ABC, abstractmethod
from typing import Optional

from omegaconf import DictConfig
from registrable import Registrable


class Streamer(ABC, Registrable):
    """
    This is the base class for all streaming classes. All the concrete streaming classes
    should inherit from this class and implement the __call__ method. This class also
    provides a class method to create a streaming object from the configuration file.
    """

    @abstractmethod
    def __call__(self, data: str):
        """
        This method should be implemented by the concrete streaming classes. This method
        should send the received data to the streaming server.

        Parameters:
        -----------
        data: ``str``
            The data to be sent to the streaming server
        """
        raise NotImplementedError

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Optional["Streamer"]:
        """
        This class method creates a streaming object from the configuration file.

        Parameters:
        -----------
        cfg: ``DictConfig``
            The configuration file containing the streaming server details

        Returns:
        --------
        ``Optional[Streaming]``
            The streaming object created from the configuration file
        """
        raise NotImplementedError
