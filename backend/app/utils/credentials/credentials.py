"""
This module provides a base class for credentials used in the market data API.

The `Credentials` class is an abstract base class that defines the common 
interface for all types of credentials. Subclasses of `Credentials` should
implement the `get_credentials` method to provide the actual credentials.
"""

from abc import ABC, abstractmethod

from registrable import Registrable


@abstractmethod
class Credentials(Registrable, ABC):
    """
    Base class for credentials used in the market data API. Subclasses of
    `Credentials` should implement the `get_credentials` method to provide
    the actual credentials.
    """

    @staticmethod
    def get_credentials():
        """
        This method should be implemented by subclasses to provide the actual credentials.

        Raises
        ------
        ``NotImplementedError``
            If the method is not implemented in the subclass
        """
        raise NotImplementedError("get_credentials method not implemented")
