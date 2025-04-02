"""
This module provides functionality for managing connection credentials.

The module defines the DataProvider enum, which represents the available 
connection credentials. It also provides a function to retrieve the 
credentials for a given credentials name.

Example:
    credentials = get_credentials(DataProvider.SMARTAPI)

"""

from enum import Enum

from app.utils.credentials.credentials import Credentials


class DataProvider(Enum):
    """
    Enum class representing different data providers.
    """

    SMARTAPI = "smartapi"

    @staticmethod
    def by_name(name: str):
        """
        Get the Dataprovider enum value by name.

        Parameters
        ----------
        name: ``str``
            The name of the Dataprovider, for which the enum value is to be retrieved
            eg: "smartapi"

        Returns
        -------
        ``Dataprovider``
            The Dataprovider enum value

        Raises
        ------
        ``ValueError``
            If the given name is not a valid DataProvider name
        """
        if name in [item.value for item in DataProvider]:
            return DataProvider[name]

        raise ValueError(f"Invalid data provider name: {name}")


def get_credentials(data_provider: DataProvider) -> Credentials:
    """
    Get the credentials for the given credentials name.

    Parameters
    ----------
    data_provider: ``Credentials``
        The data provider for which the credentials are to be retrieved.
        This should be a DataProvider enum value

    Returns
    -------
    ``Credentials``
        The credentials object for the given data provider

    """
    credentials = Credentials.by_name(data_provider.value)

    return credentials.get_credentials()
