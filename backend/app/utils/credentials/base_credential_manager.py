from abc import ABC, abstractmethod

from registrable import Registrable

from app.data_layer.data_models.credential_model import (
    CredentialInput,
    CredentialOutput,
)


class CredentialManager[CI: CredentialInput, CO: CredentialOutput](Registrable, ABC):
    """
    Base class for credentials used in the market data API. Subclasses of
    `CredentialManager` should implement the `generate_credentials` method to provide
    the actual credentials.
    """

    @classmethod
    @abstractmethod
    def generate_credentials(cls, credential_input: CI) -> CO:
        """
        Generate the credentials for the market data API.

        This method should be implemented by subclasses to generate the actual
        credentials required for authentication.

        Raises
        ------
        ``NotImplementedError``
            If the method is not implemented in the subclass
        """
        raise NotImplementedError("generate_credentials method not implemented")
