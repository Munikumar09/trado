from abc import ABC, abstractmethod

from omegaconf import DictConfig
from registrable import Registrable


class NotificationProvider(ABC, Registrable):
    """
    NotificationProvider is an abstract class that is used to send notifications to users.
    """

    @abstractmethod
    def send_notification(
        self, message: str, recipient_email_or_number: str, recipient_name: str
    ) -> None:
        """
        Send a notification to the user based on the recipient's email or phone number.

        Parameters:
        -----------
        message: ``str``
            The message to send to the user
        recipient_email_or_number: ``str``
            The email or phone number of the recipient
        recipient_name: ``str``
            The name of the recipient
        """
        raise NotImplementedError(
            "NotificationProvider is an abstract class and cannot be instantiated directly."
        )

    @classmethod
    @abstractmethod
    def from_cfg(cls, cfg: DictConfig) -> "NotificationProvider":
        """
        Initialize the NotificationProvider from the configuration.
        """
        raise NotImplementedError(
            "NotificationProvider is an abstract class and cannot be instantiated directly."
        )
