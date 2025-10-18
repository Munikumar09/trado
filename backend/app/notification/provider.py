from abc import ABC, abstractmethod

from registrable import Registrable

from app.data_layer.data_models.notification_provider_model import NotificationPayload


class NotificationProvider[NP: NotificationPayload](ABC, Registrable):
    """
    NotificationProvider is an abstract class that is used to send notifications to users.
    """

    @abstractmethod
    def send_notification(self, payload: NP) -> None:
        """
        Send a notification to the user based on the recipient's email or phone number.

        Parameters
        ----------
        payload: ``NP``
            The notification payload containing the recipient's information and message
        """
        raise NotImplementedError(
            "NotificationProvider is an abstract class and cannot be instantiated directly."
        )
