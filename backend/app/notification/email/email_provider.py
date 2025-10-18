from abc import abstractmethod

from app.data_layer.data_models.notification_provider_model import (
    EmailNotificationPayload,
)
from app.notification.provider import NotificationProvider


@NotificationProvider.register("email_provider")
class EmailProvider(NotificationProvider[EmailNotificationPayload]):
    """
    This is base class for all the email providers. All the email providers
    should inherit this class and implement the `send_notification` method.
    This is mostly used to send the verification code to the user's email.
    """

    @abstractmethod
    def send_notification(self, payload: EmailNotificationPayload) -> None:
        """
        Send a notification to the user based on the recipient's email.

        Parameters
        ----------
        payload: ``EmailNotificationPayload``
            The notification payload containing the recipient's email and message
        """
        raise NotImplementedError(
            "EmailProvider is an abstract class and cannot be instantiated directly."
        )
