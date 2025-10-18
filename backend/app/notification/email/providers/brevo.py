"""
BrevoEmailProvider is used to send email notifications using the Brevo API.
"""

from pathlib import Path

import brevo_python
from brevo_python.rest import ApiException
from fastapi import HTTPException, status

from app.core.config import BrevoSettings
from app.core.mixins import FactoryMixin
from app.data_layer.data_models.notification_provider_model import (
    EmailNotificationPayload,
)
from app.notification.email.email_provider import EmailProvider
from app.notification.provider import NotificationProvider
from app.utils.common.logger import get_logger

logger = get_logger(Path(__file__).name)


@NotificationProvider.register("brevo")
class BrevoEmailProvider(EmailProvider, FactoryMixin[BrevoSettings]):
    """
    This class used to send verification code to the user's email using the Brevo API.

    Attributes:
    ----------
    api_key: ``str``
        The API key for authenticating with the Brevo API
    sender_name: ``str``
        The name of the sender
    sender_email: ``str``
        The email of the sender
    """

    def __init__(self, api_key: str, sender_name: str, sender_email: str) -> None:

        self.configuration = brevo_python.Configuration()
        self.configuration.api_key["api-key"] = api_key
        self.sender_name = sender_name
        self.sender_email = sender_email

    def send_notification(self, payload: EmailNotificationPayload) -> None:
        """
        This method is used to send the verification code to the user's email.

        Parameters:
        ----------
        code: ``str``
            The verification code that will be sent to the user's email
        recipient_email: ``str``
            The email address to which the verification code will be sent
        recipient_name: ``str``
            The name of the receiver
        """
        subject = "Verify your email"
        sender = {"name": self.sender_name, "email": self.sender_email}
        to = [{"email": payload.email_address, "name": payload.recipient_name}]
        html_content = (
            f"<p>Your verification code is: <strong>{payload.message}</strong></p>"
        )

        api_instance = brevo_python.TransactionalEmailsApi(
            brevo_python.ApiClient(self.configuration)
        )
        send_smtp_email = brevo_python.SendSmtpEmail(
            sender=sender, to=to, subject=subject, html_content=html_content
        )

        try:
            api_instance.send_transac_email(send_smtp_email)

        except ApiException as e:
            logger.error(
                "Exception when calling TransactionalEmailsApi->send_transac_email: %s\n",
                e,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send verification code. Please try again.",
            ) from e
