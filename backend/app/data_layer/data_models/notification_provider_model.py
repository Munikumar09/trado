from dataclasses import dataclass


@dataclass
class NotificationPayload:
    """
    Base payload for sending notifications.

    Attributes
    ________
    message: ``str``
        The message body to send to the recipient.
    recipient_name: ``str``
        The recipient's display name.
    """

    message: str
    recipient_name: str


@dataclass
class EmailNotificationPayload(NotificationPayload):
    """
    Email-specific notification payload extending the base notification payload.

    Attributes
    ________
    subject: ``str``
        The subject line of the email.
    email_address: ``str``
        The destination email address.
    """

    subject: str
    email_address: str
