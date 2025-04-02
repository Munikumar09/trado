from abc import abstractmethod

from omegaconf import DictConfig

from app.notification.provider import NotificationProvider


@NotificationProvider.register("email_provider")
class EmailProvider(NotificationProvider):
    """
    This is base class for all the email providers. All the email providers
    should inherit this class and implement the `send_notification` method.
    This is mostly used to send the verification code to the user's email.
    """

    # pylint: disable=arguments-renamed
    @abstractmethod
    def send_notification(
        self, code: str, recipient_email: str, recipient_name: str
    ) -> None:
        raise NotImplementedError(
            "EmailProvider is an abstract class and cannot be instantiated directly."
        )

    @classmethod
    @abstractmethod
    def from_cfg(cls, cfg: DictConfig) -> "EmailProvider":
        raise NotImplementedError(
            "EmailProvider is an abstract class and cannot be instantiated directly."
        )
