from typing import Self

from omegaconf import DictConfig
from pydantic_settings import BaseSettings
from registrable import Registrable


class FactoryMixin[BS: BaseSettings]:
    """
    Unified way to build from .env settings or Hydra config.
    """

    @classmethod
    def build(cls, settings: BS) -> Self:
        """
        Build an instance of the class from the provided settings.

        Parameters
        ----------
        settings: ``BaseSettings``
            The settings instance to build from

        Returns
        -------
        ``Self``
            An instance of the class
        """
        if not isinstance(settings, BaseSettings):
            raise TypeError(
                f"{cls.__name__}.build() requires a BaseSettings instance, "
                f"got {type(settings)}"
            )
        return cls(**settings.model_dump(exclude={"default_implementation"}))

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Self:
        """
        Create an instance of the class from a Hydra config.

        Parameters
        ----------
        cfg: ``DictConfig``
            The Hydra config to build the instance from

        Returns
        -------
        ``Self``
            An instance of the class
        """
        print(f"cfg: {cfg}")
        setting_type = cfg.get("setting_type")
        if setting_type is None:
            raise ValueError("setting_type is required")

        settings_cls = Registrable.by_name(setting_type)
        if not issubclass(settings_cls, BaseSettings):
            raise TypeError(f"{setting_type} must be a subclass of BaseSettings")

        # Instantiate the settings (loads from env automatically)
        settings_instance = settings_cls()

        return cls.build(settings_instance)
