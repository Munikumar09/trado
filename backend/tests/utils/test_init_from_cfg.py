from typing import Any

import pytest
from hydra.errors import InstantiationException
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError
from registrable import Registrable
from registrable.exceptions import RegistrationError

from app.utils.common import init_from_cfg, pop_name


############################# FIXTURES #############################
@Registrable.register("base_class")
class BaseClass(Registrable):
    """
    The base class for testing the init_from_cfg function.
    """

    def __init__(self, x: Any, y: Any):
        self.x = x
        self.y = y

    @classmethod
    def from_cfg(cls, cfg: DictConfig):
        """
        Initialize the class instance from a configuration object.
        """
        return cls(cfg.x, cfg.y)


@BaseClass.register("child_class1")
class ChildClass(BaseClass):
    """
    The child class for testing the init_from_cfg function.
    """

    def __init__(self, x: Any, y: Any, z: Any, *args: Any, **kwargs: Any):
        super().__init__(x, y)
        self.z = z
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def from_cfg(cls, cfg: DictConfig, *args: Any, **kwargs: Any):
        return cls(cfg.x, cfg.y, cfg.z, *args, **kwargs)


class SampleClass:
    """
    Sample class for testing the init_from_cfg function with additional args and kwargs
    passed directly to the class constructor
    """

    def __init__(self, x, y, *args, **kwargs):
        self.x = x
        self.y = y
        self.args = args
        self.kwargs = kwargs


############################# TESTS #############################


# Test: 1
@pytest.mark.parametrize(
    "cfg, expected",
    [
        (
            DictConfig({"name": "my_class", "arg1": "value1"}),
            DictConfig({"arg1": "value1"}),
        ),
        (DictConfig({"arg1": "value1"}), DictConfig({"arg1": "value1"})),
    ],
)
def test_removes_name(cfg: DictConfig, expected: DictConfig):
    """
    Test that the pop_name function correctly removes the 'name' key from the configuration.
    """
    new_cfg = pop_name(cfg)

    assert "name" not in new_cfg
    assert new_cfg == expected


# Test: 2
def test_init_from_cfg_with_target():
    """
    Test that init_from_cfg correctly initializes a class instance when _target_ is provided.
    """
    cfg = OmegaConf.create(
        {"_target_": "tests.utils.test_init_from_cfg.BaseClass", "x": 2, "y": 3}
    )
    instance = init_from_cfg(cfg)

    assert isinstance(instance, BaseClass)
    assert instance.x == 2
    assert instance.y == 3


# Test: 3
def test_init_from_cfg_with_target_and_missing_args():
    """
    Test that init_from_cfg raises an InstantiationException when required arguments are missing.
    """
    cfg = OmegaConf.create(
        {"_target_": "tests.utils.test_init_from_cfg.BaseClass", "x": 2}
    )

    with pytest.raises(InstantiationException):
        init_from_cfg(cfg)


# Test: 4
def test_init_from_cfg_with_target_and_args():
    """
    Test that init_from_cfg correctly initializes a class instance when _target_ is provided
    with additional args and kwargs.
    """
    cfg = OmegaConf.create(
        {
            "_target_": "tests.utils.test_init_from_cfg.ChildClass",
            "x": 2,
            "y": 3,
            "z": 4,
        }
    )
    instance = init_from_cfg(cfg, None, 5, 6, kwarg1="value1")

    assert isinstance(instance, ChildClass)
    assert instance.x == 2
    assert instance.y == 3
    assert instance.z == 4
    assert instance.args == (5, 6)
    assert instance.kwargs == {"kwarg1": "value1"}


# Test: 5
def test_init_from_cfg_with_invalid_target():
    """
    Test that init_from_cfg raises an InstantiationException when an invalid _target_ is provided.
    """
    cfg = OmegaConf.create({"_target_": "invalid.module.Class", "x": 2, "y": 3})

    with pytest.raises(InstantiationException) as exec_info:
        init_from_cfg(cfg)

    assert "Error locating target 'invalid.module.Class" in exec_info.value.args[0]


# Test: 6
def test_init_from_cfg_with_none_target():
    """
    Test that init_from_cfg raises an InstantiationException when an invalid _target_ is provided.
    """
    cfg = OmegaConf.create({"_target_": None, "x": 2, "y": 3})

    with pytest.raises(InstantiationException):
        init_from_cfg(cfg)


# Test: 7
def test_init_from_cfg_with_name_and_base_class():
    """
    Test that init_from_cfg correctly initializes a class instance when name and base_class
    are provided.
    """
    cfg = OmegaConf.create({"name": "child_class1", "x": 2, "y": 3, "z": 4})
    instance = init_from_cfg(cfg, BaseClass)

    assert isinstance(instance, ChildClass)
    assert instance.x == 2
    assert instance.y == 3
    assert instance.z == 4


# Test: 8
def test_init_from_cfg_with_name_only():
    """
    Test that init_from_cfg correctly initializes a class instance when only name is provided.
    """
    cfg = OmegaConf.create({"name": "base_class", "x": 2, "y": 3})
    instance = init_from_cfg(cfg)

    assert isinstance(instance, BaseClass)
    assert instance.x == 2
    assert instance.y == 3


# Test: 9
def test_init_from_cfg_with_base_class_only():
    """
    Test that init_from_cfg correctly initializes a class instance when only base_class is
    provided.
    """
    cfg = OmegaConf.create({"x": 2, "y": 3})
    instance = init_from_cfg(cfg, BaseClass)

    assert isinstance(instance, BaseClass)
    assert instance.x == 2
    assert instance.y == 3


# Test: 10
def test_init_from_cfg_without_from_cfg_method():
    """
    Test that init_from_cfg correctly initializes a class instance when from_cfg method
    is not available.
    """

    cfg = OmegaConf.create({"a": 1, "b": 2})
    instance = init_from_cfg(cfg, SampleClass)

    assert isinstance(instance, SampleClass)
    assert instance.x == 1
    assert instance.y == 2


# Test: 11
def test_init_from_cfg_with_invalid_name():
    """
    Test that init_from_cfg raises a RegistrationError when an invalid name is provided.
    """
    cfg = OmegaConf.create({"name": "invalid_class", "x": 2, "y": 3})

    with pytest.raises(RegistrationError) as exec_info:
        init_from_cfg(cfg)

    assert exec_info.match(
        "`invalid_class` is not a registered name. Please provide a proper base_class "
        "under which the class is registered."
    )


# Test: 12
def test_init_from_cfg_with_no_name_and_base_class():
    """
    Test that init_from_cfg raises a ValueError when both name and base_class are None.
    """
    cfg = OmegaConf.create({"x": 2, "y": 3})

    with pytest.raises(ValueError, match="Both name and base_class cannot be None"):
        init_from_cfg(cfg)


# Test: 13
def test_init_from_cfg_with_additional_args_and_kwargs():
    """
    Test that init_from_cfg correctly passes additional args and kwargs to the class constructor.
    """
    cfg = OmegaConf.create({"name": "child_class1", "x": 2, "y": 3, "z": 4})
    instance = init_from_cfg(cfg, BaseClass, 5, 6, kwarg1="value1")

    assert isinstance(instance, ChildClass)
    assert instance.x == 2
    assert instance.y == 3
    assert instance.z == 4
    assert instance.args == (5, 6)
    assert instance.kwargs == {"kwarg1": "value1"}


# Test: 14
def test_init_from_cfg_with_missing_required_arg():
    """
    Test that init_from_cfg raises a ConfigAttributeError when a required argument is missing.
    """
    cfg = OmegaConf.create({"name": "base_class", "x": 2})

    with pytest.raises(ConfigAttributeError):
        init_from_cfg(cfg)


# Test: 15
def test_init_from_cfg_with_extra_cfg_args():
    """
    Test that init_from_cfg correctly handles extra configuration arguments.
    """
    cfg = OmegaConf.create({"name": "base_class", "x": 2, "y": 3, "extra": "ignored"})
    instance = init_from_cfg(cfg)

    assert isinstance(instance, BaseClass)
    assert instance.x == 2
    assert instance.y == 3
    assert not hasattr(instance, "extra")


# Test: 16
def test_init_from_cfg_with_nested_config():
    """
    Test that init_from_cfg correctly handles nested configuration.
    """
    cfg = OmegaConf.create(
        {"name": "child_class1", "x": {"nested": 2}, "y": [1, 2, 3], "z": 4}
    )
    instance = init_from_cfg(cfg, BaseClass)

    assert isinstance(instance, ChildClass)
    assert instance.x == {"nested": 2}
    assert instance.y == [1, 2, 3]
    assert instance.z == 4


# Test: 17
def test_init_with_args_and_kwargs():
    """
    Test that init_from_cfg correctly initializes a class instance by directly passing parameters,
    args and kwargs to the class constructor.
    """
    cfg = OmegaConf.create(
        {
            "x": 10,
            "y": 20,
        }
    )
    sample_class = init_from_cfg(cfg, SampleClass, 1, 2, 3, a=100, b=200)

    assert sample_class.x == 10
    assert sample_class.y == 20
    assert sample_class.args == (1, 2, 3)
    assert sample_class.kwargs == {"a": 100, "b": 200}
