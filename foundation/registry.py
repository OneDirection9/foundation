from __future__ import absolute_import, division, print_function

import copy
import functools
import inspect
import logging
from typing import Any, Callable, Dict, List, Union

import six
from tabulate import tabulate

logger = logging.getLogger(__name__)

__all__ = ["Registry", "build"]


class Registry(object):
    """
    A class that keeps a set of objects that can be selected by the name.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block: python

        @BACKBONE_REGISTRY.register
        class MyBackbone():
            ...

    Or:

    .. code-block: python

        @BACKBONE_REGISTRY.register('my_backbone')
        class MyBackbone():
            ...

    Or:

    .. code-block: python

        BACKBONE_REGISTRY.register('my_backbone')(MyBackbone)
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name: The name of this registry.
        """
        self._name = name
        self._registry: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def registry(self) -> Dict[str, Any]:
        return self._registry

    def _register(self, name: str, obj: Any) -> None:
        # TODO: consider adding inspection of specific class
        if not isinstance(name, six.string_types) or not name:
            raise TypeError("Registered name must be non-empty string")

        if name in self._registry:
            raise KeyError(
                "An object named '{}' was already registered in '{}' registry!".format(
                    name, self._name
                )
            )

        self._registry[name] = obj

    def register(self, name_or_obj: Union[str, Callable[..., Any]]) -> Callable[..., Any]:
        """
        Register a python object.
        """
        if callable(name_or_obj):
            self._register(name_or_obj.__name__, name_or_obj)
            return name_or_obj

        def wrapper(obj: Any) -> Any:
            self._register(name_or_obj, obj)
            return obj

        return wrapper

    def register_partial(self, name: str, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        """
        Register a callable object presetting partial arguments.
        """

        def wrapper(obj: Callable[..., Any]) -> Callable[..., Any]:
            partial_obj = functools.partial(obj, *args, **kwargs)
            partial_obj = functools.update_wrapper(partial_obj, obj)
            self._register(name, partial_obj)
            return partial_obj

        return wrapper

    def get(self, name: str) -> Any:
        """
        Return the registered python object.
        """
        if name not in self._registry:
            raise KeyError(
                "'{}' is not registered, available keys are: {}".format(name, self.list())
            )
        return self._registry[name]

    def list(self) -> List[str]:
        """
        List all registered keys.
        """
        return list(self._registry.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __repr__(self) -> str:
        table_headers = ["Names", "Objects"]
        table = tabulate(self._registry.items(), headers=table_headers, tablefmt="fancy_grid")
        return "Registry of {}:\n".format(self._name) + table

    __str__ = __repr__


def build(registry: Registry, cfg: Dict[str, Any]) -> Any:
    """
    Build python object from registry.

    Args:
        registry: The registry to search the object from.
        cfg: Config dictionary, it should at least contain the key "name".

    Returns:
        The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg should be dictionary. Got {}".format(type(cfg)))

    _cfg = copy.deepcopy(cfg)

    try:
        obj_name = _cfg.pop("name")
        obj = registry.get(obj_name)
        # TODO: support other type
        assert inspect.isclass(obj), "build only support build instance from class"
        return obj(**_cfg)
    except Exception as e:
        logger.error("Failed to build object from '{}' with cfg={}".format(registry.name, cfg))
        raise e
