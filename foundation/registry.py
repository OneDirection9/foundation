from __future__ import absolute_import, division, print_function

import copy
import functools
import logging
from typing import Any, Callable, Dict, List, Type, Union

import six

__all__ = ['Registry', 'build']

CONFLUENCE = '_registry'

logger = logging.getLogger(__name__)


class Registry(object):
    """A class that keeps a set of objects that can be selected by the name."""

    @classmethod
    def _register(cls, key: str, value: Any) -> None:
        # TODO: consider adding inspection of specific class
        if not isinstance(key, six.string_types) or not key:
            raise TypeError('Register keys must be non-empty string')

        if CONFLUENCE not in cls.__dict__:
            setattr(cls, CONFLUENCE, dict())

        if cls.contains(key):
            raise KeyError('{} is already registered in {}'.format(key, cls.__name__))

        cls.__dict__[CONFLUENCE][key] = value

    @classmethod
    def register(cls, name_or_obj: Union[str, Callable]) -> Callable:
        """Registers a python object."""
        if callable(name_or_obj):
            cls._register(name_or_obj.__name__, name_or_obj)
            return name_or_obj

        def wrapper(obj: Callable) -> Callable:
            cls._register(name_or_obj, obj)
            return obj

        return wrapper

    @classmethod
    def register_partial(cls, name: str, *args: Any, **kwargs: Any) -> Callable:
        """Registers a callable object presetting partial arguments."""

        def wrapper(obj: Callable) -> Callable:
            cls._register(name, functools.partial(obj, *args, **kwargs))
            return obj

        return wrapper

    @classmethod
    def get(cls, key: str) -> Callable:
        """Returns the registered python object."""
        if not cls.contains(key):
            raise KeyError("'{}' is not registered, available keys are: {}".format(key, cls.list()))
        return cls.__dict__[CONFLUENCE][key]

    @classmethod
    def list(cls) -> List[str]:
        """Lists all registered keys."""
        if CONFLUENCE not in cls.__dict__:
            return []
        return list(cls.__dict__[CONFLUENCE].keys())

    @classmethod
    def contains(cls, key: str) -> bool:
        """Returns True if the given key has been registered."""
        return CONFLUENCE in cls.__dict__ and key in cls.__dict__[CONFLUENCE]


def build(registry: Type[Registry], kwargs: Dict[str, Any]) -> Any:
    """Builds python object from registry.

    Args:
        registry: :class:`Registry` or subclass.
        kwargs: Key word arguments, it should at least contain the key `name`.

    Returns:
        The constructed object.
    """
    if not isinstance(kwargs, dict):
        raise TypeError('kwargs should be dictionary. Got {}'.format(type(kwargs)))

    _kwargs = copy.deepcopy(kwargs)

    try:
        obj_name = _kwargs.pop('name')
        obj = registry.get(obj_name)
        return obj(**_kwargs)
    except Exception as e:
        logger.error(
            "{}: Failed to build object from '{}' with kwargs={}".format(
                e.__class__.__name__, registry.__name__, kwargs
            )
        )
        raise e
