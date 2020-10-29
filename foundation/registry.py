from __future__ import absolute_import, division, print_function

import functools
from typing import Any, Callable, Dict, List, Tuple, Union

import six
from tabulate import tabulate

__all__ = ['Registry']


class Registry(object):
    """A class that keeps a set of objects that can be selected by the name."""

    def __init__(self, name: str) -> None:
        """
        Args:
            name: The name of this registry.
        """
        self._name = name
        self._registry: Dict[str, Any] = {}

    def _register(self, name: str, obj: Any) -> None:
        # TODO: consider adding inspection of specific class
        if not isinstance(name, six.string_types) or not name:
            raise TypeError('Registered name must be non-empty string')

        if name in self._registry:
            raise KeyError(
                "An object named '{}' was already registered in '{}' registry!".format(
                    name, self._name
                )
            )

        self._registry[name] = obj

    def register(self, name_or_obj: Union[str, Callable[..., Any]]) -> Callable[..., Any]:
        """Registers a python object."""
        if callable(name_or_obj):
            self._register(name_or_obj.__name__, name_or_obj)
            return name_or_obj

        def wrapper(obj: Any) -> Any:
            self._register(name_or_obj, obj)
            return obj

        return wrapper

    def register_partial(self, name: str, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        """Registers a callable object presetting partial arguments."""

        def wrapper(obj: Callable[..., Any]) -> Callable[..., Any]:
            self._register(name, functools.partial(obj, *args, **kwargs))
            return obj

        return wrapper

    def get(self, name: str) -> Any:
        """Returns the registered python object."""
        if name not in self._registry:
            raise KeyError(
                "'{}' is not registered, available keys are: {}".format(name, self.keys())
            )
        return self._registry[name]

    def list(self) -> List[str]:
        """Alias of keys()"""
        return self.keys()

    def keys(self) -> List[str]:
        """Lists all registered keys."""
        return list(self._registry.keys())

    def values(self) -> List[Any]:
        """Lists all registered values."""
        return list(self._registry.values())

    def items(self) -> List[Tuple[str, Any]]:
        return [(k, v) for k, v in self._registry.items()]

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __repr__(self) -> str:
        table_headers = ['Names', 'Objects']
        table = tabulate(self._registry.items(), headers=table_headers, tablefmt='fancy_grid')
        return 'Registry of {}:\n'.format(self._name) + table

    __str__ = __repr__
