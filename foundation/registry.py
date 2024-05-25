import functools
import inspect
import logging
import types
from collections.abc import Callable, Iterator
from typing import Any, TypeVar, cast

from tabulate import tabulate

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


logger = logging.getLogger(__name__)

NAME = "_registry"


def get_fullyqualified_name(  # noqa
    obj: Any, include_partials: bool = True, include_instance: bool = True
) -> str:
    """Return a str describing as best as possible the given object's name (i.e. not its content).

    Args:
        obj: The object whose fully qualified name should be returned
        include_partials (bool): If True (default), objects that are functools.partial instances,
          will be reported as such in their name (and include a representation of their args and kwargs).
          If False, functools.partial instances will be silently unwrapped.
        include_instance (bool): If True (default), a suffix .<instance> will
          be added to the name of the class for class instances.
    """
    if hasattr(obj, "fullyqualified_name"):
        candidate_name = obj.fullyqualified_name
        # When `obj` is a class (not a class instance) which has a
        # fullyqualified_name property, then candidate_name will still exist
        # but be a `property` designed to be called on an instance, and not a
        # string-like thing. Skip those.
        if not isinstance(candidate_name, property):
            return cast(str, candidate_name)

    # Explicitly handle functools.partial by taking the fully qualified name of the
    # wrapped object, and surround it by a description of the partial
    if isinstance(obj, functools.partial):
        if include_partials:
            args = list(map(str, obj.args))

            def _repr(x: Any) -> str:
                return f"'{x}'" if isinstance(x, str) else f"{x}"

            args += [f"{key}={_repr(val)}" for key, val in obj.keywords.items()]
            return "functools.partial({}{})".format(
                get_fullyqualified_name(obj.func),
                ", ".join([""] + args),  # noqa
            )
        else:
            return get_fullyqualified_name(obj.func)

    # Handle methods of non-module objects (e.g. class instance)
    if hasattr(obj, "__self__") and hasattr(obj.__self__, "__module__"):
        return get_fullyqualified_name(obj.__self__) + "." + cast(str, obj.__name__)
    # Detect regular class instance (but ignore functions, which are instances of class "function")
    if not (isinstance(obj, type | types.BuiltinFunctionType | types.FunctionType)):
        ext = ".<instance>" if include_instance else ""
        return get_fullyqualified_name(type(obj)) + ext

    if hasattr(obj, "__module__"):
        name = obj.__module__
    else:
        # Modules don't have a __module__ attribute
        return obj.__name__
    if hasattr(obj, "__qualname__"):
        return name + "." + obj.__qualname__
    return name


class Registry:
    """Represent a set of named objects that can be selected by calling the class.

    Registries are intended to be used in conjunction with argparse in order to
    allow selection of typed objects based on a name.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        class BackboneRegistry(Registry): ...

    To register an object:

    .. code-block: python

        @BackboneRegistry.register
        class MyBackbone():
            ...

    Or:

    .. code-block: python

        @BackboneRegistry.register('my_backbone')
        class MyBackbone():
            ...

    Or:

    .. code-block: python

        BackboneRegistry.register('my_backbone')(MyBackbone)

    To register an object instance:

    .. code-block: python

        @BackboneRegistry.register_instance('my_backbone', a='a', b='b')
        class MyBackbone():
            ...
    """

    @classmethod
    def _set_registry(cls, key: Any, value: Any, force: bool = False) -> None:
        if not key or not isinstance(key, str):
            raise TypeError(f"Registry keys must be non-empty strings. Got {key}")

        if NAME not in cls.__dict__:
            setattr(cls, NAME, {})

        if key in cls.__dict__[NAME]:
            previous_value_name = get_fullyqualified_name(cls.__dict__[NAME][key])
            new_value_name = get_fullyqualified_name(value)
            # When importlib.reload is used, we might re-register objects
            # We compare the fully qualified name of both objects because in case of instances,
            # they might not be equal. If both fully-qualified names are equal, we'll register
            # the new value anyway.
            if previous_value_name != new_value_name:
                if force:
                    logging.warning(
                        f"Forcefully replacing '{previous_value_name}' by '{new_value_name}' "
                        f"in registry {cls.__name__} for key '{key}'"
                    )
                else:
                    raise KeyError(
                        f"Name '{key}' already registered in registry "
                        f"{cls.__name__}: {previous_value_name}"
                    )
            else:
                logging.warning(
                    f"Re-registering '{previous_value_name}' in registry "
                    f"{cls.__name__} for key '{key}'"
                )
                return
        cls.__dict__[NAME][key] = value

    @classmethod
    def _register(
        cls, name_or_obj: str | F, force: bool = False, skip: bool = False
    ) -> Callable[[T], T] | F:
        """Actual registration logic.

        Separated from `register` and `register_instance` to ease API specific behaviour in child
        classes (for instance disabling `register`).
        """
        if callable(name_or_obj):
            if not skip:
                cls._set_registry(name_or_obj.__name__, name_or_obj, force=force)
            return name_or_obj

        def wrapper(obj: T) -> T:
            if not skip:
                cls._set_registry(name_or_obj, obj, force=force)
            return obj

        return wrapper

    @classmethod
    def register(
        cls, name_or_obj: str | F, force: bool = False, skip: bool = False
    ) -> Callable[[T], T] | F:
        """Register an object under a given name.

        If a callable is given, this callable will be registered under its `.__name__` attribute.
        Else the given argument is used as a key in the registry, and a decorator function
        (to be called with the actual object to register) will be returned.

        Args:
            name_or_obj: A callable to register under its own name, or a name to use as a key
                in the registry.
            force (bool): Force the registration of this object, in case it already exists
                about force replace
            skip (bool): If True, skips registering this object, useful for having environment based
                registration like running on bolt, using macos, or running as part of a test.

        Returns:
            The callable itself or a decorator to register the callable with the given name.
        """
        return cls._register(name_or_obj, force=force, skip=skip)

    @classmethod
    def register_instance(
        cls, name: str, *args: Any, force: bool = False, skip: bool = False, **kwargs: Any
    ) -> Callable[[F], F]:
        """Register an instance of a class under a given name in the registry.

        Args:
            name: The name to give to the registered instance in the registry.
            force (bool): Force the registration of this object, in case it already exists.
            skip (bool): If True, skips registering this object, useful for having environment based registration
                like running on bolt, using macos, or running as part of a test.
            *args: Positional arguments to pass to the instance initialization.
            **kwargs: Keyword arguments to pass to the instance initialization.
        """

        def do_register(item_cls: F) -> F:
            # TODO: Check the cls signature doesn't have parameter named `force` or `skip`?
            cls._register(name, force=force, skip=skip)(item_cls(*args, **kwargs))
            return item_cls

        return do_register

    @classmethod
    def _registries(cls) -> Iterator[Any]:
        for base in inspect.getmro(cls):
            try:
                yield base.__dict__[NAME]
            except KeyError:  # noqa: PERF203
                continue

    @classmethod
    def items(cls) -> Iterator[tuple[str, Any]]:
        seen = set()
        for registry in cls._registries():
            for key, value in registry.items():
                if key in seen:
                    continue
                seen.add(key)
                yield key, value

    @classmethod
    def keys(cls) -> Iterator[str]:
        for key, _ in cls.items():  # noqa: PERF102
            yield key

    @classmethod
    def values(cls) -> Iterator[Any]:
        for _, value in cls.items():  # noqa: PERF102
            yield value

    @classmethod
    def reset(cls) -> None:
        if NAME in cls.__dict__:
            registry = cls.__dict__[NAME]
            assert isinstance(registry, dict)
            registry.clear()

    @classmethod
    def lazy_imports(cls) -> None:
        """Use this method to lazily import modules populating this registry.

        Can be used to solve circular imports issues while maintaining modular registry definitions.
        """
        ...

    @classmethod
    def _register_lazy_imports(cls) -> None:
        if not getattr(cls, "_lazy_import_loaded", False):
            cls.lazy_imports()
            cls._lazy_import_loaded = True  # type: ignore

    @classmethod
    def _lookup_registry(cls, key: Any) -> Any:
        # Falsy-keys (such as `None` or `''`) can't be used in a registry and
        # always result in a `None` value.
        if not key:
            return None

        cls._register_lazy_imports()

        found = []
        for registry in cls._registries():
            try:
                found.append(registry[key])
            except KeyError:  # noqa: PERF203
                continue

        if len(found) == 1:
            return found[0]
        elif len(found) > 1:
            raise KeyError(
                f"Key '{key}' returned {len(found)} matches, "
                f"which is unexpected: {', '.join(list(map(str, found)))}"
            )
        raise KeyError(f"'{key}' is not registered, available keys are: {list(cls.keys())}")

    def __contains__(self, key: str) -> bool:
        return key in self.keys()

    def __call__(self, key: Any) -> Any:
        return self._lookup_registry(key)

    def __getitem__(self, key: Any) -> Any:
        return self._lookup_registry(key)

    def __repr__(self) -> str:
        table_headers = ["Names", "Objects"]
        table = tabulate(self.items(), headers=table_headers, tablefmt="fancy_grid")
        return f"{self.__class__.__name__}:\n{table}"

    __str__ = __repr__

    def prefix_search(self, key: str) -> list[Any]:
        matches = [k for k in self.keys() if k.startswith(key)]
        return matches
