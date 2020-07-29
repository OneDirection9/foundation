# Copyright (c) Open-MMLab. All rights reserved.
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import json
import pickle
from abc import abstractmethod
from typing import IO, Any

import yaml

from foundation.registry import Registry

try:
    from yaml import CDumper as Dumper, CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader

__all__ = [
    'HandlerRegistry',
    'BaseFileHandler',
    'JsonHandler',
    'PickleHandler',
    'YamlHandler',
]


class HandlerRegistry(Registry):
    """Registry of file handlers."""
    pass


class BaseFileHandler(object):
    """Basic class for file handler."""

    @abstractmethod
    def load_from_fileobj(self, file: IO, **kwargs: Any) -> Any:
        pass

    def load_from_filepath(self, filepath: str, mode: str = 'r', **kwargs: Any) -> Any:
        with open(filepath, mode) as f:
            return self.load_from_fileobj(f, **kwargs)

    @abstractmethod
    def dump_to_fileobj(self, obj: Any, file: IO, **kwargs: Any) -> None:
        pass

    def dump_to_filepath(self, obj: Any, filepath: str, mode: str = 'w', **kwargs: Any) -> None:
        with open(filepath, mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)

    @abstractmethod
    def dump_to_str(self, obj: Any, **kwargs: Any) -> None:
        pass


@HandlerRegistry.register('json')
class JsonHandler(BaseFileHandler):
    """Json file handler."""

    def load_from_fileobj(self, file: IO, **kwargs: Any) -> Any:
        return json.load(file, **kwargs)

    def dump_to_fileobj(self, obj: Any, file: IO, **kwargs: Any) -> None:
        json.dump(obj, file, **kwargs)

    def dump_to_str(self, obj: Any, **kwargs: Any) -> str:
        return json.dumps(obj, **kwargs)


@HandlerRegistry.register('pickle')
class PickleHandler(BaseFileHandler):
    """Pickle file handler."""

    def load_from_fileobj(self, file: IO, **kwargs: Any) -> Any:
        return pickle.load(file, **kwargs)

    def load_from_filepath(self, filepath: str, **kwargs: Any) -> Any:
        return super().load_from_filepath(filepath, mode='rb', **kwargs)

    def dump_to_fileobj(self, obj: Any, file: IO, **kwargs: Any) -> None:
        kwargs.setdefault('protocol', 2)
        pickle.dump(obj, file, **kwargs)

    def dump_to_filepath(self, obj: Any, filepath: str, **kwargs: Any) -> None:
        super().dump_to_filepath(obj, filepath, mode='wb', **kwargs)

    def dump_to_str(self, obj: Any, **kwargs: Any) -> str:
        kwargs.setdefault('protocol', 2)
        encoding = kwargs.pop('encoding', 'ASCII')
        return pickle.dumps(obj, **kwargs).decode(encoding)


# Register an alias of PickleHandler
HandlerRegistry.register('pkl')(PickleHandler)


@HandlerRegistry.register('yaml')
class YamlHandler(BaseFileHandler):
    """Yaml file handler."""

    def load_from_fileobj(self, file: IO, **kwargs: Any) -> Any:
        kwargs.setdefault('Loader', Loader)
        return yaml.load(file, **kwargs)

    def dump_to_fileobj(self, obj: Any, file: IO, **kwargs: Any) -> None:
        kwargs.setdefault('Dumper', Dumper)
        yaml.dump(obj, file, **kwargs)

    def dump_to_str(self, obj: Any, **kwargs: Any) -> str:
        kwargs.setdefault('Dumper', Dumper)
        return yaml.dump(obj, **kwargs)


# Register an alias of YamlHandler
HandlerRegistry.register('yml')(YamlHandler)
