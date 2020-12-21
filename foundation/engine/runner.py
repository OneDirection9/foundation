# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import logging
from typing import NoReturn, Sequence

from .hook import BaseHook

__all__ = ["BaseRunner"]

logger = logging.getLogger(__name__)


class BaseRunner(object):
    """
    Base class for iterative runner with hooks.

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.

    Attributes:
        _hooks (list): List of :class:`BaseHook` instances.
        _iter (int): The current iteration.
        _start_iter (int): The iteration to start with. By convention the minimum value is 0.
        _max_iter (int): The iteration to end training.
    """

    def __init__(self) -> None:
        self._hooks = []
        self._iter = 0
        self._start_iter = 0
        self._max_iter = 0

    @property
    def iter(self) -> int:
        return self._iter

    @property
    def start_iter(self) -> int:
        return self._start_iter

    @property
    def max_iter(self) -> int:
        return self._max_iter

    def register_hooks(self, hooks: Sequence[BaseHook]) -> None:
        """
        Register hooks to the runner.

        The hooks are executed in the order they are registered.

        Args:
            hooks: List of hooks to be registered.
        """
        for hook in hooks:
            if not isinstance(hook, BaseHook):
                raise TypeError("hook should be BaseHook. Got {}".format(type(hook)))
            self._hooks.append(hook)

    def train(self, start_iter: int, max_iter: int) -> None:
        """
        Args:
            start_iter, max_iter: See docs above.
        """
        logger.info("Starting training from iteration {}".format(start_iter))

        self._iter = self._start_iter = start_iter
        self._max_iter = max_iter

        try:
            self.before_train()
            for self._iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
        except Exception as e:
            logger.exception("Exception during training:")
            raise e
        finally:
            self.after_train()

    def before_train(self) -> None:
        for h in self._hooks:
            h.before_train(self)

    def after_train(self) -> None:
        for h in self._hooks:
            h.after_train(self)

    def before_step(self) -> None:
        for h in self._hooks:
            h.before_step(self)

    def after_step(self) -> None:
        for h in self._hooks:
            h.after_step(self)

    def run_step(self) -> NoReturn:
        raise NotImplementedError
