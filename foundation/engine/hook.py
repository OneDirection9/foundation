# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

__all__ = ['BaseHook']


class BaseHook(object):
    """Base class for hooks that can be registered with :class:`Runner`.

    Each hook can implement 4 methods and take the instance of :class:`Runner` as input. The way
    they are called is demonstrated in the following snippet:

    .. code-block:: python

        hook.before_train(runner)
        for iter in range(start_iter, max_iter):
            hook.before_step(runner)
            trainer.run_step(runner)
            hook.after_step(runner)
        hook.after_train(runner)

    Notes:
        A hook that does something in :meth:`before_step` can often be implemented equivalently in
        :meth:`after_step`. If the hook takes non-trivial time, it is strongly recommended to
        implement the hook in :meth:`after_step` instead of :meth:`before_step`. The convention is
        that :meth:`before_step` should only take negligible time.

        Following this convention will allow hooks that do care about the difference between
        :meth:`before_step` and :meth:`after_step` (e.g., timer) to function properly.
    """

    def before_train(self, runner) -> None:
        """Called before the first iteration."""
        pass

    def after_train(self, runner) -> None:
        """Called after the last iteration."""
        pass

    def before_step(self, runner) -> None:
        """Called before each iteration."""
        pass

    def after_step(self, runner) -> None:
        """Called after each iteration."""
        pass
