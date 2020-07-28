# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

from time import perf_counter
from typing import Optional

__all__ = ['Timer']


class Timer(object):
    """A timer which computes the time elapsed since the start/reset of the timer."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Resets the timer."""
        self._start = perf_counter()
        self._paused: Optional[float] = None
        self._total_paused = 0
        self._count_start = 1

    def pause(self) -> None:
        """Pauses the timer."""
        if self._paused is not None:
            raise ValueError('Trying to pause a Timer that is already paused!')
        self._paused = perf_counter()

    def is_paused(self) -> bool:
        """
        Returns:
            Whether the timer is currently paused.
        """
        return self._paused is not None

    def resume(self) -> None:
        """Resumes the timer."""
        if self._paused is None:
            raise ValueError('Trying to resume a Timer that is not paused!')
        self._total_paused += perf_counter() - self._paused  # pyre-ignore
        self._paused = None
        self._count_start += 1

    def seconds(self) -> float:
        """
        Returns:
            The total number of seconds since the start/reset of the timer, excluding the time when
            the timer is paused.
        """
        if self._paused is not None:
            end_time: float = self._paused  # type: ignore
        else:
            end_time = perf_counter()
        return end_time - self._start - self._total_paused

    def avg_seconds(self) -> float:
        """
        Returns:
            The average number of seconds between every start/reset and pause.
        """
        return self.seconds() / self._count_start
