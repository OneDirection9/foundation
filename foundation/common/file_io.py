from __future__ import absolute_import, division, print_function

import os.path as osp
from typing import List, Optional

__all__ = ["find_vcs_root"]


def find_vcs_root(path: str, markers: List[str] = (".git",)) -> Optional[str]:
    """
    Find the root directory (including itself) of specified markers.

    Args:
        path: Path of directory or file.
        markers: List of file or directory names. Default: ('.git',)

    Returns:
        The directory contained one of the markers or None if not found.
    """
    if osp.isfile(path):
        path = osp.dirname(path)

    prev, cur = None, osp.abspath(osp.expanduser(path))
    while cur != prev:
        if any(osp.exists(osp.join(cur, marker)) for marker in markers):
            return cur
        prev, cur = cur, osp.split(cur)[0]
    return None
