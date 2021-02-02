# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# Modified by: Zhipeng Han
"""
An awesome colormap for really neat visualizations.
Copied from Detectron, and removed gray colors.
"""
from __future__ import absolute_import, division, print_function

import numpy as np

__all__ = ["colormap", "random_color"]

# RGB:
# fmt: off
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3)
# fmt: on


def colormap(rgb: bool = False, maximum: int = 255) -> np.ndarray:
    """
    Args:
        rgb (bool): Whether to return RGB colors or BGR colors.
        maximum (int): Either 1 or 255.

    Returns:
        np.ndarray: A float32 array of Nx3 colors, in range [0, 1] or [0, 255].
    """
    if maximum not in (1, 255):
        raise ValueError("maximum should be 1 or 255. Got {}".format(maximum))

    c = _COLORS * maximum
    if not rgb:
        c = c[:, ::-1]
    return c


def random_color(rgb: bool = False, maximum: int = 255) -> np.ndarray:
    """
    Args:
        rgb (bool): Whether to return RGB colors or BGR colors.
        maximum (int): Either 1 or 255.

    Returns:
        np.ndarray: A float32 array of 3 numbers, in range [0, 1] or [0, 255].
    """
    if maximum not in (1, 255):
        raise ValueError("maximum should be 1 or 255. Got {}".format(maximum))

    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


if __name__ == "__main__":
    import cv2

    H, W, size = 10, 10, 100
    canvas = np.random.rand(H * size, W * size, 3).astype("float32")
    for h in range(H):
        for w in range(W):
            idx = h * W + w
            if idx >= len(_COLORS):
                break
            canvas[h * size : (h + 1) * size, w * size : (w + 1) * size] = _COLORS[idx]
    cv2.imshow("colors", canvas)
    cv2.waitKey(0)
