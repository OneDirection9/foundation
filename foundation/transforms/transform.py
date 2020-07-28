# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import inspect
from abc import ABCMeta, abstractmethod
from typing import Callable, List, Optional, Sequence, TypeVar, Union

import cv2
import numpy as np

__all__ = [
    'CV2_INTER_CODES',
    'is_numpy',
    'is_numpy_image',
    'is_numpy_coords',
    'Transform',
    'TransformList',
    'HFlipTransform',
    'VFlipTransform',
    'NoOpTransform',
    'ResizeTransform',
    'CropTransform',
    'BlendTransform',
]

CV2_INTER_CODES = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'area': cv2.INTER_AREA,
    'bicubic': cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4,
}


def is_numpy(obj: object) -> bool:
    """Image, coordinates, segmentation and boxes should be np.ndarray."""
    return isinstance(obj, np.ndarray)


def is_numpy_image(image: np.ndarray) -> bool:
    """The image should be np.ndarray of shape HxWxC or HxW."""
    return image.ndim in {2, 3}


def is_numpy_coords(coords: np.ndarray) -> bool:
    """The coordinates should be np.ndarray of shape Nx2."""
    return coords.ndim == 2 and coords.shape[1] == 2


class Transform(object, metaclass=ABCMeta):
    """Base class for deterministic transformations for image and other data structures.

    Base class for implementations of **deterministic** transformations for image and other data
    structures. "Deterministic" requires that the output of all methods of this class are
    deterministic w.r.t their input arguments. Note that this is different from (random) data
    augmentations. To perform data augmentations in training, there should be a higher-level policy
    that generates these transform ops. Each transform op may handle several data types, e.g.:
    image, coordinates, segmentation, bounding boxes, with its ``apply_*`` methods. Some of them
    have a default implementation, but can be overwritten if the default isn't appropriate. See
    documentation of each pre-defined ``apply_*`` methods for details. Note that The implementation
    of these method may choose to modify its input data in-place for efficient transformation.

    The class can be extended to support arbitrary new data types with its :meth:`register_type`
    method.
    """

    @abstractmethod
    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """Applies the transform on the image.

        Args:
            image: Array of shape HxWxC or HxW. The array can be of type uint8 in range [0, 255], or
                floating point in range [0, 1] or [0, 255].

        Returns:
            Image after apply the transformation.
        """
        pass

    @abstractmethod
    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """Applies the transform on coordinates.

        Args:
            coords: Floating point array of shape Nx2 of (x, y) format in absolute coordinates.

        Returns:
            Coordinates after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates inside an image of shape HxW are in
            range [0, H] or [0, W].
            This function should correctly transform coordinates outside the image as well.
        """
        pass

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        """Applies the transform on axis-aligned box.

        By default will transform the corner points and use their minimum/maximum to create a new
        axis-aligned box. Note that this default may change the size of your box, e.g. after
        rotations.

        Args:
            box: Floating point array of shape Nx4 of XYXY format in absolute coordinates.

        Returns:
            Box after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates inside an image of shape HxW are in
            range [0, H] or [0, W].
            This function does not clip boxes to force them inside the image.
            It is up to the application that uses the boxes to decide.
        """
        # Convert x1, y1, x2, y2 box into 4 coordinates of ([x1, y1], [x2, y1], [x1, y2], [x2, y2])
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.asarray(box).reshape(-1, 4)[:, idxs].reshape(-1, 2)
        coords = self.apply_coords(coords).reshape((-1, 4, 2))
        minxy = coords.min(axis=1)
        maxxy = coords.max(axis=1)
        return np.concatenate((minxy, maxxy), axis=1)

    def apply_polygons(self, polygons: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Applies the transform on a list of polygons, each represented by a Nx2 array.

        By default will just transform all the points.

        Args:
            polygons: List of floating point array of shape Nx2 of (x, y) format in absolute
                coordinates.

        Returns:
            Polygons after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates on an image of shape (H, W) are in
            range [0, H] or [0, W].
        """
        return [self.apply_coords(p) for p in polygons]

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """Applies the transform on a full-image segmentation.

        By default will just perform "apply_image".

        Args:
            segmentation: The array with shape (H, W) should be type of integer or bool.

        Returns:
            Segmentation after apply the transformation.
        """
        return self.apply_image(segmentation)

    @classmethod
    def register_type(cls, data_type: str, func: Optional[Callable] = None) -> Optional[Callable]:
        """Registers the given function as a handler that will use for a specific data type.

        Args:
            data_type: The name of the data type (e.g., box).
            func: Takes a transform and a data, returns the transformed data.

        Examples:
        .. code-block:: python
            def func(flip_transform, voxel_data):
                return transformed_voxel_data
            HFlipTransform.register_type('voxel', func)

            # or, use it as a decorator
            @HFlipTransform.register_type('voxel')
            def func(flip_transform, voxel_data):
                return transformed_voxel_data

            # ...
            transform = HFlipTransform(...)
            transform.apply_voxel(voxel_data)  # func will be called
        """
        if func is None:

            def wrapper(decorated_func: Callable) -> Callable:
                if not callable(decorated_func):
                    raise TypeError(
                        'You can only register a callable to a Transform. Got {}'.format(func)
                    )
                cls.register_type(data_type, decorated_func)
                return decorated_func

            return wrapper

        if not callable(func):
            raise TypeError('You can only register a callable to a Transform. Got {}'.format(func))

        argspec = inspect.getfullargspec(func)
        if len(argspec.args) != 2:
            raise TypeError(
                'You can only register a function that takes two positional '
                'arguments to a Transform! Got a function with spec {}'.format(str(argspec))
            )
        setattr(cls, 'apply_' + data_type, func)

    def inverse(self) -> 'Transform':
        """Creates a transform that inverts the geometric changes of this transform.

        Note that the inverse is meant for geometric changes only (i.e. change of coordinates). The
        inverse of photometric transforms that do not change coordinates is defined to be a no-op,
        even if they may be invertible.

        Returns:
            Transform
        """
        raise NotImplementedError


_T = TypeVar('_T')


class TransformList(object):
    """Maintains a list of transform operations which will be applied in sequence.

    Attributes:
        transforms (list[Transform])
    """

    def __init__(self, transforms: List[Transform]) -> None:
        """
        Args:
            transforms: List of transforms to perform.
        """
        for t in transforms:
            if not isinstance(t, Transform):
                raise TypeError('Expected Transform. Got {}'.format(type(t)))
        self.transforms = transforms

    def _apply(self, x: _T, meth: str) -> _T:
        """Applies the transforms on the input.

        Args:
            x: input to apply the transform operations.
            meth: meth.

        Returns:
            x: After apply the transformation.
        """
        for t in self.transforms:
            x = getattr(t, meth)(x)
        return x

    def __getattribute__(self, name: str):
        # use __getattribute__ to win priority over any registered dtypes
        if name.startswith('apply_'):
            return lambda x: self._apply(x, name)
        return super().__getattribute__(name)

    def __add__(self, other: Union['TransformList', Transform]) -> 'TransformList':
        """
        Args:
            other: Transformation(s) to add.

        Returns:
            List of transforms.
        """
        if isinstance(other, TransformList):
            others = other.transforms
        elif isinstance(other, Transform):
            others = [other]
        else:
            raise TypeError(
                'other should either TransformList or Transform. Got {}'.format(type(other))
            )
        return TransformList(self.transforms + others)

    def __iadd__(self, other: Union['TransformList', Transform]) -> 'TransformList':
        """
        Args:
            other: Transformation(s) to add.

        Returns:
            List of transforms.
        """
        if isinstance(other, TransformList):
            others = other.transforms
        elif isinstance(other, Transform):
            others = [other]
        else:
            raise TypeError(
                'other should either TransformList or Transform. Got {}'.format(type(other))
            )
        self.transforms.extend(others)
        return self

    def __radd__(self, other: Union['TransformList', Transform]) -> 'TransformList':
        """
        Args:
            other: Transformation(s) to add.
        Returns:
            List of transforms.
        """
        if isinstance(other, TransformList):
            others = other.transforms
        elif isinstance(other, Transform):
            others = [other]
        else:
            raise TypeError(
                'other should either TransformList or Transform. Got {}'.format(type(other))
            )
        return TransformList(others + self.transforms)

    def __len__(self) -> int:
        """
        Returns:
            Number of transforms contained in the TransformList.
        """
        return len(self.transforms)

    def __getitem__(self, idx: Union[int, slice]) -> Transform:
        return self.transforms[idx]

    def inverse(self) -> 'TransformList':
        """Inverts each transform in reversed order."""
        return TransformList([t.inverse() for t in self.transforms[::-1]])


class HFlipTransform(Transform):
    """Horizontal flip."""

    def __init__(self, width: int) -> None:
        self.width = width

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        if not is_numpy(image):
            raise TypeError('image should be np.ndarray. Got {}'.format(type(image)))
        if not is_numpy_image(image):
            raise ValueError('image should be 2D/3D. Got {}D'.format(image.ndim))

        # HxWxC, HxW
        return np.flip(image, axis=1)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Note:
            The inputs are floating point coordinates, not pixel indices. Therefore they are flipped
            by `(W - x, H - y)`, not `(W - 1 - x, H - 1 - y)`.
        """
        if not is_numpy(coords):
            raise TypeError('coords should be np.ndarray. Got {}'.format(type(coords)))
        if not is_numpy_coords(coords):
            raise ValueError('coords should be of shape Nx2. Got {}'.format(coords.shape))

        coords[:, 0] = self.width - coords[:, 0]
        return coords

    def inverse(self) -> 'Transform':
        """The inverse is to flip again."""
        return self


class VFlipTransform(Transform):
    """Vertical flip."""

    def __init__(self, height: int) -> None:
        self.height = height

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        if not is_numpy(image):
            raise TypeError('image should be np.ndarray. Got {}'.format(type(image)))
        if not is_numpy_image(image):
            raise ValueError('image should be 2D/3D. Got {}D'.format(image.ndim))

        # HxWxC, HxW
        return np.flip(image, axis=0)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Note:
            The inputs are floating point coordinates, not pixel indices. Therefore they are flipped
             by `(W - x, H - y)`, not `(W - 1 - x, H - 1 - y)`.
        """
        if not is_numpy(coords):
            raise TypeError('coords should be np.ndarray. Got {}'.format(type(coords)))
        if not is_numpy_coords(coords):
            raise ValueError('coords should be of shape Nx2. Got {}'.format(coords.shape))

        coords[:, 1] = self.height - coords[:, 1]
        return coords

    def inverse(self) -> 'Transform':
        """The inverse is to flip again."""
        return self


class NoOpTransform(Transform):
    """A transform that does nothing."""

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        return image

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def __getattr__(self, name: str) -> Callable:
        if name.startswith('apply_'):
            return lambda x: x
        raise AttributeError('NoOpTransform object has no attribute {}'.format(name))

    def inverse(self) -> 'Transform':
        return self


class ResizeTransform(Transform):
    """Resizing image to a target size"""

    def __init__(self, h: int, w: int, new_h: int, new_w: int, interp: str) -> None:
        """
        Args:
            h: Original image height.
            w: Original image width.
            new_h: New image height.
            new_w: New image width.
            interp: cv2 interpolation methods. See :const:`CV2_INTER_CODES` for all options.
        """
        self.h = h
        self.w = w
        self.new_h = new_h
        self.new_w = new_w
        self.interp = interp

    def apply_image(self, image: np.ndarray, interp: Optional[str] = None) -> np.ndarray:
        if not is_numpy(image):
            raise TypeError('image should be np.ndarray. Got {}'.format(type(image)))
        if not is_numpy_image(image):
            raise ValueError('image should be 2D/3D. Got {}D'.format(image.ndim))

        h, w = image.shape[:2]
        if self.h != h or self.w != w:
            raise ValueError('Input size mismatch h w {}:{} -> {}:{}'.format(self.h, self.w, h, w))

        interp_method = interp if interp is not None else self.interp
        resized_image = cv2.resize(
            image, (self.new_w, self.new_h), interpolation=CV2_INTER_CODES[interp_method]
        )
        return resized_image

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        if not is_numpy(coords):
            raise TypeError('coords should be np.ndarray. Got {}'.format(type(coords)))
        if not is_numpy_coords(coords):
            raise ValueError('coords should be of shape Nx2. Got {}'.format(coords.shape))

        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return self.apply_image(segmentation, interp='nearest')

    def inverse(self) -> 'Transform':
        return ResizeTransform(self.new_h, self.new_w, self.h, self.w, self.interp)


class CropTransform(Transform):
    """Cropping with giving x1, y1, h, w."""

    def __init__(self, x1: int, y1: int, h: int, w: int) -> None:
        """
        Args:
            x1, y1, h, w: Crop the image by [y1:y1+h, x1:x1+w].
        """
        if x1 < 0:
            raise ValueError('x1 should >= 0. Got {}'.format(x1))
        if y1 < 0:
            raise ValueError('y1 should >= 0. Got {}'.format(y1))
        if h <= 0:
            raise ValueError('h should > 0. Got {}'.format(h))
        if w <= 0:
            raise ValueError('w should > 0. Got {}'.format(w))

        self.x1 = x1
        self.y1 = y1
        self.x2 = x1 + w
        self.y2 = y1 + h

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        if not is_numpy(image):
            raise TypeError('image should be np.ndarray. Got {}'.format(type(image)))
        if not is_numpy_image(image):
            raise ValueError('image should be 2D/3D. Got {}D'.format(image.ndim))

        # HxW, HxWxC
        return image[self.y1:self.y2, self.x1:self.x2]

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        if not is_numpy(coords):
            raise TypeError('coords should be np.ndarray. Got {}'.format(type(coords)))
        if not is_numpy_coords(coords):
            raise ValueError('coords should be of shape Nx2. Got {}'.format(coords.shape))

        coords[:, 0] -= self.x1
        coords[:, 1] -= self.y1
        return coords

    def apply_polygons(self, polygons: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Crops the polygon with the box and the number of points in the polygon might change."""
        from shapely import geometry

        # Create a window that will be used to crop
        crop_box = geometry.box(self.x1, self.y1, self.x2, self.y2).buffer(0.0)

        cropped_polygons = []

        for polygon in polygons:
            polygon = geometry.Polygon(polygon).buffer(0.0)
            # polygon must be valid to perform intersection.
            if not polygon.is_valid:
                raise ValueError('polygon is not valid')
            cropped = polygon.intersection(crop_box)
            if cropped.is_empty:
                continue
            if not isinstance(cropped, geometry.collection.BaseMultipartGeometry):
                cropped = [cropped]
            # one polygon may be cropped to multiple ones
            for poly in cropped:
                # It could produce lower dimensional objects like lines or
                # points, which we want to ignore
                if not isinstance(poly, geometry.Polygon) or not poly.is_valid:
                    continue
                coords = np.asarray(poly.exterior.coords)
                # NOTE This process will produce an extra identical vertex at the end.
                # So we remove it. This is tested by `tests/test_transform.py`
                cropped_polygons.append(coords[:-1])
        return [self.apply_coords(p) for p in cropped_polygons]


class BlendTransform(Transform):
    """Transforms pixel colors with PIL enhance functions."""

    def __init__(self, src_image: np.ndarray, src_weight: float, dst_weight: float) -> None:
        """Blends the input image (dst_image) with the src_image using formula:
        ``src_weight * src_image + dst_weight * dst_image``

        Args:
            src_image: Input image is blended with this image.
            src_weight: Blend weighting of src_image.
            dst_weight: Blend weighting of dst_image.
        """
        self.src_image = src_image
        self.src_weight = src_weight
        self.dst_weight = dst_weight

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        if not is_numpy(image):
            raise TypeError('image should be np.ndarray. Got {}'.format(type(image)))
        if not is_numpy_image(image):
            raise ValueError('image should be 2D/3D. Got {}D'.format(image.ndim))

        if image.dtype == np.uint8:
            image = image.astype(np.float32)
            image = self.src_weight * self.src_image + self.dst_weight * image
            return np.clip(image, 0, 255).astype(np.uint8)
        else:
            return self.src_weight * self.src_image + self.dst_weight * image

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        if not is_numpy(coords):
            raise TypeError('coords should be np.ndarray. Got {}'.format(type(coords)))
        if not is_numpy_coords(coords):
            raise ValueError('coords should be of shape Nx2. Got {}'.format(coords.shape))

        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    def inverse(self) -> 'Transform':
        """The inverse is a no-op."""
        return NoOpTransform()
