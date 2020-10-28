# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
import itertools
import unittest
from typing import Any, Tuple

import cv2
import numpy as np

from foundation.transforms import transform as T
from foundation.transforms.transform import CV2_INTER_CODES


class TestTransforms(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)

    def test_register(self):
        dtype = 'int'

        def add1(t, x):
            return x + 1

        def flip_sub_width(t, x):
            return x - t.width

        T.Transform.register_type(dtype, add1)
        T.HFlipTransform.register_type(dtype, flip_sub_width)

        transforms = T.TransformList(
            [
                T.ResizeTransform(0, 0, 0, 0, 'bilinear'),
                T.CropTransform(0, 0, 10, 10),
                T.HFlipTransform(3),
            ]
        )
        self.assertEqual(transforms.apply_int(3), 2)

        # Testing __add__, __iadd__, __radd__, __len__.
        transforms = transforms + transforms
        transforms += transforms
        transforms = T.NoOpTransform() + transforms
        self.assertEqual(len(transforms), 13)

        with self.assertRaises(TypeError):
            T.HFlipTransform.register_type(dtype, lambda x: 1)

        with self.assertRaises(AttributeError):
            transforms.no_existing

    def test_register_with_decorator(self):
        """
        Test register using decorator.
        """
        dtype = 'float'

        @T.HFlipTransform.register_type(dtype)
        def add1(t, x):
            return x + 1

        transforms = T.TransformList([T.HFlipTransform(3)])
        self.assertEqual(transforms.apply_float(3), 4)

    def test_noop_transform_no_register(self):
        """NoOpTransform does not need register - it's by default no-op."""
        t = T.NoOpTransform()
        self.assertEqual(t.apply_anything(1), 1)

    @staticmethod
    def BlendTransform_img_gt(img, *args) -> Tuple[np.ndarray, list]:
        """
        Given the input array, return the expected output array and shape after
        applying the blend transformation.
        Args:
            imgs (array): image array before the transform.
            args (list): list of arguments. Details can be found in test case.
        Returns:
            img (array): expected output array after apply the transformation.
            (list): expected shape of the output array.
        """
        src_image, src_weight, dst_weight = args
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = src_weight * src_image + dst_weight * img
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = src_weight * src_image + dst_weight * img
        return img, img.shape

    @staticmethod
    def CropTransform_img_gt(img, *args) -> Tuple[np.ndarray, list]:
        """
        Given the input array, return the expected output array and shape after
        applying the crop transformation.
        Args:
            img (array): image array before the transform.
            args (list): list of arguments. Details can be found in test case.
        Returns:
            img (array): expected output array after apply the transformation.
            (list): expected shape of the output array.
        """
        x1, y1, h, w = args
        ret = img[y1:y1 + h, x1:x1 + w]
        return ret, ret.shape

    @staticmethod
    def VFlipTransform_img_gt(img, *args) -> Tuple[np.ndarray, list]:
        """
        Given the input array, return the expected output array and shape after
        applying the vertical flip transformation.
        Args:
            img (array): image array before the transform.
            args (list): list of arguments. Details can be found in test case.
        Returns:
            img (array): expected output array after apply the transformation.
            (list): expected shape of the output array.
        """
        return img[::-1, :], img.shape

    @staticmethod
    def HFlipTransform_img_gt(img, *args) -> Tuple[np.ndarray, list]:
        """
        Given the input array, return the expected output array and shape after
        applying the horizontal flip transformation.
        Args:
            img (array): image array before the transform.
            args (list): list of arguments. Details can be found in test case.
        Returns:
            img (array): expected output array after apply the transformation.
            (list): expected shape of the output array.
        """
        return img[:, ::-1], img.shape

    @staticmethod
    def NoOpTransform_img_gt(img, *args) -> Tuple[np.ndarray, list]:
        """
        Given the input array, return the expected output array and shape after
        applying no transformation.
        Args:
            img (array): image array before the transform.
            args (list): list of arguments. Details can be found in test case.
        Returns:
            img (array): expected output array after apply the transformation.
            (list): expected shape of the output array.
        """
        return img, img.shape

    @staticmethod
    def ResizeTransform_img_gt(img, *args) -> Tuple[Any, Any]:
        """
        Given the input array, return the expected output array and shape after
        applying the resize transformation.
        Args:
            img (array): image array before the transform.
            args (list): list of arguments. Details can be found in test case.
        Returns:
            img (array): expected output array after apply the transformation.
                None means does not have expected output array for sanity check.
            (list): expected shape of the output array. None means does not have
                expected output shape for sanity check.
        """
        h, w, new_h, new_w, interp = args
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=CV2_INTER_CODES[interp])
        return resized_img, resized_img.shape

    @staticmethod
    def _seg_provider(n: int = 8, h: int = 10, w: int = 10) -> np.ndarray:
        """
        Provide different segmentations as test cases.
        Args:
            n (int): number of points to generate in the image as segmentations.
            h, w (int): height and width dimensions.
        Returns:
            (np.ndarray): the segmentation to test on.
        """
        # Prepare random segmentation as test cases.
        for _ in range(n):
            yield np.random.randint(2, size=(h, w))

    @staticmethod
    def _img_provider(c: int = 3, h: int = 10, w: int = 10) -> Tuple[np.ndarray, type, str]:
        """
        Provide different image inputs as test cases.
        Args:
            c, h, w (int): channel, height, and width dimensions.
        Returns:
            (np.ndarray): an image to test on.
            (type): type of the current array.
            (str): string to represent the shape. Options include `hw`, `hwc`,
                `nhwc`.
        """
        # Prepare mesh grid as test case.
        img_h_grid, img_w_grid = np.mgrid[0:h * 2:2, 0:w * 2:2]
        img_hw_grid = img_h_grid * w + img_w_grid
        img_hwc_grid = np.repeat(img_hw_grid[:, :, None], c, axis=2)

        # Prepare random array as test case.
        img_hw_random = np.random.rand(h, w)
        img_hwc_random = np.random.rand(h, w, c)

        for array_type, input_shape, init in itertools.product(
            [np.uint8, np.float32], ['hw', 'hwc'], ['grid', 'random']
        ):  # yapf: disable
            yield locals()['img_{}_{}'.format(input_shape,
                                              init)].astype(array_type), array_type, input_shape

    def test_blend_img_transforms(self):
        """
        Test BlendTransform.
        """
        _trans_name = 'BlendTransform'
        blend_src_hw = np.ones((10, 10))
        blend_src_hwc = np.ones((10, 10, 3))
        blend_src_nhwc = np.ones((8, 10, 10, 3))

        for img, array_type, shape_str in TestTransforms._img_provider():
            blend_src = locals()['blend_src_{}'.format(shape_str)].astype(array_type)
            params = (
                (blend_src, 0.0, 1.0),
                (blend_src, 0.3, 0.7),
                (blend_src, 0.5, 0.5),
                (blend_src, 0.7, 0.3),
                (blend_src, 1.0, 0.0),
            )
            for param in params:
                gt_transformer = getattr(self, '{}_img_gt'.format(_trans_name))
                transformer = getattr(T, _trans_name)(*param)

                result = transformer.apply_image(img)
                img_gt, shape_gt = gt_transformer(img, *param)

                self.assertEqual(
                    shape_gt,
                    result.shape,
                    'transform {} failed to pass the shape check with'
                    'params {} given input with shape {} and type {}'.format(
                        _trans_name, param, shape_str, array_type
                    ),
                )
                self.assertTrue(
                    np.allclose(result, img_gt),
                    'transform {} failed to pass the value check with'
                    'params {} given input with shape {} and type {}'.format(
                        _trans_name, param, shape_str, array_type
                    ),
                )

    def test_crop_img_transforms(self):
        """
        Test CropTransform..
        """
        _trans_name = 'CropTransform'
        params = (
            # (0, 0, 0, 0),
            (0, 0, 1, 1),
            (0, 0, 6, 1),
            (0, 0, 1, 6),
            (0, 0, 6, 6),
            (1, 3, 6, 6),
            (3, 1, 6, 6),
            (3, 3, 6, 6),
            (6, 6, 6, 6),
        )
        for (img, array_type, shape_str), param in itertools.product(
            TestTransforms._img_provider(), params
        ):  # yapf: disable
            gt_transformer = getattr(self, '{}_img_gt'.format(_trans_name))
            transformer = getattr(T, _trans_name)(*param)

            result = transformer.apply_image(img)
            img_gt, shape_gt = gt_transformer(img, *param)

            self.assertEqual(
                shape_gt,
                result.shape,
                'transform {} failed to pass the shape check with'
                'params {} given input with shape {} and type {}'.format(
                    _trans_name, param, shape_str, array_type
                ),
            )
            self.assertTrue(
                np.allclose(result, img_gt),
                'transform {} failed to pass the value check with'
                'params {} given input with shape {} and type {}'.format(
                    _trans_name, param, shape_str, array_type
                ),
            )

    def test_vflip_img_transforms(self):
        """
        Test VFlipTransform..
        """
        _trans_name = 'VFlipTransform'
        params = ((0,), (1,))

        for (img, array_type, shape_str), param in itertools.product(
            TestTransforms._img_provider(), params
        ):  # yapf: disable
            gt_transformer = getattr(self, '{}_img_gt'.format(_trans_name))
            transformer = getattr(T, _trans_name)(*param)

            result = transformer.apply_image(img)
            img_gt, shape_gt = gt_transformer(img, *param)

            self.assertEqual(
                shape_gt,
                result.shape,
                'transform {} failed to pass the shape check with'
                'params {} given input with shape {} and type {}'.format(
                    _trans_name, param, shape_str, array_type
                ),
            )
            self.assertTrue(
                np.allclose(result, img_gt),
                'transform {} failed to pass the value check with'
                'params {} given input with shape {} and type {}.\n'
                'Output: {} -> {}'.format(
                    _trans_name, param, shape_str, array_type, result, img_gt
                ),
            )

    def test_hflip_img_transforms(self):
        """
        Test HFlipTransform..
        """
        _trans_name = 'HFlipTransform'
        params = ((0,), (1,))

        for (img, array_type, shape_str), param in itertools.product(
            TestTransforms._img_provider(), params
        ):  # yapf: disable
            gt_transformer = getattr(self, '{}_img_gt'.format(_trans_name))
            transformer = getattr(T, _trans_name)(*param)

            result = transformer.apply_image(img)
            img_gt, shape_gt = gt_transformer(img, *param)

            self.assertEqual(
                shape_gt,
                result.shape,
                'transform {} failed to pass the shape check with'
                'params {} given input with shape {} and type {}'.format(
                    _trans_name, param, shape_str, array_type
                ),
            )
            self.assertTrue(
                np.allclose(result, img_gt),
                'transform {} failed to pass the value check with'
                'params {} given input with shape {} and type {}.\n'
                'Output: {} -> {}'.format(
                    _trans_name, param, shape_str, array_type, result, img_gt
                ),
            )

    def test_no_op_img_transforms(self):
        """
        Test NoOpTransform..
        """
        _trans_name = 'NoOpTransform'
        params = ()

        for (img, array_type, shape_str), param in itertools.product(
            TestTransforms._img_provider(), params
        ):  # yapf: disable
            gt_transformer = getattr(self, '{}_img_gt'.format(_trans_name))
            transformer = getattr(T, _trans_name)(*param)

            result = transformer.apply_image(img)
            img_gt, shape_gt = gt_transformer(img, *param)

            self.assertEqual(
                shape_gt,
                result.shape,
                'transform {} failed to pass the shape check with'
                'params {} given input with shape {} and type {}'.format(
                    _trans_name, param, shape_str, array_type
                ),
            )
            self.assertTrue(
                np.allclose(result, img_gt),
                'transform {} failed to pass the value check with'
                'params {} given input with shape {} and type {}'.format(
                    _trans_name, param, shape_str, array_type
                ),
            )

    def test_resize_img_transforms(self):
        """
        Test ResizeTransform.
        """
        _trans_name = 'ResizeTransform'
        # Testing success cases.
        params = (
            (10, 20, 20, 20, 'nearest'),
            (10, 20, 10, 20, 'nearest'),
            (10, 20, 20, 10, 'nearest'),
            (10, 20, 1, 1, 'nearest'),
            (10, 20, 3, 3, 'nearest'),
            (10, 20, 5, 10, 'nearest'),
            (10, 20, 10, 5, 'nearest'),
            (10, 20, 20, 20, 'bilinear'),
            (10, 20, 10, 20, 'bilinear'),
            (10, 20, 20, 10, 'bilinear'),
            (10, 20, 1, 1, 'bilinear'),
            (10, 20, 3, 3, 'bilinear'),
            (10, 20, 5, 10, 'bilinear'),
            (10, 20, 10, 5, 'bilinear'),
        )

        for (img, array_type, shape_str), param in itertools.product(
            TestTransforms._img_provider(h=10, w=20), params
        ):  # yapf: disable
            gt_transformer = getattr(self, '{}_img_gt'.format(_trans_name))
            transformer = getattr(T, _trans_name)(*param)

            result = transformer.apply_image(img)
            img_gt, shape_gt = gt_transformer(img, *param)

            if shape_gt is not None:
                self.assertEqual(
                    shape_gt,
                    result.shape,
                    'transform {} failed to pass the shape check with'
                    'params {} given input with shape {} and type {}'.format(
                        _trans_name, param, shape_str, array_type
                    ),
                )
            if img_gt is not None:
                self.assertTrue(
                    np.allclose(result, img_gt),
                    'transform {} failed to pass the value check with'
                    'params {} given input with shape {} and type {}'.format(
                        _trans_name, param, shape_str, array_type
                    ),
                )

        # Testing failure cases.
        params = (
            (0, 0, 20, 20, 'nearest'),
            (0, 0, 0, 0, 'nearest'),
            (-1, 0, 0, 0, 'nearest'),
            (0, -1, 0, 0, 'nearest'),
            (0, 0, -1, 0, 'nearest'),
            (0, 0, 0, -1, 'nearest'),
            (20, 10, 0, -1, 'nearest'),
        )

        for (img, _, _), param in itertools.product(
            TestTransforms._img_provider(h=10, w=20), params
        ):  # yapf: disable
            gt_transformer = getattr(self, '{}_img_gt'.format(_trans_name))
            transformer = getattr(T, _trans_name)(*param)
            with self.assertRaises((RuntimeError, ValueError)):
                result = transformer.apply_image(img)

    def test_crop_polygons(self):
        # Ensure that shapely produce an extra vertex at the end
        # This is assumed when copping polygons
        try:
            import shapely.geometry as geometry
        except ImportError:
            return

        polygon = np.asarray([3, 3.5, 11, 10.0, 38, 98, 15.0, 100.0]).reshape(-1, 2)
        g = geometry.Polygon(polygon)
        coords = np.asarray(g.exterior.coords)
        self.assertEqual(coords[0].tolist(), coords[-1].tolist())

    @staticmethod
    def _coords_provider(
        num_coords: int = 5,
        n: int = 50,
        h_max: int = 10,
        h_min: int = 0,
        w_max: int = 10,
        w_min: int = 0,
    ) -> Tuple[np.ndarray, type, str]:
        """
        Provide different coordinate inputs as test cases.
        Args:
            num_coords (int): number of coordinates to provide.
            n (int): size of the batch.
            h_max, h_min (int): max, min coordinate value on height dimension.
            w_max, w_min (int): max, min coordinate value on width dimension.
        Returns:
            (np.ndarray): coordinates array of shape Nx2 to test on.
        """
        for _ in range(num_coords):
            yield np.concatenate(
                [
                    np.random.randint(low=h_min, high=h_max, size=(n, 1)),
                    np.random.randint(low=w_min, high=w_max, size=(n, 1)),
                ],
                axis=1,
            ).astype('float32')

    @staticmethod
    def BlendTransform_coords_gt(coords, *args) -> Tuple[np.ndarray, list]:
        """
        Given the input array, return the expected output array and shape after
        applying the blend transformation.
        Args:
            coords (array): coordinates before the transform.
            args (list): list of arguments. Details can be found in test case.
        Returns:
            coords (array): expected output coordinates after apply the
                transformation.
            (list): expected shape of the output array.
        """
        return coords, coords.shape

    def test_blend_coords_transforms(self):
        """
        Test BlendTransform.
        """
        _trans_name = 'BlendTransform'
        for coords in TestTransforms._coords_provider(w_max=10, h_max=20):
            params = (
                (coords, 0.0, 1.0),
                (coords, 0.3, 0.7),
                (coords, 0.5, 0.5),
                (coords, 0.7, 0.3),
                (coords, 1.0, 0.0),
            )
            for param in params:
                gt_transformer = getattr(self, '{}_coords_gt'.format(_trans_name))
                transformer = getattr(T, _trans_name)(*param)

                result = transformer.apply_coords(np.copy(coords))
                coords_gt, shape_gt = gt_transformer(np.copy(coords), *param)

                self.assertEqual(
                    shape_gt,
                    result.shape,
                    'transform {} failed to pass the shape check with'
                    'params {} given input with shape {}'.format(_trans_name, param, result.shape),
                )
                self.assertTrue(
                    np.allclose(result, coords_gt),
                    'transform {} failed to pass the value check with'
                    'params {} given input with shape {}'.format(_trans_name, param, result.shape),
                )

                coords_inversed = transformer.inverse().apply_coords(result)
                self.assertTrue(
                    np.allclose(coords_inversed, coords), "Transform {}'s inverse fails to "
                    'produce the original coordinates.'.format(_trans_name)
                )

    @staticmethod
    def VFlipTransform_coords_gt(coords, *args) -> Tuple[np.ndarray, list]:
        """
        Given the input array, return the expected output array and shape after
        applying the vflip transformation.
        Args:
            coords (array): coordinates before the transform.
            args (list): list of arguments. Details can be found in test case.
        Returns:
            coords (array): expected output coordinates after apply the
                transformation.
            (list): expected shape of the output array.
        """
        height = args
        coords[:, 1] = height - coords[:, 1]
        return coords, coords.shape

    def test_vflip_coords_transforms(self):
        """
        Test VFlipTransform.
        """
        _trans_name = 'VFlipTransform'

        params = ((20,), (30,))
        for coords, param in itertools.product(TestTransforms._coords_provider(), params):
            gt_transformer = getattr(self, '{}_coords_gt'.format(_trans_name))
            transformer = getattr(T, _trans_name)(*param)

            result = transformer.apply_coords(np.copy(coords))
            coords_gt, shape_gt = gt_transformer(np.copy(coords), *param)

            self.assertEqual(
                shape_gt,
                result.shape,
                'transform {} failed to pass the shape check with'
                'params {} given input with shape {}'.format(_trans_name, param, result.shape),
            )
            self.assertTrue(
                np.allclose(result, coords_gt),
                'transform {} failed to pass the value check with'
                'params {} given input with shape {}'.format(_trans_name, param, result.shape),
            )

            coords_inversed = transformer.inverse().apply_coords(result)
            self.assertTrue(
                np.allclose(coords_inversed, coords), "Transform {}'s inverse fails to "
                'produce the original coordinates.'.format(_trans_name)
            )

    @staticmethod
    def HFlipTransform_coords_gt(coords, *args) -> Tuple[np.ndarray, list]:
        """
        Given the input array, return the expected output array and shape after
        applying the hflip transformation.
        Args:
            coords (array): coordinates before the transform.
            args (list): list of arguments. Details can be found in test case.
        Returns:
            coords (array): expected output coordinates after apply the
                transformation.
            (list): expected shape of the output array.
        """
        width = args
        coords[:, 0] = width - coords[:, 0]
        return coords, coords.shape

    def test_hflip_coords_transforms(self):
        """
        Test HFlipTransform.
        """
        _trans_name = 'HFlipTransform'

        params = ((20,), (30,))
        for coords, param in itertools.product(TestTransforms._coords_provider(), params):
            gt_transformer = getattr(self, '{}_coords_gt'.format(_trans_name))
            transformer = getattr(T, _trans_name)(*param)

            result = transformer.apply_coords(np.copy(coords))
            coords_gt, shape_gt = gt_transformer(np.copy(coords), *param)

            self.assertEqual(
                shape_gt,
                result.shape,
                'transform {} failed to pass the shape check with'
                'params {} given input with shape {}'.format(_trans_name, param, result.shape),
            )
            self.assertTrue(
                np.allclose(result, coords_gt),
                'transform {} failed to pass the value check with'
                'params {} given input with shape {}'.format(_trans_name, param, result.shape),
            )

            coords_inversed = transformer.inverse().apply_coords(result)
            self.assertTrue(
                np.allclose(coords_inversed, coords), "Transform {}'s inverse fails to "
                'produce the original coordinates.'.format(_trans_name)
            )

    @staticmethod
    def CropTransform_coords_gt(coords, *args) -> Tuple[np.ndarray, list]:
        """
        Given the input array, return the expected output array and shape after
        applying the crop transformation.
        Args:
            coords (array): coordinates before the transform.
            args (list): list of arguments. Details can be found in test case.
        Returns:
            coords (array): expected output coordinates after apply the
                transformation.
            (list): expected shape of the output array.
        """
        x1, y1, h, w = args
        coords[:, 0] -= x1
        coords[:, 1] -= y1
        return coords, coords.shape

    def test_crop_coords_transforms(self):
        """
        Test CropTransform.
        """
        _trans_name = 'CropTransform'
        params = (
            # (0, 0, 0, 0),
            (0, 0, 1, 1),
            (0, 0, 6, 1),
            (0, 0, 1, 6),
            (0, 0, 6, 6),
            (1, 3, 6, 6),
            (3, 1, 6, 6),
            (3, 3, 6, 6),
            (6, 6, 6, 6),
        )
        for coords, param in itertools.product(TestTransforms._coords_provider(), params):
            gt_transformer = getattr(self, '{}_coords_gt'.format(_trans_name))
            transformer = getattr(T, _trans_name)(*param)

            result = transformer.apply_coords(np.copy(coords))
            coords_gt, shape_gt = gt_transformer(np.copy(coords), *param)

            self.assertEqual(
                shape_gt,
                result.shape,
                'transform {} failed to pass the shape check with'
                'params {} given input with shape {}'.format(_trans_name, param, result.shape),
            )
            self.assertTrue(
                np.allclose(result, coords_gt),
                'transform {} failed to pass the value check with'
                'params {} given input with shape {}'.format(_trans_name, param, result.shape),
            )

            with self.assertRaises((NotImplementedError,)):
                _ = transformer.inverse().apply_coords(result)

    @staticmethod
    def ResizeTransform_coords_gt(coords, *args) -> Tuple[np.ndarray, list]:
        """
        Given the input array, return the expected output array and shape after
        applying the crop transformation.
        Args:
            coords (array): coordinates before the transform.
            args (list): list of arguments. Details can be found in test case.
        Returns:
            coords (array): expected output coordinates after apply the
                transformation.
            (list): expected shape of the output array.
        """
        h, w, new_h, new_w = args

        coords[:, 0] = coords[:, 0] * (new_w * 1.0 / w)
        coords[:, 1] = coords[:, 1] * (new_h * 1.0 / h)
        return coords, coords.shape

    def test_resize_coords_transforms(self):
        """
        Test ResizeTransform.
        """
        _trans_name = 'ResizeTransform'
        params = (
            (10, 20, 20, 20),
            (10, 20, 10, 20),
            (10, 20, 20, 10),
            (10, 20, 1, 1),
            (10, 20, 3, 3),
            (10, 20, 5, 10),
            (10, 20, 10, 5),
        )

        for coords, param in itertools.product(TestTransforms._coords_provider(), params):
            gt_transformer = getattr(self, '{}_coords_gt'.format(_trans_name))
            transformer = getattr(T, _trans_name)(*param, interp='bilinear')

            result = transformer.apply_coords(np.copy(coords))
            coords_gt, shape_gt = gt_transformer(np.copy(coords), *param)

            self.assertEqual(
                shape_gt,
                result.shape,
                'transform {} failed to pass the shape check with'
                'params {} given input with shape {}'.format(_trans_name, param, result.shape),
            )
            self.assertTrue(
                np.allclose(result, coords_gt),
                'transform {} failed to pass the value check with'
                'params {} given input with shape {}'.format(_trans_name, param, result.shape),
            )

            coords_inversed = transformer.inverse().apply_coords(result)
            self.assertTrue(
                np.allclose(coords_inversed, coords), "Transform {}'s inverse fails to "
                'produce the original coordinates.'.format(_trans_name)
            )

    @staticmethod
    def BlendTransform_seg_gt(seg, *args) -> Tuple[np.ndarray, list]:
        """
        Given the input segmentation, return the expected output array and shape
        after applying the blend transformation.
        Args:
            seg (array): segmentation before the transform.
            args (list): list of arguments. Details can be found in test case.
        Returns:
            seg (array): expected output segmentation after apply the
                transformation.
            (list): expected shape of the output array.
        """
        return seg, seg.shape

    def test_blend_seg_transforms(self):
        """
        Test BlendTransform.
        """
        _trans_name = 'BlendTransform'
        for seg in TestTransforms._seg_provider(w=10, h=20):
            params = (
                (seg, 0.0, 1.0),
                (seg, 0.3, 0.7),
                (seg, 0.5, 0.5),
                (seg, 0.7, 0.3),
                (seg, 1.0, 0.0),
            )
            for param in params:
                gt_transformer = getattr(self, '{}_seg_gt'.format(_trans_name))
                transformer = getattr(T, _trans_name)(*param)

                result = transformer.apply_segmentation(seg)
                seg_gt, shape_gt = gt_transformer(seg, *param)

                self.assertEqual(
                    shape_gt,
                    result.shape,
                    'transform {} failed to pass the shape check with'
                    'params {} given input with shape {}'.format(_trans_name, param, result.shape),
                )
                self.assertTrue(
                    np.allclose(result, seg_gt),
                    'transform {} failed to pass the value check with'
                    'params {} given input with shape {}'.format(_trans_name, param, result.shape),
                )

    @staticmethod
    def ResizeTransform_seg_gt(seg, *args) -> Tuple[np.ndarray, list]:
        """
        Given the input segmentation, return the expected output array and shape
        after applying the blend transformation.
        Args:
            seg (array): segmentation before the transform.
            args (list): list of arguments. Details can be found in test case.
        Returns:
            seg (array): expected output segmentation after apply the
                transformation.
            (list): expected shape of the output array.
        """
        h, w, new_h, new_w = args
        resized_seg = cv2.resize(seg, (new_w, new_h), interpolation=CV2_INTER_CODES['nearest'])
        return resized_seg, resized_seg.shape

    def test_resize_seg_transforms(self):
        """
        Test ResizeTransform.
        """
        _trans_name = 'ResizeTransform'
        params = (
            (10, 20, 20, 20),
            (10, 20, 10, 20),
            (10, 20, 20, 10),
            (10, 20, 1, 1),
            (10, 20, 3, 3),
            (10, 20, 5, 10),
            (10, 20, 10, 5),
        )

        for seg, param in itertools.product(TestTransforms._seg_provider(h=10, w=20), params):
            gt_transformer = getattr(self, '{}_seg_gt'.format(_trans_name))
            transformer = getattr(T, _trans_name)(*param, interp='bilinear')

            result = transformer.apply_segmentation(seg)
            seg_gt, shape_gt = gt_transformer(seg, *param)

            if shape_gt is not None:
                self.assertEqual(
                    shape_gt,
                    result.shape,
                    'transform {} failed to pass the shape check with'
                    'params {} given input with shape {}'.format(_trans_name, param, result.shape),
                )
            if seg_gt is not None:
                self.assertTrue(
                    np.allclose(result, seg_gt),
                    'transform {} failed to pass the value check with'
                    'params {} given input with shape {}'.format(_trans_name, param, result.shape),
                )

        # Testing failure cases.
        params = (
            (0, 0, 20, 20),
            (0, 0, 0, 0),
            (-1, 0, 0, 0),
            (0, -1, 0, 0),
            (0, 0, -1, 0),
            (0, 0, 0, -1),
            (20, 10, 0, -1),
        )
        for seg, param in itertools.product(TestTransforms._seg_provider(w=10, h=20), params):
            gt_transformer = getattr(self, '{}_seg_gt'.format(_trans_name))
            transformer = getattr(T, _trans_name)(*param, interp='bilinear')
            with self.assertRaises((RuntimeError, ValueError, cv2.error)):
                result = transformer.apply_image(seg)

    @staticmethod
    def NoOpTransform_coords_gt(coords, *args) -> Tuple[np.ndarray, list]:
        """
        Given the input array, return the expected output array and shape after
        applying no transformation.
        Args:
            coords (array): coordinates before the transform.
            args (list): list of arguments. Details can be found in test case.
        Returns:
            coords (array): expected output coordinates after apply the
                transformation.
            (list): expected shape of the output array.
        """
        return coords, coords.shape

    def test_no_op_coords_transforms(self):
        """
        Test NoOpTransform..
        """
        _trans_name = 'NoOpTransform'
        params = ()

        for coords, param in itertools.product(TestTransforms._coords_provider(), params):
            gt_transformer = getattr(self, '{}_coords_gt'.format(_trans_name))
            transformer = getattr(T, _trans_name)(*param)

            result = transformer.apply_coords(np.copy(coords))
            coords_gt, shape_gt = gt_transformer(np.copy(coords), *param)

            self.assertEqual(
                shape_gt,
                result.shape,
                'transform {} failed to pass the shape check with'
                'params {} given input with shape {}'.format(_trans_name, param, result.shape),
            )
            self.assertTrue(
                np.allclose(result, coords_gt),
                'transform {} failed to pass the value check with'
                'params {} given input with shape {}'.format(_trans_name, param, result.shape),
            )

            coords_inversed = transformer.inverse().apply_coords(result)
            self.assertTrue(
                np.allclose(coords_inversed, coords), "Transform {}'s inverse fails to "
                'produce the original coordinates.'.format(_trans_name)
            )

    def test_transformlist_flatten(self):
        t0 = T.HFlipTransform(width=100)
        t1 = T.ResizeTransform(3, 4, 5, 6, None)
        t2 = T.CropTransform(4, 5, 6, 7)
        t = T.TransformList([T.TransformList([t0, t1]), t2])
        self.assertEqual(len(t.transforms), 3)

    def test_print_transform(self):
        t0 = T.HFlipTransform(width=100)
        self.assertEqual(str(t0), 'HFlipTransform(width=100)')

        t = T.TransformList([T.NoOpTransform(), t0])
        self.assertEqual(str(t), f'TransformList[NoOpTransform(), {t0}]')

        t = T.BlendTransform(np.zeros((100, 100, 100)), 1.0, 1.0)
        self.assertEqual(str(t), 'BlendTransform(src_image=..., src_weight=1.0, dst_weight=1.0)')
