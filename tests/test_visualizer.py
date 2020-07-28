# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import contextlib
import io
import os.path as osp
import unittest
from typing import List, Tuple, Union

import cv2
import numpy as np
from pycocotools import mask as mask_util
from pycocotools.coco import COCO

from foundation.visualization import Visualizer, random_color


class TestVisualizer(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        np.random.seed(12)
        self.data_root = osp.join(osp.dirname(__file__), './data')

    def _random_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[List[np.ndarray]], List[np.ndarray]]:
        H, W = 100, 100
        N = 10
        img = np.random.rand(H, W, 3) * 255
        boxxy = np.random.rand(N, 2) * (H // 2)
        boxes = np.concatenate((boxxy, boxxy + H // 2), axis=1)

        def _rand_poly():
            return np.random.rand(3, 2) * H

        polygons = [[_rand_poly() for _ in range(np.random.randint(1, 5))] for _ in range(N)]

        mask = np.zeros_like(img[:, :, 0], dtype=np.bool)
        mask[:10, 10:20] = 1

        labels = [str(i) for i in range(N)]
        return img, boxes, labels, polygons, [mask] * N

    def test_correct_output_shape(self) -> None:
        img = np.random.rand(928, 928, 3) * 255
        v = Visualizer(img)
        out = v.get_image()
        self.assertEqual(out.shape, img.shape)

    def test_draw_random_instance(self) -> None:
        img, boxes, labels, polygons, masks = self._random_data()

        v = Visualizer(img)
        for box, label, polygon, mask in zip(boxes, labels, polygons, masks):
            color = tuple(random_color(rgb=True, maximum=1))
            v.draw_box(box, edge_color=color, label=label)
            for p in polygon:
                v.draw_polygon(p, color=color)
            v.draw_binary_mask(mask, color=color)
        out = v.get_image()
        self.assertEqual(out.shape, img.shape)

        # Text 2x scaling
        v = Visualizer(img, 2.0)
        for box, label, polygon, mask in zip(boxes, labels, polygons, masks):
            color = tuple(random_color(rgb=True, maximum=1))
            v.draw_box(box, edge_color=color, label=label)
            for p in polygon:
                v.draw_polygon(p, color=color)
            v.draw_binary_mask(mask, color=color)
        out = v.get_image()
        self.assertEqual(out.shape[0], img.shape[0] * 2)

    def test_draw_random_rotated_instances(self) -> None:
        H, W = 100, 150
        N = 50
        img = np.random.rand(H, W, 3) * 255
        boxes_5d = np.zeros((N, 5))
        boxes_5d[:, 0] = np.random.uniform(-0.1 * W, 1.1 * W, size=N)
        boxes_5d[:, 1] = np.random.uniform(-0.1 * H, 1.1 * H, size=N)
        boxes_5d[:, 2] = np.random.uniform(0, max(W, H), size=N)
        boxes_5d[:, 3] = np.random.uniform(0, max(W, H), size=N)
        boxes_5d[:, 4] = np.random.uniform(-1800, 1800, size=N)
        labels = [str(i) for i in range(N)]

        v = Visualizer(img)
        for box, label in zip(boxes_5d, labels):
            color = tuple(random_color(rgb=True, maximum=1))
            v.draw_rotated_box(box, edge_color=color, label=label)
        out = v.get_image()
        self.assertEqual(out.shape, img.shape)

    def _convert_segmentation(self, segmentation: Union[list, dict]) -> Union[list, np.ndarray]:
        m = segmentation
        if isinstance(m, list):
            # list[np.ndarray]
            return [np.asarray(x).reshape(-1, 2) for x in m]
        elif isinstance(m, dict):
            # RLEs
            assert 'counts' in m and 'size' in m
            if isinstance(m['counts'], list):  # uncompressed RLEs
                h, w = m['size']
                m = mask_util.frPyObjects(m, h, w)
            return mask_util.decode(m)[:, :]
        else:
            raise TypeError(
                'segmentation should be list or dict. Got {}'.format(type(segmentation))
            )

    def _draw_and_save_coco_instances(self, c: COCO, scale: float) -> None:
        cat_ids = c.getCatIds()
        cat_info_list = c.loadCats(cat_ids)
        cat_id_to_name = {x['id']: x['name'] for x in cat_info_list}

        for img_id in sorted(c.getImgIds()):
            img_info = c.loadImgs(img_id)[0]

            file_name = img_info['file_name']
            img = cv2.imread(osp.join(self.data_root, img_info['file_name']))[:, :, ::-1]
            v = Visualizer(img, scale=scale)

            ann_ids = c.getAnnIds(img_id)
            anns = c.loadAnns(ann_ids)
            for ann in anns:
                label = cat_id_to_name[ann['category_id']]
                color = tuple(random_color(rgb=True, maximum=1))

                x, y, w, h = ann['bbox']
                v.draw_box((x, y, x + w, y + h), label=label, edge_color=color)
                segmentation = self._convert_segmentation(ann['segmentation'])

                if isinstance(segmentation, list):
                    for p in segmentation:
                        v.draw_polygon(p, color=color)
                else:
                    v.draw_binary_mask(segmentation, color=color)

            save_filename = '{}_{}.jpg'.format(file_name[:-4], scale)
            v.save(osp.join(osp.dirname(__file__), 'outputs', save_filename))

    def test_draw_coco_instances(self) -> None:
        inst_ann_file = osp.join(self.data_root, 'instances_val2014_demo.json')
        with contextlib.redirect_stdout(io.StringIO()):
            c = COCO(inst_ann_file)

        self._draw_and_save_coco_instances(c, scale=1.0)

        # Text 2x scaling
        self._draw_and_save_coco_instances(c, scale=2.0)
