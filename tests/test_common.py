# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import math
import time
import unittest
from typing import IO, Any

from foundation.common.config import CfgNode
from foundation.common.file_handler import BaseFileHandler, HandlerRegistry
from foundation.common.timer import Timer


class TestTimer(unittest.TestCase):

    def test_timer(self) -> None:
        """Test basic timer functions (pause, resume, and reset)."""
        timer = Timer()
        time.sleep(0.5)
        self.assertTrue(0.99 > timer.seconds() >= 0.5)

        timer.pause()
        time.sleep(0.5)

        self.assertTrue(0.99 > timer.seconds() >= 0.5)

        timer.resume()
        time.sleep(0.5)
        self.assertTrue(1.49 > timer.seconds() >= 1.0)

        timer.reset()
        self.assertTrue(0.49 > timer.seconds() >= 0)

    def test_avg_second(self) -> None:
        """Test avg_seconds that counts the average time."""
        for pause_second in (0.1, 0.15):
            timer = Timer()
            for t in (pause_second,) * 10:
                if timer.is_paused():
                    timer.resume()
                time.sleep(t)
                timer.pause()
                self.assertTrue(
                    math.isclose(pause_second, timer.avg_seconds(), rel_tol=5e-2),
                    msg='{}: {}'.format(pause_second, timer.avg_seconds()),
                )


class TestFileHandler(unittest.TestCase):

    def test_handler_registry(self) -> None:
        """Test file handler registry."""
        # Preset file handler
        available_keys = ['json', 'yaml', 'yml', 'pickle', 'pkl']
        for key in available_keys:
            HandlerRegistry.get(key)

        @HandlerRegistry.register('noop')
        class NoOpHandler(BaseFileHandler):
            """A handler that does nothing."""

            def load_from_fileobj(self, file: IO, **kwargs: Any) -> Any:
                pass

            def dump_to_fileobj(self, obj: Any, file: IO, **kwargs: Any) -> None:
                pass

            def dump_to_str(self, obj: Any, **kwargs: Any) -> str:
                pass

        noop = HandlerRegistry.get('noop')()
        self.assertTrue(isinstance(noop, NoOpHandler))


class TestCfgNode(unittest.TestCase):

    @staticmethod
    def gen_default_cfg() -> CfgNode:
        cfg = CfgNode()
        cfg.KEY1 = 'default'
        cfg.KEY2 = 'default'
        cfg.EXPRESSION = [3.0]

        return cfg

    def test_merge_from_file(self) -> None:
        """Tests merge_from_file function provided in the class."""
        import pkg_resources

        base_yaml = pkg_resources.resource_filename(__name__, 'configs/base.yaml')
        config_yaml = pkg_resources.resource_filename(__name__, 'configs/config.yaml')

        cfg = TestCfgNode.gen_default_cfg()
        cfg.merge_from_file(base_yaml)
        self.assertEqual(cfg.KEY1, 'base')
        self.assertEqual(cfg.KEY2, 'base')

        cfg = TestCfgNode.gen_default_cfg()

        with self.assertRaises(Exception):
            # config.yaml contains unsafe yaml tags,
            # test if an exception is thrown
            cfg.merge_from_file(config_yaml)

        cfg.merge_from_file(config_yaml, allow_unsafe=True)
        self.assertEqual(cfg.KEY1, 'base')
        self.assertEqual(cfg.KEY2, 'config')
        self.assertEqual(cfg.EXPRESSION, [1, 4, 9])

    def test_merge_from_list(self) -> None:
        """Tests merge_from_list function provided in the class."""
        cfg = TestCfgNode.gen_default_cfg()
        cfg.merge_from_list(['KEY1', 'list1', 'KEY2', 'list2'])
        self.assertEqual(cfg.KEY1, 'list1')
        self.assertEqual(cfg.KEY2, 'list2')

    def test_setattr(self) -> None:
        """Tests __setattr__ function provided in the class."""
        cfg = TestCfgNode.gen_default_cfg()
        cfg.KEY1 = 'new1'
        cfg.KEY3 = 'new3'
        self.assertEqual(cfg.KEY1, 'new1')
        self.assertEqual(cfg.KEY3, 'new3')

        # Test computed attributes, which can be inserted regardless of whether
        # the CfgNode is frozen or not.
        cfg = TestCfgNode.gen_default_cfg()
        cfg.COMPUTED_1 = 'computed1'
        self.assertEqual(cfg.COMPUTED_1, 'computed1')
        cfg.freeze()
        cfg.COMPUTED_2 = 'computed2'
        self.assertEqual(cfg.COMPUTED_2, 'computed2')

        # Test computed attributes, which should be 'insert only' (could not be
        # updated).
        cfg = TestCfgNode.gen_default_cfg()
        cfg.COMPUTED_1 = 'computed1'
        with self.assertRaises(KeyError) as err:
            cfg.COMPUTED_1 = 'update_computed1'
        self.assertTrue("Computed attributed 'COMPUTED_1' already exists" in str(err.exception))

        # Resetting the same value should be safe:
        cfg.COMPUTED_1 = 'computed1'
