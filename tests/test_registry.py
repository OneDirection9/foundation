from __future__ import absolute_import, division, print_function
import unittest

from foundation.registry import Registry


class TestRegistry(unittest.TestCase):
    def test_register(self):
        """Test register method."""

        MyRegistry = Registry("Test")

        @MyRegistry.register("T2")
        @MyRegistry.register
        class T1(object):
            pass

        self.assertTrue(MyRegistry.get("T1") == T1)
        self.assertTrue(MyRegistry.get("T2") == T1)

        MyRegistry.register("T3")(T1)
        self.assertTrue(MyRegistry.get("T3") == T1)

        @MyRegistry.register("f2")
        @MyRegistry.register
        def f1():
            pass

        self.assertTrue(MyRegistry.get("f1") == f1)
        self.assertTrue(MyRegistry.get("f2") == f1)

        MyRegistry.register("f3")(f1)
        self.assertTrue(MyRegistry.get("f3") == f1)

        with self.assertRaises(TypeError):
            MyRegistry.register("")(int)
            MyRegistry.register(3)(int)

        with self.assertRaises(KeyError):
            MyRegistry.register("f1")
            MyRegistry.get("T")

    def test_register_partial(self):
        """Test register_partial method."""

        My_Registry = Registry("Test")

        @My_Registry.register_partial("T1", 3, b=4)
        class T1(object):
            __slots__ = ["a", "b"]

            def __init__(self, a, b=2):
                self.a = a
                self.b = b

        inst = My_Registry.get("T1")()
        self.assertTrue(inst.a == 3 and inst.b == 4)

        My_Registry.register_partial("T2", b=2)(T1)
        inst = My_Registry.get("T2")(1)
        self.assertTrue(inst.a == 1 and inst.b == 2)

        @My_Registry.register_partial("f1", 3, b=4)
        def f1(a, b=2):
            return a + b

        res = My_Registry.get("f1")()
        self.assertTrue(res == f1(3, 4))

        My_Registry.register_partial("f2", 3)(f1)
        self.assertTrue(My_Registry.get("f2")() == f1(3))
