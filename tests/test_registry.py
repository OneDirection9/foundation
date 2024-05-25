import unittest

from foundation.registry import Registry


def get_emtpy_registry() -> Registry:
    class MyRegistry(Registry): ...

    return MyRegistry()


class TestRegistry(unittest.TestCase):
    def test_register(self):
        """Test register method."""
        registry = get_emtpy_registry()

        @registry.register("T2")
        @registry.register
        class T1:
            pass

        self.assertTrue(registry("T1") == T1)
        self.assertTrue(registry("T2") == T1)

        registry.register("T3")(T1)
        self.assertTrue(registry("T3") == T1)

        @registry.register("f2")
        @registry.register
        def f1():
            pass

        self.assertTrue(registry("f1") == f1)
        self.assertTrue(registry("f2") == f1)

        registry.register("f3")(f1)
        self.assertTrue(registry("f3") == f1)

        with self.assertRaises(TypeError):
            registry.register("")(int)
            registry.register(3)(int)

        with self.assertRaises(KeyError):
            registry.register("f1")
            registry("T")

    def test_register_instance(self):
        """Test register_instance method."""
        registry = get_emtpy_registry()

        @registry.register_instance("t1_inst", 3, b=4)
        class T1:
            def __init__(self, a, b=2):
                self.a = a
                self.b = b

        inst = registry("t1_inst")
        self.assertTrue(inst.a == 3 and inst.b == 4)

    def test_independent_register(self):
        r1, r2 = get_emtpy_registry(), get_emtpy_registry()
        r1.register("a")("x")
        r2.register("a")("y")
        self.assertTrue("a" in r1)
        self.assertTrue("a" in r2)
        self.assertTrue(r1["a"] != r2["a"])
