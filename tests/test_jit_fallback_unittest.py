import unittest

import iris.jit as jit_mod


class TestJitFallback(unittest.TestCase):
    def test_jit_panic_runtime_error_falls_back(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload

        try:
            def fake_call_jit(_func, _args, _kwargs):
                raise RuntimeError("jit panic: simulated")

            jit_mod.call_jit = fake_call_jit
            jit_mod.register_offload = lambda *a, **k: None

            @jit_mod.offload(strategy="jit")
            def add1(x):
                return x + 1

            self.assertEqual(add1(41), 42)
        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload


if __name__ == "__main__":
    unittest.main()
