import unittest
import array

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

    def test_complex_body_skips_jit_wrapper(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload

        try:
            def fake_call_jit(_func, _args, _kwargs):
                raise AssertionError("call_jit should not run for complex function body")

            jit_mod.call_jit = fake_call_jit
            jit_mod.register_offload = lambda *a, **k: None

            @jit_mod.offload(strategy="jit")
            def prng_like(seed, n):
                numbers = []
                x = seed
                for i in range(n):
                    if i % 2 == 0:
                        x += 0.1
                    x = (x + i * 0.1) % 1.0
                    numbers.append(x)
                return numbers

            out = prng_like(0.25, 3)
            self.assertEqual(len(out), 3)
            self.assertAlmostEqual(out[0], 0.35)
        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload

    def test_stateful_loop_uses_step_jit(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload

        calls = {"jit": 0, "register": 0}

        try:
            def fake_call_jit(_func, _args, _kwargs):
                calls["jit"] += 1
                return _func(*_args)

            def fake_register_offload(*_args, **_kwargs):
                calls["register"] += 1
                return None

            jit_mod.call_jit = fake_call_jit
            jit_mod.register_offload = fake_register_offload

            @jit_mod.offload(strategy="jit")
            def accum(seed, n):
                numbers = []
                x = seed
                for i in range(n):
                    x = x + i + 1
                    numbers.append(x)
                return numbers

            out = accum(0.0, 3)
            self.assertEqual(out, [1.0, 3.0, 6.0])
            self.assertGreaterEqual(calls["register"], 1)
            self.assertEqual(calls["jit"], 3)
        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload

    def test_complex_body_array_inputs_vectorized_fallback(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload

        try:
            jit_mod.call_jit = lambda _func, _args, _kwargs: (_ for _ in ()).throw(
                RuntimeError("no JIT entry found")
            )
            jit_mod.register_offload = lambda *a, **k: None

            @jit_mod.offload(strategy="jit", return_type="float")
            def branch_like(price, vol, strike):
                x = price / strike
                return x + vol

            prices = array.array("d", [100.0, 101.0, 102.0])
            vols = array.array("d", [0.2, 0.2, 0.2])
            strikes = array.array("d", [105.0, 105.0, 105.0])

            out = branch_like(prices, vols, strikes)
            self.assertEqual(len(out), 3)
            self.assertAlmostEqual(float(out[0]), (100.0 / 105.0) + 0.2)
            self.assertAlmostEqual(float(out[2]), (102.0 / 105.0) + 0.2)
        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload

    def test_complex_body_vector_inputs_use_aggressive_jit(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload

        calls = {"jit": 0, "register": 0}

        try:
            def fake_call_jit(_func, _args, _kwargs):
                calls["jit"] += 1
                if isinstance(_args[0], (int, float)):
                    return (_args[0] / _args[2]) + _args[1]
                # mimic old aggressive mode behavior for vector path
                return _args[0]

            def fake_register_offload(*_args, **_kwargs):
                calls["register"] += 1
                return None

            jit_mod.call_jit = fake_call_jit
            jit_mod.register_offload = fake_register_offload

            @jit_mod.offload(strategy="jit", return_type="float")
            def branch_like(price, vol, strike):
                x = price / strike
                return x + vol

            scalar_out = branch_like(100.0, 0.2, 105.0)
            self.assertAlmostEqual(float(scalar_out), (100.0 / 105.0) + 0.2)

            prices = array.array("d", [100.0, 101.0, 102.0])
            vols = array.array("d", [0.2, 0.2, 0.2])
            strikes = array.array("d", [105.0, 105.0, 105.0])
            out = branch_like(prices, vols, strikes)
            self.assertEqual(len(out), 3)
            self.assertGreaterEqual(calls["register"], 1)
            self.assertEqual(calls["jit"], 2)
        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload

    def test_inlined_assignments_use_scalar_jit_path(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload

        calls = {"jit": 0, "register": 0}
        seen = {"src": None}

        try:
            def fake_call_jit(_func, _args, _kwargs):
                calls["jit"] += 1
                return 123.0

            def fake_register_offload(_func, _strategy, _return_type, src, _arg_names):
                calls["register"] += 1
                seen["src"] = src
                return None

            jit_mod.call_jit = fake_call_jit
            jit_mod.register_offload = fake_register_offload

            @jit_mod.offload(strategy="jit", return_type="float")
            def calc(price, strike, vol):
                x = price / strike
                y = x + vol
                return y * 2

            out = calc(100.0, 105.0, 0.2)
            self.assertEqual(out, 123.0)
            self.assertGreaterEqual(calls["register"], 1)
            self.assertEqual(calls["jit"], 1)
            self.assertIsNotNone(seen["src"])

        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload

    def test_inlined_if_else_use_scalar_jit_path(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload

        calls = {"jit": 0}

        try:
            def fake_call_jit(_func, _args, _kwargs):
                calls["jit"] += 1
                return 7.0

            jit_mod.call_jit = fake_call_jit
            jit_mod.register_offload = lambda *a, **k: None

            @jit_mod.offload(strategy="jit", return_type="float")
            def branchy(x, y):
                z = x - y
                if z > 0:
                    out = z * 2
                else:
                    out = y - x
                return out + 1

            out = branchy(3.0, 2.0)
            self.assertEqual(out, 7.0)
            self.assertEqual(calls["jit"], 1)
        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload


if __name__ == "__main__":
    unittest.main()
