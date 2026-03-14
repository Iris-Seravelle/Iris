import array
import math
import unittest

import iris.jit as jit


class TestQuantumHeavyMath(unittest.TestCase):
    def setUp(self) -> None:
        self._prev_quantum = jit.get_quantum_speculation()
        self._prev_spec_threshold = jit.get_quantum_speculation_threshold()
        self._prev_log_threshold = jit.get_quantum_log_threshold()
        self._prev_budget = jit.get_quantum_compile_budget()
        self._prev_cooldown = jit.get_quantum_cooldown()

        if self._prev_budget == (0, 0) and self._prev_cooldown == (0, 0):
            self.skipTest("pyo3 quantum control APIs are not available in this environment")

    def tearDown(self) -> None:
        jit.set_quantum_speculation(self._prev_quantum)
        jit.set_quantum_speculation_threshold(self._prev_spec_threshold)
        jit.set_quantum_log_threshold(self._prev_log_threshold)
        jit.set_quantum_compile_budget(*self._prev_budget)
        jit.set_quantum_cooldown(*self._prev_cooldown)

    def test_quantum_control_knobs_roundtrip(self) -> None:
        jit.set_quantum_speculation(True)

        budget = jit.set_quantum_compile_budget(5_000_000, 1_000_000_000)
        cooldown = jit.set_quantum_cooldown(1_000, 50_000)
        spec_threshold = jit.set_quantum_speculation_threshold(0)
        log_threshold = jit.set_quantum_log_threshold(0)

        self.assertEqual(budget, (5_000_000, 1_000_000_000))
        self.assertEqual(jit.get_quantum_compile_budget(), (5_000_000, 1_000_000_000))
        self.assertEqual(cooldown, (1_000, 50_000))
        self.assertEqual(jit.get_quantum_cooldown(), (1_000, 50_000))
        self.assertEqual(spec_threshold, 0)
        self.assertEqual(log_threshold, 0)

    def test_quantum_heavy_math_vectorized_correctness(self) -> None:
        jit.set_quantum_speculation(True)
        jit.set_quantum_speculation_threshold(0)
        jit.set_quantum_log_threshold(0)
        jit.set_quantum_compile_budget(10_000_000, 1_000_000_000)
        jit.set_quantum_cooldown(0, 0)

        @jit.offload(strategy="jit", return_type="float")
        def heavy_math(a, b, c, d):
            return (
                (a * a + b * b + c * c + d * d) / (a + b + c + d + 1.0)
                + math.sin(a)
                + math.cos(b)
                + math.exp(c * 0.001)
                - math.log(d + 1.0)
            )

        scalar_out = heavy_math(2.0, 3.0, 4.0, 5.0)
        scalar_expected = (
            (2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0 + 5.0 * 5.0) / (2.0 + 3.0 + 4.0 + 5.0 + 1.0)
            + math.sin(2.0)
            + math.cos(3.0)
            + math.exp(4.0 * 0.001)
            - math.log(5.0 + 1.0)
        )
        self.assertAlmostEqual(float(scalar_out), float(scalar_expected), places=8)

        size = 4096
        a = array.array("d", (0.25 + i * 0.01 for i in range(size)))
        b = array.array("d", (0.75 + i * 0.01 for i in range(size)))
        c = array.array("d", (1.25 + i * 0.01 for i in range(size)))
        d = array.array("d", (1.75 + i * 0.01 for i in range(size)))

        out = heavy_math(a, b, c, d)
        self.assertEqual(len(out), size)

        for idx in (0, 17, 128, 1024, 2048, 4095):
            expected = (
                (a[idx] * a[idx] + b[idx] * b[idx] + c[idx] * c[idx] + d[idx] * d[idx])
                / (a[idx] + b[idx] + c[idx] + d[idx] + 1.0)
                + math.sin(a[idx])
                + math.cos(b[idx])
                + math.exp(c[idx] * 0.001)
                - math.log(d[idx] + 1.0)
            )
            self.assertAlmostEqual(float(out[idx]), float(expected), places=7)


if __name__ == "__main__":
    unittest.main()
