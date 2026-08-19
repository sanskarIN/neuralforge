from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.autograd import Value  # noqa: E402
from neuralforge.gradient_flow import (  # noqa: E402
    assess_gradient_health,
    gradient_stats,
    gradient_to_parameter_ratio,
    group_gradient_stats,
    require_finite_gradients,
)


class GradientFlowTests(unittest.TestCase):
    def test_gradient_statistics(self) -> None:
        parameters = [Value(1.0), Value(2.0), Value(3.0)]
        for parameter, gradient in zip(parameters, [3.0, 4.0, 0.0]):
            parameter.grad = gradient

        stats = gradient_stats(parameters)
        self.assertEqual(stats.count, 3)
        self.assertEqual(stats.nonfinite_count, 0)
        self.assertEqual(stats.zero_count, 1)
        self.assertAlmostEqual(stats.zero_fraction, 1.0 / 3.0)
        self.assertAlmostEqual(stats.l2_norm, 5.0)
        self.assertAlmostEqual(stats.l1_norm, 7.0)
        self.assertAlmostEqual(stats.max_abs, 4.0)
        self.assertAlmostEqual(stats.min_abs_nonzero, 3.0)

    def test_grouping_by_layer_prefix(self) -> None:
        p0 = Value(1.0, label="L0.w0")
        p1 = Value(1.0, label="L0.b")
        p2 = Value(1.0, label="L1.w0")
        p0.grad = 1.0
        p1.grad = 2.0
        p2.grad = 3.0
        grouped = group_gradient_stats([p0, p1, p2])
        self.assertEqual(set(grouped), {"L0", "L1"})
        self.assertAlmostEqual(grouped["L0"].l2_norm, math.sqrt(5.0))
        self.assertAlmostEqual(grouped["L1"].l2_norm, 3.0)

    def test_unlabeled_parameters_use_fallback_group(self) -> None:
        parameter = Value(2.0)
        parameter.grad = 1.0
        grouped = group_gradient_stats([parameter])
        self.assertIn("unlabeled", grouped)

    def test_gradient_to_parameter_ratio(self) -> None:
        parameters = [Value(3.0), Value(4.0)]
        parameters[0].grad = 0.3
        parameters[1].grad = 0.4
        self.assertAlmostEqual(gradient_to_parameter_ratio(parameters), 0.1)

    def test_health_classification(self) -> None:
        tiny = Value(1.0)
        tiny.grad = 1e-12
        self.assertEqual(assess_gradient_health([tiny], vanishing_l2=1e-10).status, "vanishing")

        large = Value(1.0)
        large.grad = 2000.0
        self.assertEqual(assess_gradient_health([large], exploding_max_abs=1000.0).status, "exploding")

        normal = Value(1.0)
        normal.grad = 0.25
        self.assertEqual(assess_gradient_health([normal]).status, "healthy")

    def test_nonfinite_gradients_are_detected(self) -> None:
        parameter = Value(1.0)
        parameter.grad = math.inf
        stats = gradient_stats([parameter])
        self.assertEqual(stats.nonfinite_count, 1)
        self.assertTrue(math.isnan(stats.l2_norm))
        self.assertEqual(assess_gradient_health([parameter]).status, "non_finite")
        with self.assertRaises(ValueError):
            require_finite_gradients([parameter])
        with self.assertRaises(ValueError):
            gradient_to_parameter_ratio([parameter])

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            gradient_stats([])
        with self.assertRaises(ValueError):
            group_gradient_stats([])
        with self.assertRaises(ValueError):
            group_gradient_stats([Value(1.0)], separator="")
        with self.assertRaises(ValueError):
            gradient_to_parameter_ratio([Value(1.0)], epsilon=0.0)
        with self.assertRaises(ValueError):
            assess_gradient_health([Value(1.0)], vanishing_l2=-1.0)
        with self.assertRaises(ValueError):
            assess_gradient_health([Value(1.0)], exploding_max_abs=0.0)


if __name__ == "__main__":
    unittest.main()
