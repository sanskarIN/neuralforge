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
    classify_gradient_health,
    gradient_flow_report,
    gradient_statistics,
    mlp_parameter_groups,
    relative_update_ratio,
)
from neuralforge.nn import MLP  # noqa: E402


class GradientFlowTests(unittest.TestCase):
    def test_gradient_statistics_known_vector(self) -> None:
        first = Value(1.0)
        second = Value(2.0)
        first.grad = 3.0
        second.grad = 4.0
        stats = gradient_statistics([first, second])
        self.assertEqual(stats.count, 2)
        self.assertEqual(stats.finite_count, 2)
        self.assertEqual(stats.nonfinite_count, 0)
        self.assertAlmostEqual(stats.mean_abs, 3.5)
        self.assertAlmostEqual(stats.l2_norm, 5.0)
        self.assertAlmostEqual(stats.max_abs, 4.0)

    def test_zero_and_vanishing_classification(self) -> None:
        zero = Value(1.0)
        zero.grad = 0.0
        self.assertEqual(classify_gradient_health(gradient_statistics([zero])), "zero")

        tiny = Value(1.0)
        tiny.grad = 1e-10
        self.assertEqual(
            classify_gradient_health(
                gradient_statistics([tiny]), vanishing_threshold=1e-8
            ),
            "vanishing",
        )

    def test_exploding_and_nonfinite_classification(self) -> None:
        huge = Value(1.0)
        huge.grad = 150.0
        self.assertEqual(
            classify_gradient_health(
                gradient_statistics([huge]), exploding_threshold=100.0
            ),
            "exploding",
        )

        invalid = Value(1.0)
        invalid.grad = math.inf
        stats = gradient_statistics([invalid])
        self.assertEqual(stats.nonfinite_count, 1)
        self.assertEqual(classify_gradient_health(stats), "nonfinite")

    def test_named_flow_report_prioritizes_critical_status(self) -> None:
        healthy = Value(1.0)
        exploding = Value(1.0)
        healthy.grad = 0.5
        exploding.grad = 200.0
        report = gradient_flow_report(
            {"encoder": [healthy], "head": [exploding]},
            exploding_threshold=100.0,
        )
        self.assertEqual(report.overall_status, "exploding")
        self.assertEqual(report.layers[0].name, "encoder")
        self.assertEqual(report.layers[1].status, "exploding")

    def test_relative_update_ratio(self) -> None:
        first = Value(3.0)
        second = Value(4.0)
        first.grad = 0.3
        second.grad = 0.4
        self.assertAlmostEqual(relative_update_ratio([first, second], 0.1), 0.01)

    def test_mlp_parameter_groups_and_real_backprop(self) -> None:
        model = MLP(2, [3, 1], hidden_activation="tanh", seed=8)
        output = model([1.0, -0.5])
        self.assertIsInstance(output, Value)
        loss = (output - 0.7) ** 2
        loss.backward()

        groups = mlp_parameter_groups(model)
        self.assertEqual(tuple(groups), ("layer_0", "layer_1"))
        report = gradient_flow_report(groups, vanishing_threshold=1e-14, exploding_threshold=100.0)
        self.assertFalse(report.has_nonfinite)
        self.assertTrue(any(layer.stats.max_abs > 0.0 for layer in report.layers))

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            gradient_statistics([])
        with self.assertRaises(ValueError):
            classify_gradient_health(gradient_statistics([Value(1.0)]), vanishing_threshold=0.0)
        with self.assertRaises(ValueError):
            gradient_flow_report({})
        with self.assertRaises(ValueError):
            relative_update_ratio([Value(1.0)], 0.0)
        with self.assertRaises(TypeError):
            mlp_parameter_groups(object())


if __name__ == "__main__":
    unittest.main()
