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
from neuralforge.losses import (  # noqa: E402
    binary_cross_entropy_with_logits,
    categorical_cross_entropy_with_logits,
    huber_loss,
    logsumexp,
    mean_absolute_error,
    mean_categorical_cross_entropy_with_logits,
    mean_squared_error,
    recommend_output_design,
    softplus,
)


class LossTests(unittest.TestCase):
    def test_regression_losses(self) -> None:
        predictions = [Value(1.0), Value(3.0)]
        targets = [0.0, 1.0]
        self.assertAlmostEqual(mean_squared_error(predictions, targets).data, 2.5)
        self.assertAlmostEqual(mean_absolute_error(predictions, targets).data, 1.5)
        self.assertAlmostEqual(huber_loss(predictions, targets, delta=1.0).data, 1.0)

    def test_huber_is_quadratic_near_zero(self) -> None:
        prediction = Value(0.25)
        loss = huber_loss([prediction], [0.0], delta=1.0)
        self.assertAlmostEqual(loss.data, 0.03125)
        loss.backward()
        self.assertAlmostEqual(prediction.grad, 0.25)

    def test_softplus_is_stable_for_large_magnitudes(self) -> None:
        positive = Value(1000.0)
        negative = Value(-1000.0)
        self.assertTrue(math.isfinite(softplus(positive).data))
        self.assertTrue(math.isfinite(softplus(negative).data))
        self.assertAlmostEqual(softplus(positive).data, 1000.0, places=9)
        self.assertAlmostEqual(softplus(negative).data, 0.0, places=12)

    def test_binary_cross_entropy_logits_has_correct_gradient(self) -> None:
        logit = Value(0.0)
        loss = binary_cross_entropy_with_logits([logit], [1])
        self.assertAlmostEqual(loss.data, math.log(2.0))
        loss.backward()
        self.assertAlmostEqual(logit.grad, -0.5)

    def test_binary_cross_entropy_logits_handles_extreme_logits(self) -> None:
        loss = binary_cross_entropy_with_logits([Value(1000.0), Value(-1000.0)], [1, 0])
        self.assertTrue(math.isfinite(loss.data))
        self.assertLess(loss.data, 1e-10)

    def test_logsumexp_is_stable(self) -> None:
        value = logsumexp([Value(1000.0), Value(1000.0)])
        self.assertAlmostEqual(value.data, 1000.0 + math.log(2.0), places=9)

    def test_categorical_cross_entropy_gradient_sums_to_zero(self) -> None:
        logits = [Value(1.0), Value(2.0), Value(3.0)]
        loss = categorical_cross_entropy_with_logits(logits, 2)
        loss.backward()
        self.assertAlmostEqual(sum(item.grad for item in logits), 0.0, places=10)
        self.assertLess(logits[2].grad, 0.0)

    def test_mean_categorical_cross_entropy(self) -> None:
        loss = mean_categorical_cross_entropy_with_logits(
            [[Value(4.0), Value(1.0)], [Value(-1.0), Value(2.0)]],
            [0, 1],
        )
        self.assertGreater(loss.data, 0.0)
        self.assertLess(loss.data, 0.1)

    def test_output_design_recommendations(self) -> None:
        regression = recommend_output_design("regression")
        self.assertEqual(regression.final_activation, "linear")

        binary = recommend_output_design("binary classification")
        self.assertEqual(binary.objective, "binary_cross_entropy_with_logits")

        multiclass = recommend_output_design("multiclass", classes=5)
        self.assertEqual(multiclass.output_units, 5)
        self.assertEqual(multiclass.final_activation, "linear_logits")

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            mean_squared_error([], [])
        with self.assertRaises(ValueError):
            mean_absolute_error([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            huber_loss([1.0], [0.0], delta=0.0)
        with self.assertRaises(ValueError):
            binary_cross_entropy_with_logits([0.0], [2])
        with self.assertRaises(ValueError):
            categorical_cross_entropy_with_logits([1.0, 2.0], 2)
        with self.assertRaises(ValueError):
            recommend_output_design("multiclass", classes=1)
        with self.assertRaises(ValueError):
            recommend_output_design("segmentation")


if __name__ == "__main__":
    unittest.main()
