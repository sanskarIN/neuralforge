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
    huber_loss,
    mean_absolute_error,
    mean_squared_error,
    multiclass_cross_entropy,
    recommended_output_loss,
    softplus,
)


class LossTests(unittest.TestCase):
    def test_softplus_is_stable_for_large_magnitudes(self) -> None:
        positive = Value(1000.0)
        negative = Value(-1000.0)
        self.assertAlmostEqual(softplus(positive).data, 1000.0, places=10)
        self.assertAlmostEqual(softplus(negative).data, 0.0, places=10)

    def test_mse_value_and_gradient(self) -> None:
        prediction = Value(3.0)
        loss = mean_squared_error([prediction], [1.0])
        self.assertAlmostEqual(loss.data, 4.0)
        loss.backward()
        self.assertAlmostEqual(prediction.grad, 4.0)

    def test_mae_value_and_gradient(self) -> None:
        high = Value(3.0)
        low = Value(-2.0)
        loss = mean_absolute_error([high, low], [1.0, -1.0])
        self.assertAlmostEqual(loss.data, 1.5)
        loss.backward()
        self.assertAlmostEqual(high.grad, 0.5)
        self.assertAlmostEqual(low.grad, -0.5)

    def test_huber_uses_quadratic_and_linear_regions(self) -> None:
        near = Value(0.5)
        far = Value(3.0)
        loss = huber_loss([near, far], [0.0, 0.0], delta=1.0)
        expected = (0.5 * 0.5**2 + (3.0 - 0.5)) / 2.0
        self.assertAlmostEqual(loss.data, expected)
        loss.backward()
        self.assertAlmostEqual(near.grad, 0.25)
        self.assertAlmostEqual(far.grad, 0.5)

    def test_bce_with_logits_matches_known_zero_logit_loss(self) -> None:
        logit = Value(0.0)
        loss = binary_cross_entropy_with_logits([logit], [1])
        self.assertAlmostEqual(loss.data, math.log(2.0))
        loss.backward()
        self.assertAlmostEqual(logit.grad, -0.5)

    def test_bce_with_logits_remains_finite_for_extreme_logits(self) -> None:
        positive = Value(1000.0)
        negative = Value(-1000.0)
        good = binary_cross_entropy_with_logits([positive, negative], [1, 0])
        self.assertTrue(math.isfinite(good.data))
        self.assertLess(good.data, 1e-9)

    def test_multiclass_cross_entropy_value_and_gradients(self) -> None:
        row = [Value(1.0), Value(2.0), Value(3.0)]
        loss = multiclass_cross_entropy([row], [2])
        expected = math.log(math.exp(-2.0) + math.exp(-1.0) + 1.0)
        self.assertAlmostEqual(loss.data, expected)
        loss.backward()
        self.assertAlmostEqual(sum(value.grad for value in row), 0.0, places=12)
        self.assertLess(row[2].grad, 0.0)
        self.assertGreater(row[0].grad, 0.0)

    def test_multiclass_cross_entropy_is_shift_invariant(self) -> None:
        first = multiclass_cross_entropy([[Value(1.0), Value(2.0), Value(3.0)]], [1])
        shifted = multiclass_cross_entropy([[Value(1001.0), Value(1002.0), Value(1003.0)]], [1])
        self.assertAlmostEqual(first.data, shifted.data)

    def test_output_loss_pairings(self) -> None:
        binary = recommended_output_loss("binary_classification")
        self.assertEqual(binary.output_activation, "linear logits")
        self.assertEqual(binary.loss, "binary_cross_entropy_with_logits")
        multiclass = recommended_output_loss("multiclass_classification")
        self.assertIn("logit", multiclass.output_activation)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            mean_squared_error([], [])
        with self.assertRaises(ValueError):
            mean_absolute_error([Value(1.0)], [1.0, 2.0])
        with self.assertRaises(ValueError):
            huber_loss([Value(1.0)], [0.0], delta=0.0)
        with self.assertRaises(ValueError):
            binary_cross_entropy_with_logits([Value(1.0)], [2])
        with self.assertRaises(ValueError):
            multiclass_cross_entropy([[Value(1.0), Value(2.0)]], [2])
        with self.assertRaises(ValueError):
            recommended_output_loss("unknown")


if __name__ == "__main__":
    unittest.main()
