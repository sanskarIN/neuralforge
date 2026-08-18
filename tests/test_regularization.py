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
from neuralforge.regularization import (  # noqa: E402
    EarlyStopping,
    generalization_gap,
    inverted_dropout,
    l1_penalty,
    l2_penalty,
    parameter_l2_norm,
    regularized_loss,
)


class RegularizationTests(unittest.TestCase):
    def test_l1_penalty_value_and_gradient(self) -> None:
        first = Value(3.0)
        second = Value(-4.0)
        penalty = l1_penalty([first, second], strength=0.1)
        self.assertAlmostEqual(penalty.data, 0.7)
        penalty.backward()
        self.assertAlmostEqual(first.grad, 0.1)
        self.assertAlmostEqual(second.grad, -0.1)

    def test_l2_penalty_value_and_gradient(self) -> None:
        first = Value(3.0)
        second = Value(-4.0)
        penalty = l2_penalty([first, second], strength=0.2)
        self.assertAlmostEqual(penalty.data, 2.5)
        penalty.backward()
        self.assertAlmostEqual(first.grad, 0.6)
        self.assertAlmostEqual(second.grad, -0.8)

    def test_regularized_loss_combines_terms(self) -> None:
        parameter = Value(2.0)
        base = Value(1.0)
        combined = regularized_loss(base, [parameter], l1=0.1, l2=0.2)
        self.assertAlmostEqual(combined.data, 1.6)

    def test_dropout_is_seeded_and_inverted(self) -> None:
        values = [Value(float(index + 1)) for index in range(12)]
        first = inverted_dropout(values, drop_probability=0.5, seed=17)
        second = inverted_dropout(values, drop_probability=0.5, seed=17)
        self.assertEqual(first.kept, second.kept)
        self.assertEqual(first.scale, 2.0)
        self.assertTrue(any(first.kept))
        self.assertTrue(any(not flag for flag in first.kept))

        for original, output, kept in zip(values, first.outputs, first.kept):
            expected = original.data * 2.0 if kept else 0.0
            self.assertAlmostEqual(output.data, expected)

    def test_dropout_gradient_matches_inverted_scale(self) -> None:
        values = [Value(1.0), Value(2.0), Value(3.0), Value(4.0)]
        result = inverted_dropout(values, drop_probability=0.5, seed=5)
        total = sum(result.outputs, Value(0.0))
        total.backward()
        for value, kept in zip(values, result.kept):
            self.assertAlmostEqual(value.grad, result.scale if kept else 0.0)

    def test_dropout_evaluation_is_identity(self) -> None:
        values = [Value(1.0), Value(2.0)]
        result = inverted_dropout(
            values,
            drop_probability=0.9,
            training=False,
            seed=1,
        )
        self.assertEqual(result.outputs, tuple(values))
        self.assertEqual(result.kept, (True, True))
        self.assertEqual(result.scale, 1.0)

    def test_early_stopping(self) -> None:
        stopper = EarlyStopping(patience=2, min_delta=0.01)
        self.assertFalse(stopper.update(1.0))
        self.assertFalse(stopper.update(0.95))
        self.assertFalse(stopper.update(0.945))
        self.assertTrue(stopper.update(0.944))
        self.assertTrue(stopper.stopped)
        self.assertAlmostEqual(stopper.best, 0.95)

        stopper.reset()
        self.assertIsNone(stopper.best)
        self.assertFalse(stopper.stopped)

    def test_generalization_gap_and_parameter_norm(self) -> None:
        self.assertAlmostEqual(generalization_gap(0.2, 0.35), 0.15)
        self.assertAlmostEqual(parameter_l2_norm([Value(3.0), Value(4.0)]), 5.0)

    def test_invalid_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            l1_penalty([], strength=0.1)
        with self.assertRaises(ValueError):
            l2_penalty([Value(1.0)], strength=-0.1)
        with self.assertRaises(ValueError):
            inverted_dropout([Value(1.0)], drop_probability=1.0)
        with self.assertRaises(ValueError):
            EarlyStopping(patience=0)
        with self.assertRaises(ValueError):
            generalization_gap(math.inf, 1.0)


if __name__ == "__main__":
    unittest.main()
