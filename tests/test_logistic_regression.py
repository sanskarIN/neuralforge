from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.logistic_regression import (  # noqa: E402
    LogisticRegression,
    train_logistic_regression,
)


class LogisticRegressionTests(unittest.TestCase):
    def test_logistic_regression_learns_or_gate(self) -> None:
        features = [[0, 0], [0, 1], [1, 0], [1, 1]]
        labels = [0, 1, 1, 1]
        result = train_logistic_regression(
            features,
            labels,
            learning_rate=0.5,
            epochs=2_000,
        )
        predictions = [result.model.predict(row) for row in features]
        self.assertEqual(predictions, labels)
        self.assertLess(result.losses[-1], result.losses[0])

    def test_probabilities_are_bounded(self) -> None:
        model = LogisticRegression((2.0, -1.0), 0.25)
        probability = model.predict_probability([1.5, 0.5])
        self.assertGreater(probability, 0.0)
        self.assertLess(probability, 1.0)

    def test_threshold_is_validated(self) -> None:
        model = LogisticRegression((1.0,), 0.0)
        with self.assertRaises(ValueError):
            model.predict([1.0], threshold=0.0)
        with self.assertRaises(ValueError):
            model.predict([1.0], threshold=1.0)

    def test_l2_training_remains_finite(self) -> None:
        result = train_logistic_regression(
            [[0.0], [1.0], [2.0], [3.0]],
            [0, 0, 1, 1],
            learning_rate=0.2,
            epochs=500,
            l2=0.1,
        )
        self.assertTrue(all(loss >= 0.0 for loss in result.losses))
        self.assertLess(result.losses[-1], result.losses[0])

    def test_invalid_training_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            train_logistic_regression([], [])
        with self.assertRaises(ValueError):
            train_logistic_regression([[1.0]], [2])
        with self.assertRaises(ValueError):
            train_logistic_regression([[1.0]], [1], l2=-0.1)


if __name__ == "__main__":
    unittest.main()
