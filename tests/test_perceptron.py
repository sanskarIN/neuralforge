from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.perceptron import Perceptron, train_perceptron  # noqa: E402


class PerceptronTests(unittest.TestCase):
    def test_perceptron_learns_and_gate(self) -> None:
        features = [[0, 0], [0, 1], [1, 0], [1, 1]]
        labels = [0, 0, 0, 1]
        result = train_perceptron(features, labels, seed=7, epochs=100)
        predictions = [result.model.predict(row) for row in features]
        self.assertEqual(predictions, labels)
        self.assertEqual(result.mistakes_per_epoch[-1], 0)

    def test_training_is_reproducible_for_same_seed(self) -> None:
        features = [[0, 0], [0, 1], [1, 0], [1, 1]]
        labels = [0, 1, 1, 1]
        first = train_perceptron(features, labels, seed=11)
        second = train_perceptron(features, labels, seed=11)
        self.assertEqual(first, second)

    def test_model_validates_feature_width(self) -> None:
        model = Perceptron((1.0, -1.0), 0.0)
        with self.assertRaises(ValueError):
            model.predict([1.0])

    def test_invalid_training_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            train_perceptron([], [])
        with self.assertRaises(ValueError):
            train_perceptron([[1.0]], [2])
        with self.assertRaises(ValueError):
            train_perceptron([[1.0]], [1], learning_rate=0.0)


if __name__ == "__main__":
    unittest.main()
