from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.foundations import (  # noqa: E402
    LogisticNeuron,
    binary_cross_entropy,
    sigmoid,
    train_logistic_neuron,
)


class FoundationsTests(unittest.TestCase):
    def test_sigmoid_is_stable_for_large_values(self) -> None:
        self.assertAlmostEqual(sigmoid(0.0), 0.5)
        self.assertGreater(sigmoid(1_000.0), 0.999)
        self.assertLess(sigmoid(-1_000.0), 0.001)

    def test_binary_cross_entropy_prefers_correct_confidence(self) -> None:
        good = binary_cross_entropy(1, 0.9)
        bad = binary_cross_entropy(1, 0.1)
        self.assertLess(good, bad)
        self.assertTrue(math.isfinite(binary_cross_entropy(1, 1.0)))

    def test_logistic_neuron_validates_feature_width(self) -> None:
        neuron = LogisticNeuron((1.0, -1.0), 0.25)
        with self.assertRaises(ValueError):
            neuron.predict_proba([1.0])

    def test_training_learns_or_gate(self) -> None:
        features = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
        targets = [0, 1, 1, 1]

        result = train_logistic_neuron(
            features,
            targets,
            learning_rate=0.5,
            epochs=2_000,
        )

        predictions = [result.neuron.predict(row) for row in features]
        self.assertEqual(predictions, targets)
        self.assertLess(result.losses[-1], result.losses[0])

    def test_training_rejects_invalid_dataset(self) -> None:
        with self.assertRaises(ValueError):
            train_logistic_neuron([], [])
        with self.assertRaises(ValueError):
            train_logistic_neuron([[1.0]], [0, 1])
        with self.assertRaises(ValueError):
            train_logistic_neuron([[1.0], [1.0, 2.0]], [0, 1])


if __name__ == "__main__":
    unittest.main()
