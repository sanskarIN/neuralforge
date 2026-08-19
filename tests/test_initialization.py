from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.initialization import (  # noqa: E402
    activate,
    dense_forward,
    initialize_matrix,
    initialization_scale,
    propagate_signal,
    recommend_initialization,
)


class InitializationTests(unittest.TestCase):
    def test_scale_formulas(self) -> None:
        self.assertAlmostEqual(initialization_scale("xavier_uniform", 4, 6), math.sqrt(6.0 / 10.0))
        self.assertAlmostEqual(initialization_scale("xavier_normal", 4, 6), math.sqrt(2.0 / 10.0))
        self.assertAlmostEqual(initialization_scale("he_uniform", 4, 6), math.sqrt(6.0 / 4.0))
        self.assertAlmostEqual(initialization_scale("he_normal", 4, 6), math.sqrt(2.0 / 4.0))
        self.assertAlmostEqual(initialization_scale("lecun_normal", 4, 6), 0.5)

    def test_recommendations(self) -> None:
        self.assertEqual(recommend_initialization("relu"), "he_normal")
        self.assertEqual(recommend_initialization("tanh"), "xavier_uniform")
        self.assertEqual(recommend_initialization("sigmoid"), "xavier_uniform")
        self.assertEqual(recommend_initialization("linear"), "xavier_uniform")

    def test_seeded_matrix_is_reproducible(self) -> None:
        first = initialize_matrix(3, 2, scheme="xavier_uniform", seed=9)
        second = initialize_matrix(3, 2, scheme="xavier_uniform", seed=9)
        other = initialize_matrix(3, 2, scheme="xavier_uniform", seed=10)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 2)
        self.assertTrue(all(len(row) == 3 for row in first))

    def test_zero_initialization(self) -> None:
        matrix = initialize_matrix(2, 3, scheme="zeros")
        self.assertEqual(matrix, ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)))

    def test_dense_forward(self) -> None:
        output = dense_forward([2.0, -1.0], [[0.5, 1.0], [2.0, -0.5]], [0.25, -1.0])
        self.assertEqual(output, (0.25, 3.5))

    def test_activations(self) -> None:
        self.assertEqual(activate([-2.0, 0.0, 3.0], "relu"), (0.0, 0.0, 3.0))
        tanh_values = activate([-1.0, 0.0, 1.0], "tanh")
        self.assertAlmostEqual(tanh_values[1], 0.0)
        sigmoid_values = activate([-1000.0, 0.0, 1000.0], "sigmoid")
        self.assertAlmostEqual(sigmoid_values[0], 0.0)
        self.assertAlmostEqual(sigmoid_values[1], 0.5)
        self.assertAlmostEqual(sigmoid_values[2], 1.0)

    def test_signal_propagation_is_reproducible_and_finite(self) -> None:
        batch = [
            [-1.0, -0.5, 0.0, 0.5],
            [0.2, 0.4, 0.6, 0.8],
            [1.0, -1.0, 1.0, -1.0],
            [-0.3, 0.7, -0.9, 0.1],
        ]
        first = propagate_signal(batch, [8, 8, 4], activation="relu", scheme="he_normal", seed=123)
        second = propagate_signal(batch, [8, 8, 4], activation="relu", scheme="he_normal", seed=123)
        self.assertEqual(first, second)
        self.assertEqual([item.width for item in first], [4, 8, 8, 4])
        for item in first:
            self.assertTrue(math.isfinite(item.mean))
            self.assertTrue(math.isfinite(item.variance))
            self.assertGreaterEqual(item.variance, 0.0)
            self.assertGreaterEqual(item.zero_fraction, 0.0)
            self.assertLessEqual(item.zero_fraction, 1.0)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            initialization_scale("he_normal", 0, 2)
        with self.assertRaises(ValueError):
            initialization_scale("unknown", 2, 2)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            recommend_initialization("gelu")
        with self.assertRaises(TypeError):
            initialize_matrix(2, 2, seed=True)
        with self.assertRaises(ValueError):
            dense_forward([1.0, 2.0], [[1.0]])
        with self.assertRaises(ValueError):
            propagate_signal([[1.0], [1.0, 2.0]], [2], activation="relu")
        with self.assertRaises(ValueError):
            propagate_signal([[1.0]], [], activation="relu")


if __name__ == "__main__":
    unittest.main()
