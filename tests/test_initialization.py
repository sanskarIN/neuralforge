from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.initialization import (  # noqa: E402
    initialization_plan,
    initialize_matrix,
    population_variance,
    signal_propagation_profile,
)


class InitializationTests(unittest.TestCase):
    def test_xavier_and_he_scales(self) -> None:
        xavier = initialization_plan(100, 50, "xavier_normal")
        self.assertAlmostEqual(xavier.scale, math.sqrt(2.0 / 150.0))
        he = initialization_plan(100, 50, "he_normal")
        self.assertAlmostEqual(he.scale, math.sqrt(2.0 / 100.0))
        lecun = initialization_plan(100, 50, "lecun_normal")
        self.assertAlmostEqual(lecun.scale, 0.1)

    def test_seeded_initialization_is_reproducible(self) -> None:
        first = initialize_matrix(8, 5, scheme="xavier_uniform", seed=17)
        second = initialize_matrix(8, 5, scheme="xavier_uniform", seed=17)
        different = initialize_matrix(8, 5, scheme="xavier_uniform", seed=18)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_uniform_initializers_respect_bounds(self) -> None:
        xavier = initialization_plan(16, 8, "xavier_uniform")
        matrix = initialize_matrix(8, 16, scheme="xavier_uniform", seed=3)
        self.assertTrue(all(abs(value) <= xavier.scale for row in matrix for value in row))

        he = initialization_plan(16, 8, "he_uniform")
        matrix = initialize_matrix(8, 16, scheme="he_uniform", seed=3)
        self.assertTrue(all(abs(value) <= he.scale for row in matrix for value in row))

    def test_large_normal_sample_has_expected_variance(self) -> None:
        matrix = initialize_matrix(128, 128, scheme="he_normal", seed=11)
        values = [value for row in matrix for value in row]
        observed = population_variance(values)
        expected = 2.0 / 128.0
        self.assertAlmostEqual(observed, expected, delta=expected * 0.12)

    def test_zero_initialization_collapses_signal(self) -> None:
        rng = random.Random(5)
        batch = [[rng.gauss(0.0, 1.0) for _ in range(12)] for _ in range(24)]
        profile = signal_propagation_profile(
            batch,
            [10, 8, 4],
            scheme="zeros",
            activation="relu",
            seed=5,
        )
        self.assertGreater(profile.variances[0], 0.0)
        self.assertEqual(profile.variances[1:], (0.0, 0.0, 0.0))
        self.assertEqual(profile.variance_ratio, 0.0)

    def test_he_relu_profile_remains_nonzero_and_reproducible(self) -> None:
        rng = random.Random(7)
        batch = [[rng.gauss(0.0, 1.0) for _ in range(32)] for _ in range(64)]
        first = signal_propagation_profile(
            batch,
            [32, 32, 32],
            scheme="he_normal",
            activation="relu",
            seed=19,
        )
        second = signal_propagation_profile(
            batch,
            [32, 32, 32],
            scheme="he_normal",
            activation="relu",
            seed=19,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.widths, (32, 32, 32, 32))
        self.assertTrue(all(variance > 0.0 for variance in first.variances))
        self.assertTrue(math.isfinite(first.variance_ratio))

    def test_population_variance(self) -> None:
        self.assertAlmostEqual(population_variance([1, 2, 3]), 2.0 / 3.0)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            initialization_plan(0, 2, "he_normal")
        with self.assertRaises(ValueError):
            initialization_plan(2, 2, "unknown")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            initialize_matrix(2, 2, seed=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            signal_propagation_profile([[1.0], [1.0, 2.0]], [2], scheme="he_normal", activation="relu")
        with self.assertRaises(ValueError):
            signal_propagation_profile([[1.0]], [], scheme="he_normal", activation="relu")


if __name__ == "__main__":
    unittest.main()
