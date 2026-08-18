from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.statistics import (  # noqa: E402
    bernoulli_log_likelihood,
    bootstrap_mean_interval,
    correlation,
    covariance,
    mean,
    normal_pdf,
    standard_deviation,
    variance,
)


class StatisticsTests(unittest.TestCase):
    def test_mean_and_variance(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(mean(values), 2.5)
        self.assertAlmostEqual(variance(values), 1.25)
        self.assertAlmostEqual(variance(values, sample=True), 5.0 / 3.0)
        self.assertAlmostEqual(standard_deviation(values), math.sqrt(1.25))

    def test_covariance_and_correlation(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        y = [2.0, 4.0, 6.0, 8.0]
        self.assertAlmostEqual(covariance(x, y), 2.5)
        self.assertAlmostEqual(correlation(x, y), 1.0)
        self.assertAlmostEqual(correlation(x, list(reversed(y))), -1.0)

    def test_correlation_rejects_constant_variable(self) -> None:
        with self.assertRaises(ValueError):
            correlation([1, 1, 1], [1, 2, 3])

    def test_normal_pdf(self) -> None:
        self.assertAlmostEqual(normal_pdf(0.0), 1.0 / math.sqrt(2.0 * math.pi))
        with self.assertRaises(ValueError):
            normal_pdf(0.0, std=0.0)

    def test_bernoulli_log_likelihood(self) -> None:
        value = bernoulli_log_likelihood([1, 1, 0, 1], 0.75)
        expected = 3 * math.log(0.75) + math.log(0.25)
        self.assertAlmostEqual(value, expected)

    def test_bootstrap_interval_is_reproducible(self) -> None:
        first = bootstrap_mean_interval([1, 2, 3, 4, 5], resamples=500, seed=7)
        second = bootstrap_mean_interval([1, 2, 3, 4, 5], resamples=500, seed=7)
        self.assertEqual(first, second)
        self.assertLessEqual(first.lower, first.observed)
        self.assertGreaterEqual(first.upper, first.observed)

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            mean([])
        with self.assertRaises(ValueError):
            variance([1.0], sample=True)
        with self.assertRaises(ValueError):
            covariance([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            bernoulli_log_likelihood([0, 2], 0.5)


if __name__ == "__main__":
    unittest.main()
