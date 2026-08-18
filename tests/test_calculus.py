from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.calculus import (  # noqa: E402
    check_gradient,
    numerical_derivative,
    numerical_gradient,
)


class CalculusTests(unittest.TestCase):
    def test_numerical_derivative_of_square(self) -> None:
        derivative = numerical_derivative(lambda x: x * x, 3.0)
        self.assertAlmostEqual(derivative, 6.0, places=5)

    def test_numerical_derivative_of_sine(self) -> None:
        derivative = numerical_derivative(math.sin, 0.0)
        self.assertAlmostEqual(derivative, 1.0, places=7)

    def test_numerical_gradient(self) -> None:
        def objective(point):
            x, y = point
            return x * x + 3.0 * y * y

        gradient = numerical_gradient(objective, [2.0, -1.5])
        self.assertAlmostEqual(gradient[0], 4.0, places=5)
        self.assertAlmostEqual(gradient[1], -9.0, places=5)

    def test_gradient_check_passes_for_correct_gradient(self) -> None:
        def objective(point):
            x, y = point
            return x * x + x * y + 2.0 * y * y

        def analytical(point):
            x, y = point
            return [2.0 * x + y, x + 4.0 * y]

        result = check_gradient(objective, analytical, [1.25, -0.75])
        self.assertTrue(result.passed)
        self.assertLess(result.max_absolute_error, 1e-5)

    def test_gradient_check_detects_bad_gradient(self) -> None:
        result = check_gradient(
            lambda point: point[0] ** 2,
            lambda point: [0.0],
            [2.0],
        )
        self.assertFalse(result.passed)

    def test_invalid_step_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            numerical_derivative(lambda x: x, 1.0, step=0.0)
        with self.assertRaises(ValueError):
            numerical_gradient(lambda point: point[0], [1.0], step=-1.0)


if __name__ == "__main__":
    unittest.main()
