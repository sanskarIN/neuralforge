from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.linear_algebra import (  # noqa: E402
    cosine_similarity,
    dot,
    l2_norm,
    matmul,
    outer,
    transpose,
)


class LinearAlgebraTests(unittest.TestCase):
    def test_dot_product(self) -> None:
        self.assertEqual(dot([1, 2, 3], [4, 5, 6]), 32.0)
        with self.assertRaises(ValueError):
            dot([1, 2], [1])

    def test_l2_norm(self) -> None:
        self.assertEqual(l2_norm([3, 4]), 5.0)

    def test_cosine_similarity(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(cosine_similarity([1, 1], [-1, -1]), -1.0)
        with self.assertRaises(ValueError):
            cosine_similarity([0, 0], [1, 0])

    def test_transpose(self) -> None:
        self.assertEqual(
            transpose([[1, 2, 3], [4, 5, 6]]),
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
        )

    def test_matrix_multiplication(self) -> None:
        result = matmul(
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8], [9, 10], [11, 12]],
        )
        self.assertEqual(result, [[58.0, 64.0], [139.0, 154.0]])

    def test_matrix_multiplication_rejects_bad_shapes(self) -> None:
        with self.assertRaises(ValueError):
            matmul([[1, 2]], [[1, 2]])

    def test_outer_product(self) -> None:
        self.assertEqual(outer([1, 2], [3, 4, 5]), [[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]])

    def test_non_finite_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            l2_norm([1.0, math.inf])


if __name__ == "__main__":
    unittest.main()
