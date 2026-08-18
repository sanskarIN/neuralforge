from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.tensor_basics import flatten, infer_shape, numel, reshape  # noqa: E402


class TensorBasicsTests(unittest.TestCase):
    def test_infer_shape_for_scalar_vector_and_matrix(self) -> None:
        self.assertEqual(infer_shape(3.0), ())
        self.assertEqual(infer_shape([1, 2, 3]), (3,))
        self.assertEqual(infer_shape([[1, 2, 3], [4, 5, 6]]), (2, 3))

    def test_ragged_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            infer_shape([[1, 2], [3]])

    def test_flatten_uses_row_major_order(self) -> None:
        self.assertEqual(flatten([[1, 2], [3, 4]]), [1, 2, 3, 4])

    def test_numel_counts_elements(self) -> None:
        self.assertEqual(numel(7), 1)
        self.assertEqual(numel([[1, 2], [3, 4], [5, 6]]), 6)
        self.assertEqual(numel([]), 0)

    def test_reshape_preserves_row_major_order(self) -> None:
        result = reshape([[1, 2, 3], [4, 5, 6]], (3, 2))
        self.assertEqual(result, [[1, 2], [3, 4], [5, 6]])

    def test_reshape_validates_element_count(self) -> None:
        with self.assertRaises(ValueError):
            reshape([1, 2, 3], (2, 2))

    def test_reshape_to_scalar_requires_one_element(self) -> None:
        self.assertEqual(reshape([42], ()), 42)
        with self.assertRaises(ValueError):
            reshape([1, 2], ())


if __name__ == "__main__":
    unittest.main()
