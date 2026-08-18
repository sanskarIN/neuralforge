from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from numpy_vectorization import (  # noqa: E402
    as_feature_matrix,
    dense_layer,
    softmax,
    standardize_features,
)


class NumPyVectorizationTests(unittest.TestCase):
    def test_feature_matrix_validation(self) -> None:
        matrix = as_feature_matrix([[1, 2], [3, 4]])
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(matrix.dtype, np.float64)
        with self.assertRaises(ValueError):
            as_feature_matrix([1, 2, 3])

    def test_standardization_centers_variable_columns(self) -> None:
        standardized, mean, scale = standardize_features(
            [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]
        )
        np.testing.assert_allclose(mean, [2.0, 5.0])
        self.assertEqual(scale[1], 1.0)
        np.testing.assert_allclose(standardized.mean(axis=0), [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(standardized[:, 1], [0.0, 0.0, 0.0])

    def test_dense_layer_shape_and_values(self) -> None:
        output = dense_layer(
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 0.0, -1.0], [0.5, 2.0, 1.0]],
            [0.25, -0.5, 1.5],
        )
        expected = np.array(
            [[2.25, 3.5, 2.5], [5.25, 7.5, 2.5]],
            dtype=np.float64,
        )
        np.testing.assert_allclose(output, expected)

    def test_softmax_is_stable_and_normalized(self) -> None:
        probabilities = softmax([[1_000.0, 1_001.0, 1_002.0], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(probabilities.sum(axis=1), [1.0, 1.0])
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities[1], [1 / 3, 1 / 3, 1 / 3])

    def test_dense_layer_rejects_incompatible_shapes(self) -> None:
        with self.assertRaises(ValueError):
            dense_layer([[1.0, 2.0]], [[1.0], [2.0], [3.0]], [0.0])


if __name__ == "__main__":
    unittest.main()
