from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.data_preparation import (  # noqa: E402
    Standardizer,
    random_split_indices,
    select_rows,
    stratified_split_indices,
)


class DataPreparationTests(unittest.TestCase):
    def test_random_split_is_reproducible_and_disjoint(self) -> None:
        first = random_split_indices(20, seed=99)
        second = random_split_indices(20, seed=99)
        self.assertEqual(first, second)
        self.assertEqual(sorted(first.all_indices), list(range(20)))
        self.assertEqual(len(first.all_indices), len(set(first.all_indices)))
        self.assertGreater(len(first.train), len(first.validation))
        self.assertGreater(len(first.train), len(first.test))

    def test_stratified_split_preserves_all_examples_without_overlap(self) -> None:
        labels = ["cat"] * 20 + ["dog"] * 20 + ["bird"] * 20
        split = stratified_split_indices(labels, seed=5)
        self.assertEqual(sorted(split.all_indices), list(range(len(labels))))
        self.assertEqual(len(split.all_indices), len(set(split.all_indices)))

        for partition in (split.train, split.validation, split.test):
            observed = {labels[index] for index in partition}
            self.assertEqual(observed, {"cat", "dog", "bird"})

    def test_standardizer_uses_only_fitted_training_state(self) -> None:
        training = [[0.0, 10.0], [2.0, 10.0], [4.0, 10.0]]
        validation = [[100.0, 10.0]]

        scaler = Standardizer.fit(training)
        transformed_training = scaler.transform(training)
        transformed_validation = scaler.transform(validation)

        first_column_mean = sum(row[0] for row in transformed_training) / 3
        self.assertAlmostEqual(first_column_mean, 0.0)
        self.assertEqual(scaler.mean, (2.0, 10.0))
        self.assertEqual(scaler.scale[1], 1.0)
        self.assertGreater(transformed_validation[0][0], 10.0)

    def test_select_rows_follows_split_indices(self) -> None:
        features = [[float(index), float(index + 1)] for index in range(6)]
        selected = select_rows(features, [4, 1, 5])
        self.assertEqual(selected, [[4.0, 5.0], [1.0, 2.0], [5.0, 6.0]])

    def test_invalid_ratios_and_shapes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            random_split_indices(10, train_ratio=0.8, validation_ratio=0.2, test_ratio=0.2)
        with self.assertRaises(ValueError):
            Standardizer.fit([[1.0, 2.0], [3.0]])
        with self.assertRaises(IndexError):
            select_rows([[1.0], [2.0]], [2])


if __name__ == "__main__":
    unittest.main()
