from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.eda import (  # noqa: E402
    describe,
    histogram,
    iqr_outlier_mask,
    quantile,
    scatter_svg,
)


class EdaTests(unittest.TestCase):
    def test_quantiles_and_summary(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(quantile(values, 0.0), 1.0)
        self.assertEqual(quantile(values, 0.5), 3.0)
        self.assertEqual(quantile(values, 1.0), 5.0)

        summary = describe(values)
        self.assertEqual(summary.count, 5)
        self.assertEqual(summary.minimum, 1.0)
        self.assertEqual(summary.median, 3.0)
        self.assertEqual(summary.maximum, 5.0)
        self.assertEqual(summary.mean, 3.0)
        self.assertEqual(summary.iqr, 2.0)

    def test_iqr_outlier_mask(self) -> None:
        mask = iqr_outlier_mask([1, 2, 3, 4, 100])
        self.assertEqual(mask, (False, False, False, False, True))

    def test_histogram_counts_every_value_once(self) -> None:
        result = histogram([0, 1, 2, 3, 4, 5], bins=3)
        self.assertEqual(result.total, 6)
        self.assertEqual(len(result.edges), 4)
        self.assertEqual(len(result.counts), 3)

    def test_histogram_handles_constant_data(self) -> None:
        result = histogram([7, 7, 7], bins=4)
        self.assertEqual(result.total, 3)
        self.assertEqual(sum(result.counts), 3)

    def test_scatter_svg_contains_points_and_escapes_title(self) -> None:
        svg = scatter_svg([1, 2, 3], [4, 5, 6], title="Loss < Accuracy")
        self.assertTrue(svg.startswith("<svg"))
        self.assertEqual(svg.count("<circle"), 3)
        self.assertIn("Loss &lt; Accuracy", svg)

    def test_scatter_svg_rejects_mismatched_data(self) -> None:
        with self.assertRaises(ValueError):
            scatter_svg([1, 2], [1])

    def test_invalid_quantile_and_bins_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            quantile([1, 2, 3], 1.5)
        with self.assertRaises(ValueError):
            histogram([1, 2, 3], bins=0)


if __name__ == "__main__":
    unittest.main()
