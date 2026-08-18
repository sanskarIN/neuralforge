from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.segmentation import (  # noqa: E402
    binary_dice_score,
    binary_mask_iou,
    confusion_matrix,
    panoptic_quality,
    segmentation_report,
)


class SegmentationTests(unittest.TestCase):
    def test_confusion_matrix_and_semantic_metrics(self) -> None:
        target = ((0, 0), (1, 1))
        prediction = ((0, 1), (1, 1))
        report = segmentation_report(prediction, target, num_classes=2)

        self.assertEqual(report.confusion_matrix, ((1, 1), (0, 2)))
        self.assertAlmostEqual(report.pixel_accuracy, 0.75)
        self.assertAlmostEqual(report.per_class[0].iou, 0.5)
        self.assertAlmostEqual(report.per_class[0].dice, 2 / 3)
        self.assertAlmostEqual(report.per_class[1].iou, 2 / 3)
        self.assertAlmostEqual(report.per_class[1].dice, 0.8)
        self.assertAlmostEqual(report.mean_iou, (0.5 + 2 / 3) / 2)
        self.assertAlmostEqual(report.mean_dice, ((2 / 3) + 0.8) / 2)

    def test_ignore_index_excludes_target_and_prediction_pixel(self) -> None:
        target = ((0, -1), (1, 1))
        prediction = ((0, 99), (1, 0))
        matrix = confusion_matrix(prediction, target, ignore_index=-1)
        self.assertEqual(matrix, ((1, 0), (1, 1)))

    def test_absent_class_is_excluded_from_mean_metrics(self) -> None:
        report = segmentation_report(((0, 0), (1, 1)), ((0, 0), (1, 1)), num_classes=3)
        self.assertIsNone(report.per_class[2].iou)
        self.assertIsNone(report.per_class[2].dice)
        self.assertEqual(report.mean_iou, 1.0)
        self.assertEqual(report.mean_dice, 1.0)

    def test_binary_mask_iou_and_dice(self) -> None:
        target = ((1, 1), (0, 0))
        prediction = ((1, 0), (1, 0))
        self.assertAlmostEqual(binary_mask_iou(prediction, target), 1 / 3)
        self.assertAlmostEqual(binary_dice_score(prediction, target), 0.5)
        self.assertEqual(binary_mask_iou(((0,),), ((0,),)), 1.0)
        self.assertEqual(binary_dice_score(((0,),), ((0,),)), 1.0)

    def test_panoptic_quality_decomposition(self) -> None:
        metrics = panoptic_quality(
            matched_iou_sum=1.5,
            true_positives=2,
            false_positives=1,
            false_negatives=1,
        )
        self.assertAlmostEqual(metrics.segmentation_quality, 0.75)
        self.assertAlmostEqual(metrics.recognition_quality, 2 / 3)
        self.assertAlmostEqual(metrics.panoptic_quality, 0.5)

    def test_perfect_empty_panoptic_case(self) -> None:
        metrics = panoptic_quality(
            matched_iou_sum=0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
        )
        self.assertEqual(metrics.panoptic_quality, 1.0)
        self.assertEqual(metrics.segmentation_quality, 1.0)
        self.assertEqual(metrics.recognition_quality, 1.0)

    def test_invalid_masks_and_panoptic_counts_fail(self) -> None:
        with self.assertRaises(ValueError):
            confusion_matrix(((0, 1),), ((0,),), num_classes=2)
        with self.assertRaises(ValueError):
            segmentation_report(((0, 2),), ((0, 1),), num_classes=2)
        with self.assertRaises(ValueError):
            panoptic_quality(
                matched_iou_sum=2.1,
                true_positives=2,
                false_positives=0,
                false_negatives=0,
            )


if __name__ == "__main__":
    unittest.main()
