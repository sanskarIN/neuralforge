from __future__ import annotations

import unittest

from neuralforge.segmentation import (
    as_label_mask,
    binary_confusion,
    binary_iou,
    dice_score,
    majority_filter,
    mean_iou,
    multiclass_confusion_matrix,
    multiclass_metrics,
    threshold_probabilities,
)


class SegmentationTests(unittest.TestCase):
    def test_binary_confusion_iou_and_dice(self) -> None:
        prediction = [[1, 1], [0, 0]]
        target = [[1, 0], [1, 0]]
        confusion = binary_confusion(prediction, target)
        self.assertEqual(confusion.true_positives, 1)
        self.assertEqual(confusion.false_positives, 1)
        self.assertEqual(confusion.false_negatives, 1)
        self.assertEqual(confusion.true_negatives, 1)
        self.assertAlmostEqual(confusion.precision, 0.5)
        self.assertAlmostEqual(confusion.recall, 0.5)
        self.assertAlmostEqual(confusion.accuracy, 0.5)
        self.assertAlmostEqual(binary_iou(prediction, target), 1.0 / 3.0)
        self.assertAlmostEqual(dice_score(prediction, target), 0.5)

    def test_empty_positive_class_is_perfect_when_both_masks_are_empty(self) -> None:
        prediction = [[0, 0], [0, 0]]
        target = [[0, 0], [0, 0]]
        confusion = binary_confusion(prediction, target)
        self.assertEqual(confusion.iou, 1.0)
        self.assertEqual(confusion.dice, 1.0)
        self.assertEqual(confusion.precision, 1.0)
        self.assertEqual(confusion.recall, 1.0)

    def test_multiclass_confusion_and_metrics(self) -> None:
        target = [[0, 1], [2, 2]]
        prediction = [[0, 2], [2, 1]]
        matrix = multiclass_confusion_matrix(prediction, target, num_classes=3)
        self.assertEqual(matrix, ((1, 0, 0), (0, 0, 1), (0, 1, 1)))

        metrics = multiclass_metrics(prediction, target, num_classes=3)
        self.assertEqual(metrics[0].iou, 1.0)
        self.assertEqual(metrics[1].iou, 0.0)
        self.assertAlmostEqual(metrics[2].iou, 1.0 / 3.0)
        self.assertAlmostEqual(mean_iou(metrics), 4.0 / 9.0)

    def test_ignore_label_removes_pixels_from_confusion(self) -> None:
        target = [[0, 255], [1, 1]]
        prediction = [[0, 99], [1, 0]]
        matrix = multiclass_confusion_matrix(
            prediction,
            target,
            num_classes=2,
            ignore_label=255,
        )
        self.assertEqual(matrix, ((1, 0), (1, 1)))

    def test_probability_thresholding(self) -> None:
        probabilities = [[0.1, 0.5, 0.9], [0.49, 0.51, 1.0]]
        self.assertEqual(
            threshold_probabilities(probabilities, threshold=0.5),
            ((0, 1, 1), (0, 1, 1)),
        )

    def test_majority_filter_removes_isolated_center_label(self) -> None:
        noisy = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
        filtered = majority_filter(noisy, kernel_size=3)
        self.assertEqual(filtered[1][1], 1)
        self.assertEqual(filtered, ((1, 1, 1), (1, 1, 1), (1, 1, 1)))

    def test_majority_filter_tie_preserves_center(self) -> None:
        mask = [[0, 1], [1, 0]]
        filtered = majority_filter(mask, kernel_size=3)
        self.assertEqual(filtered, ((0, 1), (1, 0)))

    def test_mean_iou_excludes_absent_classes_by_default(self) -> None:
        metrics = multiclass_metrics([[0, 0]], [[0, 0]], num_classes=3)
        self.assertEqual(mean_iou(metrics), 1.0)
        self.assertEqual(mean_iou(metrics, include_empty=True), 1.0)

    def test_invalid_masks_labels_and_probabilities_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            as_label_mask([])
        with self.assertRaises(ValueError):
            as_label_mask([[0, 1], [1]])
        with self.assertRaises(TypeError):
            as_label_mask([[0.0]])  # type: ignore[list-item]
        with self.assertRaises(ValueError):
            multiclass_confusion_matrix([[0]], [[2]], num_classes=2)
        with self.assertRaises(ValueError):
            binary_confusion([[0]], [[0, 1]])
        with self.assertRaises(ValueError):
            threshold_probabilities([[1.2]])
        with self.assertRaises(ValueError):
            majority_filter([[0]], kernel_size=2)


if __name__ == "__main__":
    unittest.main()
