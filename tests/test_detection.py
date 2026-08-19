from __future__ import annotations

import unittest

from neuralforge.detection import (
    BoundingBox,
    Detection,
    LabeledBox,
    box_from_cxcywh,
    clip_box,
    intersection_over_union,
    match_detections,
    non_maximum_suppression,
)


class DetectionTests(unittest.TestCase):
    def test_xyxy_and_center_size_conversion(self) -> None:
        box = box_from_cxcywh(5, 5, 4, 2)
        self.assertEqual(box, BoundingBox(3, 4, 7, 6))
        self.assertEqual(box.to_cxcywh(), (5.0, 5.0, 4.0, 2.0))
        self.assertEqual(box.area, 8.0)

    def test_clipping_to_image_bounds(self) -> None:
        clipped = clip_box(BoundingBox(-2, -1, 6, 5), width=4, height=3)
        self.assertEqual(clipped, BoundingBox(0, 0, 4, 3))
        self.assertIsNone(clip_box(BoundingBox(5, 5, 7, 7), width=4, height=4))

    def test_iou_matches_hand_computed_overlap(self) -> None:
        first = BoundingBox(0, 0, 2, 2)
        second = BoundingBox(1, 1, 3, 3)
        self.assertAlmostEqual(intersection_over_union(first, second), 1.0 / 7.0)
        self.assertEqual(intersection_over_union(first, BoundingBox(3, 3, 4, 4)), 0.0)
        self.assertEqual(intersection_over_union(first, first), 1.0)

    def test_class_aware_nms_suppresses_duplicates_only_within_class(self) -> None:
        detections = (
            Detection(BoundingBox(0, 0, 10, 10), 0.95, class_id=0),
            Detection(BoundingBox(1, 1, 9, 9), 0.90, class_id=0),
            Detection(BoundingBox(1, 1, 9, 9), 0.85, class_id=1),
            Detection(BoundingBox(20, 20, 30, 30), 0.70, class_id=0),
        )
        kept = non_maximum_suppression(detections, iou_threshold=0.5, class_aware=True)
        self.assertEqual([item.score for item in kept], [0.95, 0.85, 0.70])
        class_agnostic = non_maximum_suppression(detections, iou_threshold=0.5, class_aware=False)
        self.assertEqual([item.score for item in class_agnostic], [0.95, 0.70])

    def test_nms_limit_keeps_highest_scores(self) -> None:
        detections = (
            Detection(BoundingBox(0, 0, 2, 2), 0.2),
            Detection(BoundingBox(3, 3, 5, 5), 0.9),
            Detection(BoundingBox(6, 6, 8, 8), 0.6),
        )
        kept = non_maximum_suppression(detections, max_detections=2)
        self.assertEqual([item.score for item in kept], [0.9, 0.6])

    def test_greedy_matching_counts_tp_fp_fn(self) -> None:
        truth = (
            LabeledBox(BoundingBox(0, 0, 10, 10), class_id=0),
            LabeledBox(BoundingBox(20, 20, 30, 30), class_id=1),
            LabeledBox(BoundingBox(40, 40, 50, 50), class_id=1),
        )
        predictions = (
            Detection(BoundingBox(0, 0, 10, 10), 0.99, class_id=0),
            Detection(BoundingBox(1, 1, 9, 9), 0.80, class_id=0),  # duplicate -> FP
            Detection(BoundingBox(20, 20, 30, 30), 0.75, class_id=1),
            Detection(BoundingBox(60, 60, 70, 70), 0.60, class_id=1),
        )
        metrics = match_detections(predictions, truth, iou_threshold=0.5)
        self.assertEqual(metrics.true_positives, 2)
        self.assertEqual(metrics.false_positives, 2)
        self.assertEqual(metrics.false_negatives, 1)
        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 2.0 / 3.0)
        self.assertEqual(len(metrics.matches), 2)

    def test_class_mismatch_is_not_a_match_by_default(self) -> None:
        prediction = Detection(BoundingBox(0, 0, 5, 5), 0.9, class_id=1)
        truth = LabeledBox(BoundingBox(0, 0, 5, 5), class_id=0)
        class_aware = match_detections([prediction], [truth])
        self.assertEqual(class_aware.true_positives, 0)
        class_agnostic = match_detections([prediction], [truth], class_aware=False)
        self.assertEqual(class_agnostic.true_positives, 1)

    def test_invalid_boxes_scores_and_thresholds_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BoundingBox(1, 1, 1, 2)
        with self.assertRaises(ValueError):
            Detection(BoundingBox(0, 0, 1, 1), 1.2)
        with self.assertRaises(ValueError):
            non_maximum_suppression([], iou_threshold=-0.1)
        with self.assertRaises(ValueError):
            box_from_cxcywh(0, 0, 0, 1)


if __name__ == "__main__":
    unittest.main()
