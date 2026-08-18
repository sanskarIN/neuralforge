from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.detection import (  # noqa: E402
    BoundingBox,
    evaluate_detections,
    intersection_over_union,
    iou_matrix,
    non_maximum_suppression,
)


class DetectionTests(unittest.TestCase):
    def test_box_geometry_and_center_round_trip(self) -> None:
        box = BoundingBox(1, 2, 5, 8)
        self.assertEqual(box.width, 4.0)
        self.assertEqual(box.height, 6.0)
        self.assertEqual(box.area, 24.0)
        self.assertEqual(box.to_cxcywh(), (3.0, 5.0, 4.0, 6.0))
        self.assertEqual(BoundingBox.from_cxcywh(*box.to_cxcywh()), box)

    def test_iou_matches_manual_union(self) -> None:
        first = BoundingBox(0, 0, 2, 2)
        second = BoundingBox(1, 1, 3, 3)
        self.assertAlmostEqual(intersection_over_union(first, second), 1 / 7)
        self.assertEqual(intersection_over_union(first, BoundingBox(3, 3, 4, 4)), 0.0)

    def test_iou_matrix_shape(self) -> None:
        detections = (BoundingBox(0, 0, 1, 1), BoundingBox(2, 2, 3, 3))
        targets = (BoundingBox(0, 0, 1, 1),)
        self.assertEqual(iou_matrix(detections, targets), ((1.0,), (0.0,)))

    def test_nms_suppresses_overlapping_boxes(self) -> None:
        boxes = (
            BoundingBox(0, 0, 2, 2),
            BoundingBox(0.2, 0.2, 2.2, 2.2),
            BoundingBox(5, 5, 6, 6),
        )
        kept = non_maximum_suppression(boxes, (0.9, 0.8, 0.7), iou_threshold=0.5)
        self.assertEqual(kept, (0, 2))

    def test_class_aware_nms_keeps_different_classes(self) -> None:
        boxes = (BoundingBox(0, 0, 2, 2), BoundingBox(0.1, 0.1, 2.1, 2.1))
        kept = non_maximum_suppression(
            boxes,
            (0.9, 0.8),
            iou_threshold=0.5,
            class_ids=("cat", "dog"),
        )
        self.assertEqual(kept, (0, 1))

    def test_detection_evaluation_is_score_ordered_and_one_to_one(self) -> None:
        ground_truth = (BoundingBox(0, 0, 2, 2), BoundingBox(5, 5, 7, 7))
        boxes = (
            BoundingBox(0.1, 0.1, 1.9, 1.9),
            BoundingBox(0, 0, 2, 2),
            BoundingBox(5, 5, 7, 7),
            BoundingBox(10, 10, 11, 11),
        )
        result = evaluate_detections(boxes, (0.95, 0.90, 0.80, 0.70), ground_truth)
        self.assertEqual(result.true_positives, 2)
        self.assertEqual(result.false_positives, 2)
        self.assertEqual(result.false_negatives, 0)
        self.assertAlmostEqual(result.precision, 0.5)
        self.assertAlmostEqual(result.recall, 1.0)
        self.assertEqual(tuple(match.detection_index for match in result.matches), (0, 2))

    def test_class_aware_matching_rejects_wrong_class(self) -> None:
        target = (BoundingBox(0, 0, 2, 2),)
        result = evaluate_detections(
            (BoundingBox(0, 0, 2, 2),),
            (0.99,),
            target,
            predicted_classes=("dog",),
            ground_truth_classes=("cat",),
        )
        self.assertEqual(result.true_positives, 0)
        self.assertEqual(result.false_positives, 1)
        self.assertEqual(result.false_negatives, 1)

    def test_invalid_boxes_and_mismatched_inputs_fail(self) -> None:
        with self.assertRaises(ValueError):
            BoundingBox(1, 1, 1, 2)
        with self.assertRaises(ValueError):
            non_maximum_suppression((BoundingBox(0, 0, 1, 1),), (), iou_threshold=0.5)
        with self.assertRaises(ValueError):
            evaluate_detections(
                (BoundingBox(0, 0, 1, 1),),
                (0.5,),
                (),
                predicted_classes=(0,),
            )


if __name__ == "__main__":
    unittest.main()
