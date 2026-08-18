"""Demonstrate IoU, NMS, and detection evaluation for Part 024."""

from __future__ import annotations

from neuralforge.detection import (
    BoundingBox,
    evaluate_detections,
    intersection_over_union,
    non_maximum_suppression,
)


def main() -> None:
    boxes = (
        BoundingBox(0, 0, 2, 2),
        BoundingBox(0.2, 0.2, 2.2, 2.2),
        BoundingBox(5, 5, 7, 7),
    )
    scores = (0.95, 0.80, 0.90)
    targets = (BoundingBox(0, 0, 2, 2), BoundingBox(5, 5, 7, 7))

    print("NeuralForge Part 024 — Object Detection")
    print(f"IoU(box0, box1): {intersection_over_union(boxes[0], boxes[1]):.3f}")
    kept = non_maximum_suppression(boxes, scores, iou_threshold=0.5)
    print(f"NMS kept detection indices: {kept}")

    filtered_boxes = tuple(boxes[index] for index in kept)
    filtered_scores = tuple(scores[index] for index in kept)
    result = evaluate_detections(filtered_boxes, filtered_scores, targets)
    print(
        f"TP={result.true_positives} FP={result.false_positives} "
        f"FN={result.false_negatives} precision={result.precision:.3f} recall={result.recall:.3f}"
    )
    for match in result.matches:
        print(
            f"  detection {match.detection_index} -> target {match.ground_truth_index} "
            f"IoU={match.iou:.3f}"
        )


if __name__ == "__main__":
    main()
