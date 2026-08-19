"""Demonstrate box IoU, NMS, and matching for NeuralForge Part 024."""

from __future__ import annotations

from neuralforge.detection import (
    BoundingBox,
    Detection,
    LabeledBox,
    intersection_over_union,
    match_detections,
    non_maximum_suppression,
)


def main() -> None:
    truth = (
        LabeledBox(BoundingBox(10, 10, 50, 50), class_id=0),
        LabeledBox(BoundingBox(70, 20, 110, 60), class_id=1),
    )
    raw = (
        Detection(BoundingBox(9, 9, 51, 51), 0.96, class_id=0),
        Detection(BoundingBox(12, 12, 48, 48), 0.82, class_id=0),
        Detection(BoundingBox(69, 19, 111, 61), 0.91, class_id=1),
        Detection(BoundingBox(120, 80, 145, 110), 0.55, class_id=1),
    )

    print("NeuralForge Part 024 — object detection post-processing")
    print(f"duplicate IoU: {intersection_over_union(raw[0].box, raw[1].box):.3f}")

    kept = non_maximum_suppression(raw, iou_threshold=0.5)
    print(f"detections before NMS: {len(raw)}")
    print(f"detections after NMS:  {len(kept)}")
    for detection in kept:
        print(f"  class={detection.class_id} score={detection.score:.2f} box={detection.box}")

    metrics = match_detections(kept, truth, iou_threshold=0.5)
    print("\nevaluation")
    print(f"  TP={metrics.true_positives} FP={metrics.false_positives} FN={metrics.false_negatives}")
    print(f"  precision={metrics.precision:.3f} recall={metrics.recall:.3f}")


if __name__ == "__main__":
    main()
