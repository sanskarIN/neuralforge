"""Demonstrate semantic and panoptic segmentation metrics for Part 025."""

from __future__ import annotations

from neuralforge.segmentation import panoptic_quality, segmentation_report


def main() -> None:
    target = (
        (0, 0, 1, 1),
        (0, 2, 2, 1),
        (0, 2, 2, 1),
    )
    prediction = (
        (0, 0, 1, 1),
        (0, 2, 1, 1),
        (0, 2, 2, 1),
    )

    report = segmentation_report(prediction, target, num_classes=3)
    print("NeuralForge Part 025 — Image Segmentation")
    print(f"pixel accuracy: {report.pixel_accuracy:.3f}")
    print(f"mean IoU: {report.mean_iou:.3f}")
    print(f"mean Dice: {report.mean_dice:.3f}")
    for metrics in report.per_class:
        print(
            f"  class {metrics.class_id}: TP={metrics.true_positive} "
            f"FP={metrics.false_positive} FN={metrics.false_negative} "
            f"IoU={metrics.iou if metrics.iou is not None else 'absent'} "
            f"Dice={metrics.dice if metrics.dice is not None else 'absent'}"
        )

    panoptic = panoptic_quality(
        matched_iou_sum=2.4,
        true_positives=3,
        false_positives=1,
        false_negatives=1,
    )
    print(
        f"panoptic quality={panoptic.panoptic_quality:.3f} "
        f"SQ={panoptic.segmentation_quality:.3f} RQ={panoptic.recognition_quality:.3f}"
    )


if __name__ == "__main__":
    main()
