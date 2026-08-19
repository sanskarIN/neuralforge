"""Demonstrate segmentation thresholding and metrics for NeuralForge Part 025."""

from __future__ import annotations

from neuralforge.segmentation import (
    binary_confusion,
    majority_filter,
    mean_iou,
    multiclass_metrics,
    threshold_probabilities,
)


def print_mask(name: str, mask: tuple[tuple[int, ...], ...]) -> None:
    print(name)
    for row in mask:
        print("  ", " ".join(str(value) for value in row))


def main() -> None:
    probabilities = (
        (0.05, 0.15, 0.90, 0.92),
        (0.08, 0.55, 0.88, 0.95),
        (0.04, 0.12, 0.83, 0.89),
    )
    target = (
        (0, 0, 1, 1),
        (0, 0, 1, 1),
        (0, 0, 1, 1),
    )
    prediction = threshold_probabilities(probabilities, threshold=0.5)
    filtered = majority_filter(prediction, kernel_size=3)
    confusion = binary_confusion(prediction, target)

    print("NeuralForge Part 025 — image segmentation evaluation")
    print_mask("thresholded prediction", prediction)
    print_mask("majority-filtered prediction", filtered)
    print(f"IoU={confusion.iou:.3f} Dice={confusion.dice:.3f}")
    print(f"precision={confusion.precision:.3f} recall={confusion.recall:.3f}")

    multiclass_prediction = ((0, 1, 1), (0, 2, 2), (0, 2, 1))
    multiclass_target = ((0, 1, 1), (0, 2, 2), (0, 1, 1))
    metrics = multiclass_metrics(multiclass_prediction, multiclass_target, num_classes=3)
    print("\nmulticlass IoU")
    for item in metrics:
        print(f"  class {item.class_id}: IoU={item.iou:.3f} Dice={item.dice:.3f}")
    print(f"  mean IoU={mean_iou(metrics):.3f}")


if __name__ == "__main__":
    main()
