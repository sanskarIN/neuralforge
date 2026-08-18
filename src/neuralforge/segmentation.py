"""Semantic/panoptic segmentation metrics for NeuralForge Part 025."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

LabelMask = tuple[tuple[int, ...], ...]


def _mask(values: Sequence[Sequence[int]], *, name: str) -> LabelMask:
    rows = tuple(tuple(int(value) for value in row) for row in values)
    if not rows or not rows[0]:
        raise ValueError(f"{name} must be a non-empty 2D mask")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be rectangular")
    return rows


def _same_shape(first: LabelMask, second: LabelMask) -> None:
    if len(first) != len(second) or len(first[0]) != len(second[0]):
        raise ValueError("prediction and target masks must have the same shape")


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ClassSegmentationMetrics:
    class_id: int
    true_positive: int
    false_positive: int
    false_negative: int
    iou: float | None
    dice: float | None


@dataclass(frozen=True, slots=True)
class SegmentationReport:
    confusion_matrix: tuple[tuple[int, ...], ...]
    per_class: tuple[ClassSegmentationMetrics, ...]
    pixel_accuracy: float
    mean_iou: float
    mean_dice: float


@dataclass(frozen=True, slots=True)
class PanopticMetrics:
    panoptic_quality: float
    segmentation_quality: float
    recognition_quality: float


def confusion_matrix(
    prediction: Sequence[Sequence[int]],
    target: Sequence[Sequence[int]],
    *,
    num_classes: int | None = None,
    ignore_index: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    predicted = _mask(prediction, name="prediction")
    expected = _mask(target, name="target")
    _same_shape(predicted, expected)

    observed: list[int] = []
    for predicted_row, target_row in zip(predicted, expected):
        for predicted_label, target_label in zip(predicted_row, target_row):
            if target_label == ignore_index:
                continue
            observed.extend((predicted_label, target_label))

    if num_classes is None:
        if not observed:
            raise ValueError("cannot infer classes when every pixel is ignored")
        if min(observed) < 0:
            raise ValueError("class labels must be non-negative unless they are ignored")
        classes = max(observed) + 1
    else:
        classes = _positive_int(num_classes, name="num_classes")

    matrix = [[0 for _ in range(classes)] for _ in range(classes)]
    used_pixels = 0
    for predicted_row, target_row in zip(predicted, expected):
        for predicted_label, target_label in zip(predicted_row, target_row):
            if target_label == ignore_index:
                continue
            if not 0 <= target_label < classes:
                raise ValueError("target label is outside num_classes")
            if not 0 <= predicted_label < classes:
                raise ValueError("prediction label is outside num_classes")
            matrix[target_label][predicted_label] += 1
            used_pixels += 1
    if used_pixels == 0:
        raise ValueError("no non-ignored pixels remain for evaluation")
    return tuple(tuple(row) for row in matrix)


def segmentation_report(
    prediction: Sequence[Sequence[int]],
    target: Sequence[Sequence[int]],
    *,
    num_classes: int | None = None,
    ignore_index: int | None = None,
) -> SegmentationReport:
    matrix = confusion_matrix(
        prediction,
        target,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    classes = len(matrix)
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[index][index] for index in range(classes))

    per_class: list[ClassSegmentationMetrics] = []
    ious: list[float] = []
    dices: list[float] = []
    for class_id in range(classes):
        true_positive = matrix[class_id][class_id]
        false_positive = sum(matrix[row][class_id] for row in range(classes) if row != class_id)
        false_negative = sum(matrix[class_id][column] for column in range(classes) if column != class_id)
        union = true_positive + false_positive + false_negative
        dice_denominator = 2 * true_positive + false_positive + false_negative
        iou = true_positive / union if union else None
        dice = 2 * true_positive / dice_denominator if dice_denominator else None
        if iou is not None:
            ious.append(iou)
        if dice is not None:
            dices.append(dice)
        per_class.append(
            ClassSegmentationMetrics(
                class_id=class_id,
                true_positive=true_positive,
                false_positive=false_positive,
                false_negative=false_negative,
                iou=iou,
                dice=dice,
            )
        )

    return SegmentationReport(
        confusion_matrix=matrix,
        per_class=tuple(per_class),
        pixel_accuracy=correct / total,
        mean_iou=math.fsum(ious) / len(ious) if ious else 0.0,
        mean_dice=math.fsum(dices) / len(dices) if dices else 0.0,
    )


def binary_mask_iou(
    prediction: Sequence[Sequence[int | bool]],
    target: Sequence[Sequence[int | bool]],
) -> float:
    predicted = _mask(prediction, name="prediction")
    expected = _mask(target, name="target")
    _same_shape(predicted, expected)
    intersection = 0
    union = 0
    for predicted_row, target_row in zip(predicted, expected):
        for predicted_value, target_value in zip(predicted_row, target_row):
            predicted_on = bool(predicted_value)
            target_on = bool(target_value)
            intersection += int(predicted_on and target_on)
            union += int(predicted_on or target_on)
    return 1.0 if union == 0 else intersection / union


def binary_dice_score(
    prediction: Sequence[Sequence[int | bool]],
    target: Sequence[Sequence[int | bool]],
) -> float:
    predicted = _mask(prediction, name="prediction")
    expected = _mask(target, name="target")
    _same_shape(predicted, expected)
    intersection = 0
    predicted_count = 0
    target_count = 0
    for predicted_row, target_row in zip(predicted, expected):
        for predicted_value, target_value in zip(predicted_row, target_row):
            predicted_on = bool(predicted_value)
            target_on = bool(target_value)
            predicted_count += int(predicted_on)
            target_count += int(target_on)
            intersection += int(predicted_on and target_on)
    denominator = predicted_count + target_count
    return 1.0 if denominator == 0 else 2 * intersection / denominator


def panoptic_quality(
    *,
    matched_iou_sum: float,
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> PanopticMetrics:
    iou_sum = float(matched_iou_sum)
    if not math.isfinite(iou_sum) or iou_sum < 0.0:
        raise ValueError("matched_iou_sum must be finite and non-negative")
    counts = (true_positives, false_positives, false_negatives)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
        raise ValueError("panoptic counts must be non-negative integers")
    if iou_sum > true_positives + 1e-12:
        raise ValueError("matched_iou_sum cannot exceed true_positives")

    denominator = true_positives + 0.5 * false_positives + 0.5 * false_negatives
    recognition = true_positives / denominator if denominator else 1.0
    segmentation = iou_sum / true_positives if true_positives else (1.0 if denominator == 0 else 0.0)
    return PanopticMetrics(
        panoptic_quality=segmentation * recognition,
        segmentation_quality=segmentation,
        recognition_quality=recognition,
    )
