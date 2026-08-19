"""Segmentation metrics and mask post-processing for NeuralForge Part 025."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

LabelMask = tuple[tuple[int, ...], ...]
ProbabilityMask = tuple[tuple[float, ...], ...]


def as_label_mask(values: Sequence[Sequence[int | bool]], *, name: str = "mask") -> LabelMask:
    if not values:
        raise ValueError(f"{name} must contain at least one row")
    rows: list[tuple[int, ...]] = []
    width: int | None = None
    for row_index, row in enumerate(values):
        if not row:
            raise ValueError(f"{name} row {row_index} must not be empty")
        normalized: list[int] = []
        for value in row:
            if isinstance(value, bool):
                label = int(value)
            elif isinstance(value, int):
                label = value
            else:
                raise TypeError(f"{name} labels must be integers")
            if label < 0:
                raise ValueError(f"{name} labels must be non-negative")
            normalized.append(label)
        if width is None:
            width = len(normalized)
        elif len(normalized) != width:
            raise ValueError(f"{name} must be rectangular")
        rows.append(tuple(normalized))
    return tuple(rows)


def _same_shape(first: LabelMask, second: LabelMask) -> None:
    if len(first) != len(second) or len(first[0]) != len(second[0]):
        raise ValueError("prediction and target masks must have the same shape")


@dataclass(frozen=True, slots=True)
class BinaryConfusion:
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    @property
    def total(self) -> int:
        return self.true_positives + self.false_positives + self.false_negatives + self.true_negatives

    @property
    def union(self) -> int:
        return self.true_positives + self.false_positives + self.false_negatives

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 1.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 1.0

    @property
    def accuracy(self) -> float:
        return (self.true_positives + self.true_negatives) / self.total if self.total else 1.0

    @property
    def iou(self) -> float:
        return self.true_positives / self.union if self.union else 1.0

    @property
    def dice(self) -> float:
        denominator = 2 * self.true_positives + self.false_positives + self.false_negatives
        return 2 * self.true_positives / denominator if denominator else 1.0


def binary_confusion(
    prediction: Sequence[Sequence[int | bool]],
    target: Sequence[Sequence[int | bool]],
    *,
    positive_label: int = 1,
    ignore_label: int | None = None,
) -> BinaryConfusion:
    pred = as_label_mask(prediction, name="prediction")
    truth = as_label_mask(target, name="target")
    _same_shape(pred, truth)
    if isinstance(positive_label, bool) or not isinstance(positive_label, int) or positive_label < 0:
        raise ValueError("positive_label must be a non-negative integer")
    if ignore_label is not None and (
        isinstance(ignore_label, bool) or not isinstance(ignore_label, int) or ignore_label < 0
    ):
        raise ValueError("ignore_label must be a non-negative integer when provided")

    tp = fp = fn = tn = 0
    for pred_row, truth_row in zip(pred, truth):
        for predicted, actual in zip(pred_row, truth_row):
            if ignore_label is not None and actual == ignore_label:
                continue
            predicted_positive = predicted == positive_label
            actual_positive = actual == positive_label
            if predicted_positive and actual_positive:
                tp += 1
            elif predicted_positive:
                fp += 1
            elif actual_positive:
                fn += 1
            else:
                tn += 1
    return BinaryConfusion(tp, fp, fn, tn)


def binary_iou(
    prediction: Sequence[Sequence[int | bool]],
    target: Sequence[Sequence[int | bool]],
    *,
    positive_label: int = 1,
) -> float:
    return binary_confusion(prediction, target, positive_label=positive_label).iou


def dice_score(
    prediction: Sequence[Sequence[int | bool]],
    target: Sequence[Sequence[int | bool]],
    *,
    positive_label: int = 1,
) -> float:
    return binary_confusion(prediction, target, positive_label=positive_label).dice


def multiclass_confusion_matrix(
    prediction: Sequence[Sequence[int | bool]],
    target: Sequence[Sequence[int | bool]],
    *,
    num_classes: int,
    ignore_label: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    pred = as_label_mask(prediction, name="prediction")
    truth = as_label_mask(target, name="target")
    _same_shape(pred, truth)
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 1:
        raise ValueError("num_classes must be an integer greater than one")
    if ignore_label is not None and (
        isinstance(ignore_label, bool) or not isinstance(ignore_label, int) or ignore_label < 0
    ):
        raise ValueError("ignore_label must be a non-negative integer when provided")

    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for pred_row, truth_row in zip(pred, truth):
        for predicted, actual in zip(pred_row, truth_row):
            if ignore_label is not None and actual == ignore_label:
                continue
            if not 0 <= predicted < num_classes:
                raise ValueError("prediction contains a label outside num_classes")
            if not 0 <= actual < num_classes:
                raise ValueError("target contains a label outside num_classes")
            matrix[actual][predicted] += 1
    return tuple(tuple(row) for row in matrix)


@dataclass(frozen=True, slots=True)
class ClassSegmentationMetrics:
    class_id: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    iou: float
    dice: float
    precision: float
    recall: float

    @property
    def union(self) -> int:
        return self.true_positives + self.false_positives + self.false_negatives


def multiclass_metrics(
    prediction: Sequence[Sequence[int | bool]],
    target: Sequence[Sequence[int | bool]],
    *,
    num_classes: int,
    ignore_label: int | None = None,
) -> tuple[ClassSegmentationMetrics, ...]:
    matrix = multiclass_confusion_matrix(
        prediction,
        target,
        num_classes=num_classes,
        ignore_label=ignore_label,
    )
    total = sum(sum(row) for row in matrix)
    metrics: list[ClassSegmentationMetrics] = []
    for class_id in range(num_classes):
        tp = matrix[class_id][class_id]
        fn = sum(matrix[class_id]) - tp
        fp = sum(matrix[row][class_id] for row in range(num_classes)) - tp
        tn = total - tp - fp - fn
        confusion = BinaryConfusion(tp, fp, fn, tn)
        metrics.append(
            ClassSegmentationMetrics(
                class_id=class_id,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                true_negatives=tn,
                iou=confusion.iou,
                dice=confusion.dice,
                precision=confusion.precision,
                recall=confusion.recall,
            )
        )
    return tuple(metrics)


def mean_iou(
    metrics: Iterable[ClassSegmentationMetrics],
    *,
    include_empty: bool = False,
) -> float:
    values = tuple(metrics)
    if not values:
        raise ValueError("metrics must contain at least one class")
    selected = values if include_empty else tuple(item for item in values if item.union > 0)
    if not selected:
        return 1.0
    return math.fsum(item.iou for item in selected) / len(selected)


def threshold_probabilities(
    probabilities: Sequence[Sequence[float]],
    *,
    threshold: float = 0.5,
) -> LabelMask:
    cutoff = float(threshold)
    if not math.isfinite(cutoff) or not 0.0 <= cutoff <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if not probabilities:
        raise ValueError("probabilities must contain at least one row")
    rows: list[tuple[int, ...]] = []
    width: int | None = None
    for row in probabilities:
        if not row:
            raise ValueError("probability rows must not be empty")
        normalized = tuple(float(value) for value in row)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in normalized):
            raise ValueError("probabilities must be finite and in [0, 1]")
        if width is None:
            width = len(normalized)
        elif len(normalized) != width:
            raise ValueError("probabilities must be rectangular")
        rows.append(tuple(int(value >= cutoff) for value in normalized))
    return tuple(rows)


def majority_filter(
    mask: Sequence[Sequence[int | bool]],
    *,
    kernel_size: int = 3,
) -> LabelMask:
    """Replace each label with the local majority; ties preserve the center label."""

    labels = as_label_mask(mask)
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int) or kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    radius = kernel_size // 2
    height = len(labels)
    width = len(labels[0])
    output: list[tuple[int, ...]] = []
    for y in range(height):
        row: list[int] = []
        for x in range(width):
            neighbors = [
                labels[ny][nx]
                for ny in range(max(0, y - radius), min(height, y + radius + 1))
                for nx in range(max(0, x - radius), min(width, x + radius + 1))
            ]
            counts = Counter(neighbors)
            best_count = max(counts.values())
            winners = {label for label, count in counts.items() if count == best_count}
            center = labels[y][x]
            row.append(center if center in winners else min(winners))
        output.append(tuple(row))
    return tuple(output)
