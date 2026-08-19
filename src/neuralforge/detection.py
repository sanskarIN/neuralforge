"""Object-detection geometry and post-processing for NeuralForge Part 024."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


def _finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unit_interval(value: float, *, name: str, allow_one: bool = True) -> float:
    result = _finite(value, name=name)
    upper_ok = result <= 1.0 if allow_one else result < 1.0
    if result < 0.0 or not upper_ok:
        bracket = "[0, 1]" if allow_one else "[0, 1)"
        raise ValueError(f"{name} must be in {bracket}")
    return result


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned box in ``(x1, y1, x2, y2)`` edge coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = tuple(_finite(value, name="box coordinate") for value in (self.x1, self.y1, self.x2, self.y2))
        object.__setattr__(self, "x1", values[0])
        object.__setattr__(self, "y1", values[1])
        object.__setattr__(self, "x2", values[2])
        object.__setattr__(self, "y2", values[3])
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding box must have positive width and height")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_cxcywh(self) -> tuple[float, float, float, float]:
        return (
            (self.x1 + self.x2) / 2.0,
            (self.y1 + self.y2) / 2.0,
            self.width,
            self.height,
        )


def box_from_cxcywh(cx: float, cy: float, width: float, height: float) -> BoundingBox:
    center_x = _finite(cx, name="cx")
    center_y = _finite(cy, name="cy")
    w = _finite(width, name="width")
    h = _finite(height, name="height")
    if w <= 0.0 or h <= 0.0:
        raise ValueError("width and height must be greater than zero")
    return BoundingBox(
        center_x - w / 2.0,
        center_y - h / 2.0,
        center_x + w / 2.0,
        center_y + h / 2.0,
    )


def clip_box(box: BoundingBox, *, width: float, height: float) -> BoundingBox | None:
    """Clip a box to image bounds, returning ``None`` if no positive area remains."""

    image_width = _finite(width, name="width")
    image_height = _finite(height, name="height")
    if image_width <= 0.0 or image_height <= 0.0:
        raise ValueError("image width and height must be greater than zero")
    x1 = min(max(box.x1, 0.0), image_width)
    y1 = min(max(box.y1, 0.0), image_height)
    x2 = min(max(box.x2, 0.0), image_width)
    y2 = min(max(box.y2, 0.0), image_height)
    if x2 <= x1 or y2 <= y1:
        return None
    return BoundingBox(x1, y1, x2, y2)


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    """Return axis-aligned intersection-over-union in ``[0, 1]``."""

    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0.0:
        return 0.0
    union = first.area + second.area - intersection
    return intersection / union


@dataclass(frozen=True, slots=True)
class Detection:
    box: BoundingBox
    score: float
    class_id: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", _unit_interval(self.score, name="score"))
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int) or self.class_id < 0:
            raise ValueError("class_id must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class LabeledBox:
    box: BoundingBox
    class_id: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int) or self.class_id < 0:
            raise ValueError("class_id must be a non-negative integer")


def non_maximum_suppression(
    detections: Iterable[Detection],
    *,
    iou_threshold: float = 0.5,
    class_aware: bool = True,
    max_detections: int | None = None,
) -> tuple[Detection, ...]:
    """Greedily suppress lower-score overlapping detections."""

    threshold = _unit_interval(iou_threshold, name="iou_threshold")
    if max_detections is not None:
        if isinstance(max_detections, bool) or not isinstance(max_detections, int) or max_detections <= 0:
            raise ValueError("max_detections must be a positive integer when provided")
    ordered = sorted(tuple(detections), key=lambda item: item.score, reverse=True)
    kept: list[Detection] = []
    while ordered and (max_detections is None or len(kept) < max_detections):
        best = ordered.pop(0)
        kept.append(best)
        survivors: list[Detection] = []
        for candidate in ordered:
            same_suppression_group = not class_aware or candidate.class_id == best.class_id
            if same_suppression_group and intersection_over_union(best.box, candidate.box) > threshold:
                continue
            survivors.append(candidate)
        ordered = survivors
    return tuple(kept)


@dataclass(frozen=True, slots=True)
class DetectionMatch:
    prediction_index: int
    ground_truth_index: int
    iou: float


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    matches: tuple[DetectionMatch, ...]


def match_detections(
    predictions: Iterable[Detection],
    ground_truth: Iterable[LabeledBox],
    *,
    iou_threshold: float = 0.5,
    class_aware: bool = True,
) -> DetectionMetrics:
    """Greedily match scored predictions to each ground-truth box at most once."""

    threshold = _unit_interval(iou_threshold, name="iou_threshold")
    prediction_values = tuple(predictions)
    truth_values = tuple(ground_truth)
    ordered_indices = sorted(
        range(len(prediction_values)),
        key=lambda index: prediction_values[index].score,
        reverse=True,
    )
    unmatched_truth = set(range(len(truth_values)))
    matches: list[DetectionMatch] = []

    for prediction_index in ordered_indices:
        prediction = prediction_values[prediction_index]
        best_truth: int | None = None
        best_iou = -1.0
        for truth_index in unmatched_truth:
            truth = truth_values[truth_index]
            if class_aware and prediction.class_id != truth.class_id:
                continue
            overlap = intersection_over_union(prediction.box, truth.box)
            if overlap >= threshold and overlap > best_iou:
                best_truth = truth_index
                best_iou = overlap
        if best_truth is not None:
            unmatched_truth.remove(best_truth)
            matches.append(DetectionMatch(prediction_index, best_truth, best_iou))

    true_positives = len(matches)
    false_positives = len(prediction_values) - true_positives
    false_negatives = len(truth_values) - true_positives
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    return DetectionMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        matches=tuple(matches),
    )
