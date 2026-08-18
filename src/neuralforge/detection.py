"""Object-detection geometry and evaluation utilities for NeuralForge Part 024."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def _finite(value: float | int, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _threshold(value: float, *, name: str) -> float:
    result = _finite(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        coordinates = tuple(_finite(value, name="box coordinate") for value in (self.x1, self.y1, self.x2, self.y2))
        object.__setattr__(self, "x1", coordinates[0])
        object.__setattr__(self, "y1", coordinates[1])
        object.__setattr__(self, "x2", coordinates[2])
        object.__setattr__(self, "y2", coordinates[3])
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding boxes must have positive width and height")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def to_cxcywh(self) -> tuple[float, float, float, float]:
        return (self.center_x, self.center_y, self.width, self.height)

    @classmethod
    def from_cxcywh(
        cls,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
    ) -> "BoundingBox":
        cx = _finite(center_x, name="center_x")
        cy = _finite(center_y, name="center_y")
        box_width = _finite(width, name="width")
        box_height = _finite(height, name="height")
        if box_width <= 0.0 or box_height <= 0.0:
            raise ValueError("width and height must be greater than zero")
        return cls(
            cx - box_width / 2.0,
            cy - box_height / 2.0,
            cx + box_width / 2.0,
            cy + box_height / 2.0,
        )


@dataclass(frozen=True, slots=True)
class DetectionMatch:
    detection_index: int
    ground_truth_index: int
    iou: float


@dataclass(frozen=True, slots=True)
class DetectionEvaluation:
    matches: tuple[DetectionMatch, ...]
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    left = max(first.x1, second.x1)
    top = max(first.y1, second.y1)
    right = min(first.x2, second.x2)
    bottom = min(first.y2, second.y2)
    intersection_width = max(0.0, right - left)
    intersection_height = max(0.0, bottom - top)
    intersection = intersection_width * intersection_height
    union = first.area + second.area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def iou_matrix(
    detections: Sequence[BoundingBox], ground_truth: Sequence[BoundingBox]
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(intersection_over_union(detection, target) for target in ground_truth)
        for detection in detections
    )


def non_maximum_suppression(
    boxes: Sequence[BoundingBox],
    scores: Sequence[float | int],
    *,
    iou_threshold: float = 0.5,
    class_ids: Sequence[int | str] | None = None,
    max_detections: int | None = None,
) -> tuple[int, ...]:
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have the same length")
    threshold = _threshold(iou_threshold, name="iou_threshold")
    normalized_scores = tuple(_finite(score, name="score") for score in scores)
    if class_ids is not None and len(class_ids) != len(boxes):
        raise ValueError("class_ids must have the same length as boxes")
    if max_detections is not None:
        if isinstance(max_detections, bool) or not isinstance(max_detections, int) or max_detections <= 0:
            raise ValueError("max_detections must be a positive integer")

    order = sorted(range(len(boxes)), key=lambda index: (-normalized_scores[index], index))
    kept: list[int] = []
    while order:
        current = order.pop(0)
        kept.append(current)
        if max_detections is not None and len(kept) >= max_detections:
            break
        survivors: list[int] = []
        for candidate in order:
            same_class = class_ids is None or class_ids[candidate] == class_ids[current]
            if same_class and intersection_over_union(boxes[current], boxes[candidate]) > threshold:
                continue
            survivors.append(candidate)
        order = survivors
    return tuple(kept)


def evaluate_detections(
    boxes: Sequence[BoundingBox],
    scores: Sequence[float | int],
    ground_truth: Sequence[BoundingBox],
    *,
    iou_threshold: float = 0.5,
    predicted_classes: Sequence[int | str] | None = None,
    ground_truth_classes: Sequence[int | str] | None = None,
) -> DetectionEvaluation:
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have the same length")
    if (predicted_classes is None) != (ground_truth_classes is None):
        raise ValueError("provide both predicted_classes and ground_truth_classes, or neither")
    if predicted_classes is not None and len(predicted_classes) != len(boxes):
        raise ValueError("predicted_classes must match detection count")
    if ground_truth_classes is not None and len(ground_truth_classes) != len(ground_truth):
        raise ValueError("ground_truth_classes must match ground-truth count")

    threshold = _threshold(iou_threshold, name="iou_threshold")
    normalized_scores = tuple(_finite(score, name="score") for score in scores)
    order = sorted(range(len(boxes)), key=lambda index: (-normalized_scores[index], index))
    unmatched = set(range(len(ground_truth)))
    matches: list[DetectionMatch] = []

    for detection_index in order:
        best_ground_truth: int | None = None
        best_iou = threshold
        for target_index in sorted(unmatched):
            if predicted_classes is not None and predicted_classes[detection_index] != ground_truth_classes[target_index]:
                continue
            overlap = intersection_over_union(boxes[detection_index], ground_truth[target_index])
            if overlap >= best_iou:
                if overlap > best_iou or best_ground_truth is None or target_index < best_ground_truth:
                    best_ground_truth = target_index
                    best_iou = overlap
        if best_ground_truth is not None:
            unmatched.remove(best_ground_truth)
            matches.append(DetectionMatch(detection_index, best_ground_truth, best_iou))

    true_positives = len(matches)
    false_positives = len(boxes) - true_positives
    false_negatives = len(ground_truth) - true_positives
    precision = true_positives / len(boxes) if boxes else (1.0 if not ground_truth else 0.0)
    recall = true_positives / len(ground_truth) if ground_truth else 1.0
    return DetectionEvaluation(
        matches=tuple(matches),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
    )
