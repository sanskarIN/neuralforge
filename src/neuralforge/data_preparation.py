"""Leakage-resistant data preparation helpers for NeuralForge Part 007."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


def _matrix(values: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    if len(values) == 0:
        raise ValueError("features must contain at least one row")
    try:
        rows = tuple(tuple(float(value) for value in row) for row in values)
    except (TypeError, ValueError) as exc:
        raise TypeError("features must contain numeric values") from exc
    if len(rows[0]) == 0:
        raise ValueError("feature rows must contain at least one value")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("features must form a rectangular matrix")
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError("features must contain only finite values")
    return rows


def _ratios(train: float, validation: float, test: float) -> tuple[float, float, float]:
    values = (float(train), float(validation), float(test))
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("split ratios must be finite and greater than zero")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to 1.0")
    return values


@dataclass(frozen=True, slots=True)
class SplitIndices:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    def __post_init__(self) -> None:
        combined = self.train + self.validation + self.test
        if len(combined) != len(set(combined)):
            raise ValueError("split indices must not overlap")

    @property
    def all_indices(self) -> tuple[int, ...]:
        return self.train + self.validation + self.test


@dataclass(frozen=True, slots=True)
class Standardizer:
    """Column-wise standardizer fitted on training data only."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(cls, training_features: Sequence[Sequence[float]]) -> "Standardizer":
        rows = _matrix(training_features)
        count = len(rows)
        width = len(rows[0])
        means = tuple(
            math.fsum(row[column] for row in rows) / count for column in range(width)
        )
        variances = tuple(
            math.fsum((row[column] - means[column]) ** 2 for row in rows) / count
            for column in range(width)
        )
        scales = tuple(math.sqrt(value) if value > 0.0 else 1.0 for value in variances)
        return cls(mean=means, scale=scales)

    def transform(self, features: Sequence[Sequence[float]]) -> list[list[float]]:
        rows = _matrix(features)
        if len(rows[0]) != len(self.mean):
            raise ValueError(
                f"expected {len(self.mean)} features, received {len(rows[0])}"
            )
        return [
            [
                (value - self.mean[column]) / self.scale[column]
                for column, value in enumerate(row)
            ]
            for row in rows
        ]

    def fit_transform_training(
        self, training_features: Sequence[Sequence[float]]
    ) -> list[list[float]]:
        """Transform data using this already-fitted training-only state."""

        return self.transform(training_features)


def random_split_indices(
    size: int,
    *,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> SplitIndices:
    """Create mutually exclusive train/validation/test index sets."""

    if isinstance(size, bool) or not isinstance(size, int) or size < 3:
        raise ValueError("size must be an integer of at least 3")
    train_ratio, validation_ratio, test_ratio = _ratios(
        train_ratio, validation_ratio, test_ratio
    )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    indices = list(range(size))
    random.Random(seed).shuffle(indices)

    train_count = max(1, int(size * train_ratio))
    validation_count = max(1, int(size * validation_ratio))
    if train_count + validation_count >= size:
        validation_count = 1
        train_count = size - 2

    train = tuple(indices[:train_count])
    validation_end = train_count + validation_count
    validation = tuple(indices[train_count:validation_end])
    test = tuple(indices[validation_end:])

    return SplitIndices(train=train, validation=validation, test=test)


def stratified_split_indices(
    labels: Sequence[object],
    *,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> SplitIndices:
    """Split classification indices while distributing each class across splits.

    Classes with fewer than three examples cannot appear in every split; their
    examples are still assigned exactly once and the global partitions remain
    disjoint. For reliable class-level evaluation, collect enough examples for
    each class before relying on stratification.
    """

    if len(labels) < 3:
        raise ValueError("labels must contain at least three examples")
    train_ratio, validation_ratio, test_ratio = _ratios(
        train_ratio, validation_ratio, test_ratio
    )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    by_label: dict[object, list[int]] = {}
    for index, label in enumerate(labels):
        try:
            by_label.setdefault(label, []).append(index)
        except TypeError as exc:
            raise TypeError("labels must be hashable") from exc

    generator = random.Random(seed)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []

    for indices in by_label.values():
        generator.shuffle(indices)
        count = len(indices)
        if count >= 3:
            train_count = max(1, int(count * train_ratio))
            validation_count = max(1, int(count * validation_ratio))
            if train_count + validation_count >= count:
                validation_count = 1
                train_count = count - 2
        else:
            train_count = max(1, int(round(count * train_ratio)))
            train_count = min(train_count, count)
            validation_count = min(
                count - train_count,
                int(round(count * validation_ratio)),
            )

        train.extend(indices[:train_count])
        validation_end = train_count + validation_count
        validation.extend(indices[train_count:validation_end])
        test.extend(indices[validation_end:])

    generator.shuffle(train)
    generator.shuffle(validation)
    generator.shuffle(test)

    split = SplitIndices(tuple(train), tuple(validation), tuple(test))
    if sorted(split.all_indices) != list(range(len(labels))):
        raise RuntimeError("internal split error: every example must appear exactly once")
    return split


def select_rows(
    features: Sequence[Sequence[float]], indices: Sequence[int]
) -> list[list[float]]:
    rows = _matrix(features)
    selected: list[list[float]] = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("indices must be integers")
        if index < 0 or index >= len(rows):
            raise IndexError(f"row index out of range: {index}")
        selected.append(list(rows[index]))
    return selected
