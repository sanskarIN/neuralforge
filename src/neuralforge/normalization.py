"""Normalization and training-stability helpers for NeuralForge Part 015."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .autograd import Value

Scalar = Value | int | float


def _value(item: Scalar) -> Value:
    return item if isinstance(item, Value) else Value(item)


def _matrix(batch: Sequence[Sequence[Scalar]]) -> tuple[tuple[Value, ...], ...]:
    if not batch:
        raise ValueError("batch must contain at least one row")
    rows = tuple(tuple(_value(item) for item in row) for row in batch)
    if not rows[0]:
        raise ValueError("batch rows must contain at least one feature")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("batch must be rectangular")
    return rows


def _epsilon(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("epsilon must be finite and greater than zero")
    return result


def _affine(
    values: Sequence[Value] | None,
    width: int,
    *,
    default: float,
    name: str,
) -> tuple[Value, ...]:
    if values is None:
        return tuple(Value(default) for _ in range(width))
    result = tuple(values)
    if len(result) != width:
        raise ValueError(f"{name} must contain exactly {width} Values")
    if any(not isinstance(item, Value) for item in result):
        raise TypeError(f"{name} must contain only Value objects")
    return result


def normalization_parameters(size: int) -> tuple[tuple[Value, ...], tuple[Value, ...]]:
    """Create trainable affine ``gamma=1`` and ``beta=0`` parameters."""

    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("size must be a positive integer")
    gamma = tuple(Value(1.0, label=f"gamma{index}") for index in range(size))
    beta = tuple(Value(0.0, label=f"beta{index}") for index in range(size))
    return gamma, beta


@dataclass(frozen=True, slots=True)
class BatchNormalizationResult:
    outputs: tuple[tuple[Value, ...], ...]
    means: tuple[Value, ...]
    variances: tuple[Value, ...]
    gamma: tuple[Value, ...]
    beta: tuple[Value, ...]


def batch_normalize(
    batch: Sequence[Sequence[Scalar]],
    *,
    epsilon: float = 1e-5,
    gamma: Sequence[Value] | None = None,
    beta: Sequence[Value] | None = None,
) -> BatchNormalizationResult:
    """Normalize each feature across the batch with differentiable statistics."""

    rows = _matrix(batch)
    eps = _epsilon(epsilon)
    count = len(rows)
    width = len(rows[0])
    scale = _affine(gamma, width, default=1.0, name="gamma")
    shift = _affine(beta, width, default=0.0, name="beta")

    means: list[Value] = []
    variances: list[Value] = []
    for column in range(width):
        mean = sum((row[column] for row in rows), Value(0.0)) / count
        variance = (
            sum(((row[column] - mean) ** 2 for row in rows), Value(0.0))
            / count
        )
        means.append(mean)
        variances.append(variance)

    outputs: list[tuple[Value, ...]] = []
    for row in rows:
        normalized_row = []
        for column, value in enumerate(row):
            denominator = (variances[column] + eps) ** 0.5
            normalized = (value - means[column]) / denominator
            normalized_row.append(normalized * scale[column] + shift[column])
        outputs.append(tuple(normalized_row))

    return BatchNormalizationResult(
        outputs=tuple(outputs),
        means=tuple(means),
        variances=tuple(variances),
        gamma=scale,
        beta=shift,
    )


@dataclass(frozen=True, slots=True)
class LayerNormalizationResult:
    outputs: tuple[Value, ...]
    mean: Value
    variance: Value
    gamma: tuple[Value, ...]
    beta: tuple[Value, ...]


def layer_normalize(
    values: Sequence[Scalar],
    *,
    epsilon: float = 1e-5,
    gamma: Sequence[Value] | None = None,
    beta: Sequence[Value] | None = None,
) -> LayerNormalizationResult:
    """Normalize features within one example."""

    row = tuple(_value(item) for item in values)
    if not row:
        raise ValueError("values must contain at least one feature")
    eps = _epsilon(epsilon)
    width = len(row)
    scale = _affine(gamma, width, default=1.0, name="gamma")
    shift = _affine(beta, width, default=0.0, name="beta")

    mean = sum(row, Value(0.0)) / width
    variance = sum(((value - mean) ** 2 for value in row), Value(0.0)) / width
    denominator = (variance + eps) ** 0.5
    outputs = tuple(
        ((value - mean) / denominator) * scale[index] + shift[index]
        for index, value in enumerate(row)
    )
    return LayerNormalizationResult(outputs, mean, variance, scale, shift)


def stable_softmax(values: Sequence[Scalar]) -> tuple[Value, ...]:
    """Compute softmax after subtracting the largest scalar value."""

    logits = tuple(_value(item) for item in values)
    if not logits:
        raise ValueError("softmax requires at least one logit")
    shift = max(logit.data for logit in logits)
    exponentials = tuple((logit - shift).exp() for logit in logits)
    denominator = sum(exponentials, Value(0.0))
    return tuple(exponential / denominator for exponential in exponentials)


def _numeric_matrix(batch: Sequence[Sequence[Scalar]]) -> tuple[tuple[float, ...], ...]:
    rows = _matrix(batch)
    return tuple(tuple(value.data for value in row) for row in rows)


@dataclass(slots=True)
class RunningMoments:
    """Track exponential-moving feature means/variances for evaluation."""

    size: int
    momentum: float = 0.1
    means: list[float] = field(init=False)
    variances: list[float] = field(init=False)
    initialized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size <= 0:
            raise ValueError("size must be a positive integer")
        self.momentum = float(self.momentum)
        if not math.isfinite(self.momentum) or not 0.0 < self.momentum <= 1.0:
            raise ValueError("momentum must be in (0, 1]")
        self.means = [0.0] * self.size
        self.variances = [1.0] * self.size

    def update(self, batch: Sequence[Sequence[Scalar]]) -> None:
        rows = _numeric_matrix(batch)
        if len(rows[0]) != self.size:
            raise ValueError(f"expected {self.size} features, received {len(rows[0])}")
        count = len(rows)
        batch_means = [
            math.fsum(row[column] for row in rows) / count
            for column in range(self.size)
        ]
        batch_variances = [
            math.fsum(
                (row[column] - batch_means[column]) ** 2 for row in rows
            )
            / count
            for column in range(self.size)
        ]

        if not self.initialized:
            self.means[:] = batch_means
            self.variances[:] = batch_variances
            self.initialized = True
            return

        keep = 1.0 - self.momentum
        for column in range(self.size):
            self.means[column] = (
                keep * self.means[column] + self.momentum * batch_means[column]
            )
            self.variances[column] = (
                keep * self.variances[column]
                + self.momentum * batch_variances[column]
            )

    def normalize(
        self,
        values: Sequence[Scalar],
        *,
        epsilon: float = 1e-5,
        gamma: Sequence[Value] | None = None,
        beta: Sequence[Value] | None = None,
    ) -> tuple[Value, ...]:
        if not self.initialized:
            raise RuntimeError("running moments must be updated before evaluation")
        row = tuple(_value(item) for item in values)
        if len(row) != self.size:
            raise ValueError(f"expected {self.size} features, received {len(row)}")
        eps = _epsilon(epsilon)
        scale = _affine(gamma, self.size, default=1.0, name="gamma")
        shift = _affine(beta, self.size, default=0.0, name="beta")
        return tuple(
            ((value - self.means[index]) / math.sqrt(self.variances[index] + eps))
            * scale[index]
            + shift[index]
            for index, value in enumerate(row)
        )
