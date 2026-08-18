"""Loss functions and output-layer design helpers for NeuralForge Part 016."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from .autograd import Value

LossName = Literal[
    "mse",
    "mae",
    "huber",
    "binary_cross_entropy_with_logits",
    "multiclass_cross_entropy",
]


def _values(items: Iterable[Value]) -> tuple[Value, ...]:
    values = tuple(items)
    if not values:
        raise ValueError("predictions/logits must contain at least one Value")
    if any(not isinstance(value, Value) for value in values):
        raise TypeError("predictions/logits must be Value objects")
    return values


def _targets(items: Iterable[float | int]) -> tuple[float, ...]:
    values = tuple(float(item) for item in items)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("targets must be finite")
    return values


def _absolute(value: Value) -> Value:
    """Differentiable |x| with the educational zero subgradient at x == 0."""

    return value.relu() + (-value).relu()


def softplus(value: Value) -> Value:
    """Numerically stable ``log(1 + exp(x))`` for a scalar ``Value``."""

    if not isinstance(value, Value):
        raise TypeError("softplus expects a Value")
    if value.data >= 0.0:
        return value + (1.0 + (-value).exp()).log()
    return (1.0 + value.exp()).log()


def mean_squared_error(
    predictions: Iterable[Value], targets: Iterable[float | int]
) -> Value:
    prediction_values = _values(predictions)
    target_values = _targets(targets)
    if len(prediction_values) != len(target_values):
        raise ValueError("predictions and targets must have the same length")
    return sum(
        ((prediction - target) ** 2 for prediction, target in zip(prediction_values, target_values)),
        Value(0.0),
    ) / len(prediction_values)


def mean_absolute_error(
    predictions: Iterable[Value], targets: Iterable[float | int]
) -> Value:
    prediction_values = _values(predictions)
    target_values = _targets(targets)
    if len(prediction_values) != len(target_values):
        raise ValueError("predictions and targets must have the same length")
    return sum(
        (_absolute(prediction - target) for prediction, target in zip(prediction_values, target_values)),
        Value(0.0),
    ) / len(prediction_values)


def huber_loss(
    predictions: Iterable[Value],
    targets: Iterable[float | int],
    *,
    delta: float = 1.0,
) -> Value:
    prediction_values = _values(predictions)
    target_values = _targets(targets)
    threshold = float(delta)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("delta must be finite and greater than zero")
    if len(prediction_values) != len(target_values):
        raise ValueError("predictions and targets must have the same length")

    terms: list[Value] = []
    for prediction, target in zip(prediction_values, target_values):
        error = prediction - target
        magnitude = abs(error.data)
        if magnitude <= threshold:
            terms.append(0.5 * (error**2))
        else:
            terms.append(threshold * (_absolute(error) - 0.5 * threshold))
    return sum(terms, Value(0.0)) / len(terms)


def binary_cross_entropy_with_logits(
    logits: Iterable[Value], targets: Iterable[int]
) -> Value:
    logit_values = _values(logits)
    target_values = tuple(targets)
    if len(logit_values) != len(target_values):
        raise ValueError("logits and targets must have the same length")
    if any(target not in (0, 1) for target in target_values):
        raise ValueError("binary targets must be 0 or 1")

    # BCE(z, y) = softplus(z) - y*z. This avoids explicitly taking log(sigmoid(z)).
    terms = tuple(
        softplus(logit) - target * logit
        for logit, target in zip(logit_values, target_values)
    )
    return sum(terms, Value(0.0)) / len(terms)


def multiclass_cross_entropy(
    logits: Sequence[Sequence[Value]], targets: Iterable[int]
) -> Value:
    rows = tuple(tuple(row) for row in logits)
    target_values = tuple(targets)
    if not rows:
        raise ValueError("logits must contain at least one row")
    if len(rows) != len(target_values):
        raise ValueError("logit rows and targets must have the same length")

    losses: list[Value] = []
    expected_width: int | None = None
    for row, target in zip(rows, target_values):
        if not row:
            raise ValueError("each logit row must contain at least one class")
        if any(not isinstance(value, Value) for value in row):
            raise TypeError("multiclass logits must be Value objects")
        if expected_width is None:
            expected_width = len(row)
        elif len(row) != expected_width:
            raise ValueError("all logit rows must have the same width")
        if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < len(row):
            raise ValueError("each class target must be a valid integer class index")

        shift = max(value.data for value in row)
        shifted_exp = tuple((value - shift).exp() for value in row)
        log_partition = sum(shifted_exp, Value(0.0)).log() + shift
        losses.append(log_partition - row[target])

    return sum(losses, Value(0.0)) / len(losses)


@dataclass(frozen=True, slots=True)
class OutputLossPairing:
    task: str
    output_activation: str
    loss: str
    target_format: str
    note: str


_PAIRINGS: dict[str, OutputLossPairing] = {
    "regression": OutputLossPairing(
        task="regression",
        output_activation="linear",
        loss="mse / mae / huber",
        target_format="finite real values",
        note="Choose the regression loss based on sensitivity to outliers and the metric you care about.",
    ),
    "binary_classification": OutputLossPairing(
        task="binary_classification",
        output_activation="linear logits",
        loss="binary_cross_entropy_with_logits",
        target_format="0 or 1",
        note="Train from logits for numerical stability; apply sigmoid only when probabilities are needed.",
    ),
    "multiclass_classification": OutputLossPairing(
        task="multiclass_classification",
        output_activation="one linear logit per class",
        loss="multiclass_cross_entropy",
        target_format="integer class index",
        note="Cross-entropy internally normalizes logits; a separate training-time softmax is unnecessary.",
    ),
}


def recommended_output_loss(task: str) -> OutputLossPairing:
    """Return the recommended educational output/loss pairing for a task."""

    try:
        return _PAIRINGS[task]
    except KeyError as exc:
        supported = ", ".join(sorted(_PAIRINGS))
        raise ValueError(f"unsupported task {task!r}; choose one of: {supported}") from exc
