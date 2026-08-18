"""Loss functions and output-layer design helpers for NeuralForge Part 016."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .autograd import Value

Scalar = Value | int | float


def _as_value(value: Scalar) -> Value:
    return value if isinstance(value, Value) else Value(value)


def _mean(values: Sequence[Value]) -> Value:
    if not values:
        raise ValueError("at least one value is required")
    return sum(values, Value(0.0)) / len(values)


def absolute(value: Scalar) -> Value:
    """Differentiable absolute value with a zero subgradient at exactly zero."""

    item = _as_value(value)
    return item.relu() + (-item).relu()


def mean_squared_error(predictions: Iterable[Scalar], targets: Iterable[Scalar]) -> Value:
    pairs = list(zip(predictions, targets, strict=True))
    return _mean([(_as_value(prediction) - _as_value(target)) ** 2 for prediction, target in pairs])


def mean_absolute_error(predictions: Iterable[Scalar], targets: Iterable[Scalar]) -> Value:
    pairs = list(zip(predictions, targets, strict=True))
    return _mean([absolute(_as_value(prediction) - _as_value(target)) for prediction, target in pairs])


def huber_loss(
    predictions: Iterable[Scalar],
    targets: Iterable[Scalar],
    *,
    delta: float = 1.0,
) -> Value:
    """Mean Huber loss with a quadratic center and linear tails."""

    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be positive and finite")
    pairs = list(zip(predictions, targets, strict=True))
    losses: list[Value] = []
    for prediction, target in pairs:
        error = _as_value(prediction) - _as_value(target)
        magnitude = abs(error.data)
        if magnitude <= delta:
            losses.append(0.5 * (error**2))
        else:
            losses.append(delta * absolute(error) - 0.5 * delta * delta)
    return _mean(losses)


def softplus(value: Scalar) -> Value:
    """Numerically stable ``log(1 + exp(x))`` on a scalar autodiff Value."""

    item = _as_value(value)
    if item.data >= 0.0:
        return item + (Value(1.0) + (-item).exp()).log()
    return (Value(1.0) + item.exp()).log()


def binary_cross_entropy_with_logits(logits: Iterable[Scalar], targets: Iterable[int | float]) -> Value:
    """Mean binary cross-entropy computed directly from logits.

    The stable identity is ``softplus(logit) - target * logit``.
    """

    pairs = list(zip(logits, targets, strict=True))
    losses: list[Value] = []
    for logit, target in pairs:
        target_value = float(target)
        if target_value not in {0.0, 1.0}:
            raise ValueError("binary targets must be 0 or 1")
        logit_value = _as_value(logit)
        losses.append(softplus(logit_value) - target_value * logit_value)
    return _mean(losses)


def logsumexp(logits: Sequence[Scalar]) -> Value:
    """Stable log-sum-exp for a non-empty sequence of logits."""

    values = tuple(_as_value(logit) for logit in logits)
    if not values:
        raise ValueError("logsumexp requires at least one logit")
    maximum = max(value.data for value in values)
    shifted_sum = sum(((value - maximum).exp() for value in values), Value(0.0))
    return shifted_sum.log() + maximum


def categorical_cross_entropy_with_logits(logits: Sequence[Scalar], target_index: int) -> Value:
    """Cross-entropy for one multiclass example from raw logits."""

    values = tuple(_as_value(logit) for logit in logits)
    if not values:
        raise ValueError("at least one class logit is required")
    if not isinstance(target_index, int) or isinstance(target_index, bool):
        raise TypeError("target_index must be an integer")
    if not 0 <= target_index < len(values):
        raise ValueError("target_index is outside the class range")
    return logsumexp(values) - values[target_index]


def mean_categorical_cross_entropy_with_logits(
    batch_logits: Iterable[Sequence[Scalar]],
    target_indices: Iterable[int],
) -> Value:
    pairs = list(zip(batch_logits, target_indices, strict=True))
    return _mean(
        [categorical_cross_entropy_with_logits(logits, target) for logits, target in pairs]
    )


@dataclass(frozen=True)
class OutputDesign:
    """Recommended final activation/objective pairing for a learning task."""

    task: str
    output_units: int
    final_activation: str
    objective: str
    target_format: str
    inference_rule: str


def recommend_output_design(task: str, *, classes: int | None = None) -> OutputDesign:
    """Return a safe default output/objective pairing for common supervised tasks."""

    normalized = task.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == "regression":
        return OutputDesign(
            task="regression",
            output_units=1,
            final_activation="linear",
            objective="mse_or_huber",
            target_format="continuous scalar",
            inference_rule="use the scalar output directly",
        )
    if normalized in {"binary", "binary_classification"}:
        return OutputDesign(
            task="binary_classification",
            output_units=1,
            final_activation="linear_logits",
            objective="binary_cross_entropy_with_logits",
            target_format="0/1 scalar",
            inference_rule="apply sigmoid, then threshold probability",
        )
    if normalized in {"multiclass", "multiclass_classification"}:
        if classes is None or not isinstance(classes, int) or isinstance(classes, bool) or classes < 2:
            raise ValueError("multiclass tasks require classes >= 2")
        return OutputDesign(
            task="multiclass_classification",
            output_units=classes,
            final_activation="linear_logits",
            objective="categorical_cross_entropy_with_logits",
            target_format="integer class index",
            inference_rule="apply softmax for probabilities; argmax for predicted class",
        )
    raise ValueError(f"unsupported task: {task!r}")
