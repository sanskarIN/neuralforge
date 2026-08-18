"""Dependency-free neural-network foundations used by early NeuralForge labs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


def sigmoid(value: float) -> float:
    """Compute a numerically stable logistic sigmoid."""

    if value >= 0:
        denominator = 1.0 + math.exp(-value)
        return 1.0 / denominator
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def binary_cross_entropy(target: int, probability: float, *, epsilon: float = 1e-12) -> float:
    """Compute binary cross entropy for a single example."""

    if target not in (0, 1):
        raise ValueError("target must be 0 or 1")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be between 0 and 0.5")

    p = min(max(probability, epsilon), 1.0 - epsilon)
    return -(target * math.log(p) + (1 - target) * math.log(1.0 - p))


@dataclass(frozen=True, slots=True)
class LogisticNeuron:
    """A single logistic neuron with immutable learned parameters."""

    weights: tuple[float, ...]
    bias: float = 0.0

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("weights must contain at least one value")

    def logit(self, features: Sequence[float]) -> float:
        if len(features) != len(self.weights):
            raise ValueError(
                f"expected {len(self.weights)} features, received {len(features)}"
            )
        return sum(weight * feature for weight, feature in zip(self.weights, features)) + self.bias

    def predict_proba(self, features: Sequence[float]) -> float:
        return sigmoid(self.logit(features))

    def predict(self, features: Sequence[float], *, threshold: float = 0.5) -> int:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        return int(self.predict_proba(features) >= threshold)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    neuron: LogisticNeuron
    losses: tuple[float, ...]


def _normalize_dataset(
    features: Iterable[Sequence[float]], targets: Iterable[int]
) -> tuple[list[tuple[float, ...]], list[int]]:
    x_rows = [tuple(float(value) for value in row) for row in features]
    y_values = list(targets)

    if not x_rows:
        raise ValueError("features must contain at least one row")
    if len(x_rows) != len(y_values):
        raise ValueError("features and targets must contain the same number of rows")

    width = len(x_rows[0])
    if width == 0:
        raise ValueError("feature rows must contain at least one value")
    if any(len(row) != width for row in x_rows):
        raise ValueError("all feature rows must have the same width")
    if any(target not in (0, 1) for target in y_values):
        raise ValueError("all targets must be 0 or 1")

    return x_rows, y_values


def train_logistic_neuron(
    features: Iterable[Sequence[float]],
    targets: Iterable[int],
    *,
    learning_rate: float = 0.2,
    epochs: int = 1_000,
) -> TrainingResult:
    """Train one logistic neuron with full-batch gradient descent.

    This small implementation is intentionally dependency-free so learners can
    inspect every operation before moving to NumPy and deep-learning frameworks.
    """

    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be greater than zero")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")

    x_rows, y_values = _normalize_dataset(features, targets)
    width = len(x_rows[0])
    weights = [0.0] * width
    bias = 0.0
    losses: list[float] = []
    count = float(len(x_rows))

    for _ in range(epochs):
        weight_grads = [0.0] * width
        bias_grad = 0.0
        total_loss = 0.0

        for row, target in zip(x_rows, y_values):
            logit = sum(weight * value for weight, value in zip(weights, row)) + bias
            probability = sigmoid(logit)
            error = probability - target
            total_loss += binary_cross_entropy(target, probability)

            for index, value in enumerate(row):
                weight_grads[index] += error * value
            bias_grad += error

        for index in range(width):
            weights[index] -= learning_rate * (weight_grads[index] / count)
        bias -= learning_rate * (bias_grad / count)
        losses.append(total_loss / count)

    return TrainingResult(
        neuron=LogisticNeuron(tuple(weights), bias),
        losses=tuple(losses),
    )
