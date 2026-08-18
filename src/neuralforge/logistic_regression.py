"""From-scratch binary logistic regression for NeuralForge Part 010."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .foundations import binary_cross_entropy, sigmoid


def _dataset(features: Sequence[Sequence[float]], labels: Sequence[int]):
    if not features or len(features) != len(labels):
        raise ValueError("features and labels must be non-empty and have equal length")
    rows = tuple(tuple(float(value) for value in row) for row in features)
    if not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("features must form a non-empty rectangular matrix")
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError("features must contain only finite values")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("labels must be 0 or 1")
    return rows, tuple(labels)


@dataclass(frozen=True, slots=True)
class LogisticRegression:
    weights: tuple[float, ...]
    bias: float

    def logit(self, features: Sequence[float]) -> float:
        row = tuple(float(value) for value in features)
        if len(row) != len(self.weights):
            raise ValueError(f"expected {len(self.weights)} features, received {len(row)}")
        if not all(math.isfinite(value) for value in row):
            raise ValueError("features must contain only finite values")
        return math.fsum(weight * value for weight, value in zip(self.weights, row)) + self.bias

    def predict_probability(self, features: Sequence[float]) -> float:
        return sigmoid(self.logit(features))

    def predict(self, features: Sequence[float], *, threshold: float = 0.5) -> int:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be strictly between 0 and 1")
        return int(self.predict_probability(features) >= threshold)


@dataclass(frozen=True, slots=True)
class LogisticRegressionResult:
    model: LogisticRegression
    losses: tuple[float, ...]


def train_logistic_regression(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    learning_rate: float = 0.2,
    epochs: int = 1_000,
    l2: float = 0.0,
) -> LogisticRegressionResult:
    """Train logistic regression with full-batch gradient descent."""

    rows, targets = _dataset(features, labels)
    rate = float(learning_rate)
    penalty = float(l2)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("learning_rate must be finite and greater than zero")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if not math.isfinite(penalty) or penalty < 0.0:
        raise ValueError("l2 must be finite and non-negative")

    width = len(rows[0])
    weights = [0.0] * width
    bias = 0.0
    losses: list[float] = []
    count = float(len(rows))

    for _ in range(epochs):
        gradients = [0.0] * width
        bias_gradient = 0.0
        total_loss = 0.0

        for row, target in zip(rows, targets):
            logit = math.fsum(weight * value for weight, value in zip(weights, row)) + bias
            probability = sigmoid(logit)
            error = probability - target
            total_loss += binary_cross_entropy(target, probability)
            for column, value in enumerate(row):
                gradients[column] += error * value
            bias_gradient += error

        squared_norm = math.fsum(weight * weight for weight in weights)
        losses.append((total_loss + 0.5 * penalty * squared_norm) / count)

        for column in range(width):
            gradient = gradients[column] / count + penalty * weights[column] / count
            weights[column] -= rate * gradient
        bias -= rate * bias_gradient / count

    return LogisticRegressionResult(
        LogisticRegression(tuple(weights), bias),
        tuple(losses),
    )
