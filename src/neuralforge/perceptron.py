"""From-scratch binary perceptron for NeuralForge Part 010."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


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
class Perceptron:
    weights: tuple[float, ...]
    bias: float

    def score(self, features: Sequence[float]) -> float:
        row = tuple(float(value) for value in features)
        if len(row) != len(self.weights):
            raise ValueError(f"expected {len(self.weights)} features, received {len(row)}")
        if not all(math.isfinite(value) for value in row):
            raise ValueError("features must contain only finite values")
        return math.fsum(weight * value for weight, value in zip(self.weights, row)) + self.bias

    def predict(self, features: Sequence[float]) -> int:
        return int(self.score(features) >= 0.0)


@dataclass(frozen=True, slots=True)
class PerceptronResult:
    model: Perceptron
    mistakes_per_epoch: tuple[int, ...]


def train_perceptron(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    learning_rate: float = 1.0,
    epochs: int = 100,
    shuffle: bool = True,
    seed: int = 42,
) -> PerceptronResult:
    """Train with the classic mistake-driven perceptron update."""

    rows, targets = _dataset(features, labels)
    rate = float(learning_rate)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("learning_rate must be finite and greater than zero")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")

    weights = [0.0] * len(rows[0])
    bias = 0.0
    history: list[int] = []
    order = list(range(len(rows)))
    generator = random.Random(seed)

    for _ in range(epochs):
        if shuffle:
            generator.shuffle(order)
        mistakes = 0
        for index in order:
            row = rows[index]
            target = targets[index]
            score = math.fsum(w * x for w, x in zip(weights, row)) + bias
            prediction = int(score >= 0.0)
            update = rate * (target - prediction)
            if update:
                mistakes += 1
                for column, value in enumerate(row):
                    weights[column] += update * value
                bias += update
        history.append(mistakes)
        if mistakes == 0:
            break

    return PerceptronResult(Perceptron(tuple(weights), bias), tuple(history))
