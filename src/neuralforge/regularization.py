"""Regularization and generalization helpers for NeuralForge Part 014."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .autograd import Value


def _parameters(values: Iterable[Value]) -> tuple[Value, ...]:
    parameters = tuple(values)
    if not parameters:
        raise ValueError("at least one parameter is required")
    if any(not isinstance(parameter, Value) for parameter in parameters):
        raise TypeError("parameters must be Value objects")
    return parameters


def _strength(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def l1_penalty(parameters: Iterable[Value], *, strength: float) -> Value:
    """Return ``strength * sum(abs(parameter))`` as an autodiff Value.

    Absolute value is expressed as ``relu(x) + relu(-x)``. At exactly zero,
    this educational engine uses the zero subgradient.
    """

    values = _parameters(parameters)
    coefficient = _strength(strength, name="strength")
    absolute_values = tuple(
        parameter.relu() + (-parameter).relu() for parameter in values
    )
    return coefficient * sum(absolute_values, Value(0.0))


def l2_penalty(parameters: Iterable[Value], *, strength: float) -> Value:
    """Return ``0.5 * strength * sum(parameter**2)`` as an autodiff Value."""

    values = _parameters(parameters)
    coefficient = _strength(strength, name="strength")
    squares = tuple(parameter**2 for parameter in values)
    return 0.5 * coefficient * sum(squares, Value(0.0))


def regularized_loss(
    base_loss: Value,
    parameters: Iterable[Value],
    *,
    l1: float = 0.0,
    l2: float = 0.0,
) -> Value:
    """Add optional L1/L2 penalties to an existing scalar loss."""

    if not isinstance(base_loss, Value):
        raise TypeError("base_loss must be a Value")
    values = _parameters(parameters)
    l1_strength = _strength(l1, name="l1")
    l2_strength = _strength(l2, name="l2")

    result = base_loss
    if l1_strength:
        result = result + l1_penalty(values, strength=l1_strength)
    if l2_strength:
        result = result + l2_penalty(values, strength=l2_strength)
    return result


@dataclass(frozen=True, slots=True)
class DropoutResult:
    outputs: tuple[Value, ...]
    kept: tuple[bool, ...]
    scale: float


def inverted_dropout(
    values: Sequence[Value],
    *,
    drop_probability: float,
    training: bool = True,
    seed: int = 42,
) -> DropoutResult:
    """Apply deterministic-seeded inverted dropout to scalar activations.

    During training, kept activations are divided by ``1 - p`` so their
    expectation matches evaluation-time activations. During evaluation, values
    pass through unchanged.
    """

    if not values:
        raise ValueError("values must contain at least one activation")
    if any(not isinstance(value, Value) for value in values):
        raise TypeError("dropout inputs must be Value objects")
    probability = float(drop_probability)
    if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
        raise ValueError("drop_probability must be in [0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    if not training or probability == 0.0:
        return DropoutResult(tuple(values), tuple(True for _ in values), 1.0)

    generator = random.Random(seed)
    keep_probability = 1.0 - probability
    scale = 1.0 / keep_probability
    outputs: list[Value] = []
    kept: list[bool] = []

    for value in values:
        is_kept = generator.random() >= probability
        kept.append(is_kept)
        outputs.append(value * scale if is_kept else Value(0.0))

    return DropoutResult(tuple(outputs), tuple(kept), scale)


@dataclass(slots=True)
class EarlyStopping:
    """Track a validation metric and signal after sustained non-improvement."""

    patience: int = 5
    min_delta: float = 0.0
    best: float | None = None
    bad_epochs: int = 0
    stopped: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.patience, bool) or not isinstance(self.patience, int) or self.patience <= 0:
            raise ValueError("patience must be a positive integer")
        self.min_delta = _strength(self.min_delta, name="min_delta")

    def update(self, validation_loss: float) -> bool:
        """Record a loss where lower is better and return stop status."""

        value = float(validation_loss)
        if not math.isfinite(value):
            raise ValueError("validation_loss must be finite")
        if self.stopped:
            return True

        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.stopped = True
        return self.stopped

    def reset(self) -> None:
        self.best = None
        self.bad_epochs = 0
        self.stopped = False


def generalization_gap(training_loss: float, validation_loss: float) -> float:
    """Return validation loss minus training loss."""

    train = float(training_loss)
    validation = float(validation_loss)
    if not math.isfinite(train) or not math.isfinite(validation):
        raise ValueError("loss values must be finite")
    return validation - train


def parameter_l2_norm(parameters: Iterable[Value]) -> float:
    values = _parameters(parameters)
    return math.sqrt(math.fsum(parameter.data**2 for parameter in values))
