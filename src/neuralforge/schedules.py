"""Learning-rate schedules and optimization control for NeuralForge Part 017."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


def _positive(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result


def _non_negative(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _step(value: int, *, name: str = "step") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def constant_learning_rate(base_lr: float, step: int = 0) -> float:
    _step(step)
    return _positive(base_lr, name="base_lr")


def step_decay(
    base_lr: float,
    step: int,
    *,
    step_size: int,
    gamma: float = 0.1,
) -> float:
    rate = _positive(base_lr, name="base_lr")
    current = _step(step)
    if isinstance(step_size, bool) or not isinstance(step_size, int) or step_size <= 0:
        raise ValueError("step_size must be a positive integer")
    decay = float(gamma)
    if not math.isfinite(decay) or not 0.0 < decay <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    return rate * decay ** (current // step_size)


def exponential_decay(base_lr: float, step: int, *, decay_rate: float) -> float:
    rate = _positive(base_lr, name="base_lr")
    current = _step(step)
    decay = float(decay_rate)
    if not math.isfinite(decay) or not 0.0 < decay <= 1.0:
        raise ValueError("decay_rate must be in (0, 1]")
    return rate * decay**current


def cosine_decay(
    base_lr: float,
    step: int,
    *,
    total_steps: int,
    min_lr: float = 0.0,
) -> float:
    maximum = _positive(base_lr, name="base_lr")
    current = _step(step)
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")
    minimum = _non_negative(min_lr, name="min_lr")
    if minimum > maximum:
        raise ValueError("min_lr cannot exceed base_lr")

    progress = min(current, total_steps) / total_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum + (maximum - minimum) * cosine


def linear_warmup(
    target_lr: float,
    step: int,
    *,
    warmup_steps: int,
    start_lr: float = 0.0,
) -> float:
    target = _positive(target_lr, name="target_lr")
    current = _step(step)
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps <= 0:
        raise ValueError("warmup_steps must be a positive integer")
    start = _non_negative(start_lr, name="start_lr")
    if start > target:
        raise ValueError("start_lr cannot exceed target_lr")
    progress = min(current, warmup_steps) / warmup_steps
    return start + (target - start) * progress


def warmup_cosine_decay(
    peak_lr: float,
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    start_lr: float = 0.0,
    min_lr: float = 0.0,
) -> float:
    peak = _positive(peak_lr, name="peak_lr")
    current = _step(step)
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps <= 0:
        raise ValueError("warmup_steps must be a positive integer")
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= warmup_steps:
        raise ValueError("total_steps must be an integer greater than warmup_steps")
    start = _non_negative(start_lr, name="start_lr")
    minimum = _non_negative(min_lr, name="min_lr")
    if start > peak or minimum > peak:
        raise ValueError("start_lr and min_lr cannot exceed peak_lr")

    if current <= warmup_steps:
        return linear_warmup(
            peak,
            current,
            warmup_steps=warmup_steps,
            start_lr=start,
        )
    return cosine_decay(
        peak,
        current - warmup_steps,
        total_steps=total_steps - warmup_steps,
        min_lr=minimum,
    )


def apply_learning_rate(optimizer: object, learning_rate: float) -> float:
    """Set a validated learning rate on a NeuralForge-style optimizer."""

    rate = _positive(learning_rate, name="learning_rate")
    if not hasattr(optimizer, "learning_rate"):
        raise TypeError("optimizer does not expose a learning_rate attribute")
    setattr(optimizer, "learning_rate", rate)
    return rate


@dataclass(slots=True)
class ReduceLROnPlateau:
    """Small validation-metric controller that reduces LR after a plateau."""

    factor: float = 0.5
    patience: int = 2
    min_lr: float = 1e-6
    min_delta: float = 0.0
    mode: Literal["min", "max"] = "min"
    best: float | None = None
    bad_steps: int = 0
    reductions: int = 0

    def __post_init__(self) -> None:
        factor = float(self.factor)
        if not math.isfinite(factor) or not 0.0 < factor < 1.0:
            raise ValueError("factor must be in (0, 1)")
        if isinstance(self.patience, bool) or not isinstance(self.patience, int) or self.patience <= 0:
            raise ValueError("patience must be a positive integer")
        minimum = _positive(self.min_lr, name="min_lr")
        delta = _non_negative(self.min_delta, name="min_delta")
        if self.mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.factor = factor
        self.min_lr = minimum
        self.min_delta = delta

    def _improved(self, metric: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return metric < self.best - self.min_delta
        return metric > self.best + self.min_delta

    def update(self, metric: float, current_lr: float) -> float:
        value = float(metric)
        if not math.isfinite(value):
            raise ValueError("metric must be finite")
        rate = _positive(current_lr, name="current_lr")
        if rate < self.min_lr:
            raise ValueError("current_lr cannot be below min_lr")

        if self._improved(value):
            self.best = value
            self.bad_steps = 0
            return rate

        self.bad_steps += 1
        if self.bad_steps < self.patience:
            return rate

        self.bad_steps = 0
        reduced = max(self.min_lr, rate * self.factor)
        if reduced < rate:
            self.reductions += 1
        return reduced
