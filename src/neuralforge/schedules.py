"""Learning-rate schedules and optimization control for NeuralForge Part 017."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _positive_finite(name: str, value: float) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return normalized


def _nonnegative_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def constant_lr(base_lr: float, step: int = 0) -> float:
    _nonnegative_int("step", step)
    return _positive_finite("base_lr", base_lr)


def step_decay(base_lr: float, step: int, *, step_size: int, gamma: float = 0.1) -> float:
    base = _positive_finite("base_lr", base_lr)
    current = _nonnegative_int("step", step)
    if not isinstance(step_size, int) or isinstance(step_size, bool) or step_size <= 0:
        raise ValueError("step_size must be a positive integer")
    decay = float(gamma)
    if not math.isfinite(decay) or not 0.0 < decay <= 1.0:
        raise ValueError("gamma must be finite and in (0, 1]")
    return base * (decay ** (current // step_size))


def exponential_decay(base_lr: float, step: int, *, gamma: float = 0.99) -> float:
    base = _positive_finite("base_lr", base_lr)
    current = _nonnegative_int("step", step)
    decay = float(gamma)
    if not math.isfinite(decay) or not 0.0 < decay <= 1.0:
        raise ValueError("gamma must be finite and in (0, 1]")
    return base * (decay**current)


def cosine_decay(base_lr: float, step: int, *, total_steps: int, min_lr: float = 0.0) -> float:
    base = _positive_finite("base_lr", base_lr)
    current = _nonnegative_int("step", step)
    if not isinstance(total_steps, int) or isinstance(total_steps, bool) or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")
    floor = float(min_lr)
    if not math.isfinite(floor) or floor < 0.0 or floor > base:
        raise ValueError("min_lr must be finite and in [0, base_lr]")
    progress = min(current, total_steps) / total_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (base - floor) * cosine


def linear_warmup(base_lr: float, step: int, *, warmup_steps: int, start_lr: float = 0.0) -> float:
    base = _positive_finite("base_lr", base_lr)
    current = _nonnegative_int("step", step)
    if not isinstance(warmup_steps, int) or isinstance(warmup_steps, bool) or warmup_steps <= 0:
        raise ValueError("warmup_steps must be a positive integer")
    start = float(start_lr)
    if not math.isfinite(start) or start < 0.0 or start > base:
        raise ValueError("start_lr must be finite and in [0, base_lr]")
    progress = min(current + 1, warmup_steps) / warmup_steps
    return start + (base - start) * progress


def warmup_cosine_decay(
    base_lr: float,
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
    start_lr: float = 0.0,
) -> float:
    if total_steps <= warmup_steps:
        raise ValueError("total_steps must be greater than warmup_steps")
    current = _nonnegative_int("step", step)
    if current < warmup_steps:
        return linear_warmup(base_lr, current, warmup_steps=warmup_steps, start_lr=start_lr)
    return cosine_decay(
        base_lr,
        current - warmup_steps,
        total_steps=total_steps - warmup_steps,
        min_lr=min_lr,
    )


@dataclass(frozen=True)
class PlateauUpdate:
    learning_rate: float
    reduced: bool
    best_metric: float
    bad_epochs: int


class ReduceLROnPlateau:
    """Reduce a learning rate after validation progress stalls."""

    def __init__(
        self,
        initial_lr: float,
        *,
        factor: float = 0.5,
        patience: int = 2,
        min_lr: float = 0.0,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        self.learning_rate = _positive_finite("initial_lr", initial_lr)
        reduction = float(factor)
        if not math.isfinite(reduction) or not 0.0 < reduction < 1.0:
            raise ValueError("factor must be finite and in (0, 1)")
        if not isinstance(patience, int) or isinstance(patience, bool) or patience < 1:
            raise ValueError("patience must be a positive integer")
        floor = float(min_lr)
        if not math.isfinite(floor) or floor < 0.0 or floor > self.learning_rate:
            raise ValueError("min_lr must be finite and in [0, initial_lr]")
        delta = float(min_delta)
        if not math.isfinite(delta) or delta < 0.0:
            raise ValueError("min_delta must be non-negative and finite")
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")

        self.factor = reduction
        self.patience = patience
        self.min_lr = floor
        self.min_delta = delta
        self.mode = normalized_mode
        self.best_metric = math.inf if normalized_mode == "min" else -math.inf
        self.bad_epochs = 0

    def _improved(self, metric: float) -> bool:
        if self.mode == "min":
            return metric < self.best_metric - self.min_delta
        return metric > self.best_metric + self.min_delta

    def update(self, metric: float) -> PlateauUpdate:
        current = float(metric)
        if not math.isfinite(current):
            raise ValueError("metric must be finite")

        reduced = False
        if self._improved(current):
            self.best_metric = current
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                new_lr = max(self.min_lr, self.learning_rate * self.factor)
                reduced = new_lr < self.learning_rate
                self.learning_rate = new_lr
                self.bad_epochs = 0

        return PlateauUpdate(
            learning_rate=self.learning_rate,
            reduced=reduced,
            best_metric=self.best_metric,
            bad_epochs=self.bad_epochs,
        )
