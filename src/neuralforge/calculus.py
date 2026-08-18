"""Dependency-free numerical calculus helpers for NeuralForge Part 005."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

ScalarFunction = Callable[[float], float]
VectorFunction = Callable[[Sequence[float]], float]
GradientFunction = Callable[[Sequence[float]], Sequence[float]]


def _step(value: float) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("step must be a finite positive number") from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError("step must be a finite positive number")
    return normalized


def _finite(value: float, *, name: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def numerical_derivative(function: ScalarFunction, x: float, *, step: float = 1e-5) -> float:
    """Approximate a scalar derivative with a central finite difference."""

    h = _step(step)
    point = _finite(x, name="x")
    upper = _finite(function(point + h), name="function output")
    lower = _finite(function(point - h), name="function output")
    return (upper - lower) / (2.0 * h)


def numerical_gradient(
    function: VectorFunction,
    point: Sequence[float],
    *,
    step: float = 1e-5,
) -> tuple[float, ...]:
    """Approximate a gradient with one central difference per dimension."""

    h = _step(step)
    if len(point) == 0:
        raise ValueError("point must contain at least one coordinate")
    base = [_finite(value, name="point coordinate") for value in point]
    gradient: list[float] = []

    for index in range(len(base)):
        upper_point = base.copy()
        lower_point = base.copy()
        upper_point[index] += h
        lower_point[index] -= h

        upper = _finite(function(upper_point), name="function output")
        lower = _finite(function(lower_point), name="function output")
        gradient.append((upper - lower) / (2.0 * h))

    return tuple(gradient)


@dataclass(frozen=True, slots=True)
class GradientCheckResult:
    """Result of comparing an analytical gradient with a numerical estimate."""

    analytical: tuple[float, ...]
    numerical: tuple[float, ...]
    absolute_errors: tuple[float, ...]
    max_absolute_error: float
    passed: bool


def check_gradient(
    function: VectorFunction,
    analytical_gradient: GradientFunction,
    point: Sequence[float],
    *,
    step: float = 1e-5,
    atol: float = 1e-6,
    rtol: float = 1e-4,
) -> GradientCheckResult:
    """Compare an analytical gradient with a central-difference estimate."""

    if atol < 0.0 or rtol < 0.0:
        raise ValueError("atol and rtol must be non-negative")

    numerical = numerical_gradient(function, point, step=step)
    analytical_values = tuple(
        _finite(value, name="analytical gradient value")
        for value in analytical_gradient(point)
    )

    if len(analytical_values) != len(numerical):
        raise ValueError("analytical gradient has the wrong number of dimensions")

    errors = tuple(
        abs(analytical - numeric)
        for analytical, numeric in zip(analytical_values, numerical)
    )
    passed = all(
        error <= atol + rtol * abs(numeric)
        for error, numeric in zip(errors, numerical)
    )

    return GradientCheckResult(
        analytical=analytical_values,
        numerical=numerical,
        absolute_errors=errors,
        max_absolute_error=max(errors, default=0.0),
        passed=passed,
    )
