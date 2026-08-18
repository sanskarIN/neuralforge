"""Gradient-flow diagnostics for NeuralForge Part 019."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .autograd import Value


@dataclass(frozen=True)
class GradientStats:
    count: int
    finite_count: int
    nonfinite_count: int
    zero_count: int
    zero_fraction: float
    mean: float
    mean_abs: float
    l1_norm: float
    l2_norm: float
    max_abs: float
    min_abs_nonzero: float


@dataclass(frozen=True)
class GradientHealth:
    status: str
    message: str
    stats: GradientStats


def gradient_stats(parameters: Iterable[Value]) -> GradientStats:
    values = tuple(parameters)
    if not values:
        raise ValueError("at least one parameter is required")

    gradients = [float(parameter.grad) for parameter in values]
    finite = [gradient for gradient in gradients if math.isfinite(gradient)]
    nonfinite_count = len(gradients) - len(finite)
    zero_count = sum(gradient == 0.0 for gradient in finite)
    nonzero_abs = [abs(gradient) for gradient in finite if gradient != 0.0]

    if finite:
        mean = sum(finite) / len(finite)
        l1 = sum(abs(gradient) for gradient in finite)
        l2 = math.sqrt(sum(gradient * gradient for gradient in finite))
        max_abs = max(abs(gradient) for gradient in finite)
        mean_abs = l1 / len(finite)
    else:
        mean = math.nan
        l1 = math.nan
        l2 = math.nan
        max_abs = math.nan
        mean_abs = math.nan

    return GradientStats(
        count=len(gradients),
        finite_count=len(finite),
        nonfinite_count=nonfinite_count,
        zero_count=zero_count,
        zero_fraction=zero_count / len(gradients),
        mean=mean,
        mean_abs=mean_abs,
        l1_norm=l1,
        l2_norm=l2,
        max_abs=max_abs,
        min_abs_nonzero=min(nonzero_abs) if nonzero_abs else 0.0,
    )


def require_finite_gradients(parameters: Iterable[Value]) -> GradientStats:
    stats = gradient_stats(parameters)
    if stats.nonfinite_count:
        raise ValueError(f"found {stats.nonfinite_count} non-finite gradient(s)")
    return stats


def gradient_to_parameter_ratio(parameters: Iterable[Value], *, epsilon: float = 1e-12) -> float:
    values = tuple(parameters)
    if not values:
        raise ValueError("at least one parameter is required")
    eps = float(epsilon)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    stats = require_finite_gradients(values)
    parameter_norm = math.sqrt(sum(parameter.data * parameter.data for parameter in values))
    return stats.l2_norm / max(parameter_norm, eps)


def group_gradient_stats(
    parameters: Iterable[Value],
    *,
    separator: str = ".",
    fallback: str = "unlabeled",
) -> dict[str, GradientStats]:
    if not separator:
        raise ValueError("separator must not be empty")
    grouped: dict[str, list[Value]] = {}
    for parameter in parameters:
        label = parameter.label.strip()
        key = label.split(separator, 1)[0] if label else fallback
        grouped.setdefault(key, []).append(parameter)
    if not grouped:
        raise ValueError("at least one parameter is required")
    return {key: gradient_stats(values) for key, values in grouped.items()}


def assess_gradient_health(
    parameters: Iterable[Value],
    *,
    vanishing_l2: float = 1e-10,
    exploding_max_abs: float = 1e3,
) -> GradientHealth:
    vanish = float(vanishing_l2)
    explode = float(exploding_max_abs)
    if not math.isfinite(vanish) or vanish < 0.0:
        raise ValueError("vanishing_l2 must be non-negative and finite")
    if not math.isfinite(explode) or explode <= 0.0:
        raise ValueError("exploding_max_abs must be positive and finite")

    stats = gradient_stats(parameters)
    if stats.nonfinite_count:
        return GradientHealth("non_finite", "one or more gradients are NaN or infinite", stats)
    if stats.l2_norm <= vanish:
        return GradientHealth("vanishing", f"gradient L2 norm {stats.l2_norm:.3e} is at/below {vanish:.3e}", stats)
    if stats.max_abs >= explode:
        return GradientHealth("exploding", f"maximum absolute gradient {stats.max_abs:.3e} is at/above {explode:.3e}", stats)
    return GradientHealth("healthy", "gradient snapshot is finite and inside configured thresholds", stats)
