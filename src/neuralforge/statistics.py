"""Dependency-free probability and statistics helpers for NeuralForge Part 006."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass


def _finite_values(values: Sequence[float], *, name: str = "values") -> tuple[float, ...]:
    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one value")
    try:
        normalized = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values") from exc
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} must contain only finite values")
    return normalized


def mean(values: Sequence[float]) -> float:
    data = _finite_values(values)
    return math.fsum(data) / len(data)


def variance(values: Sequence[float], *, sample: bool = False) -> float:
    """Return population variance by default or sample variance when requested."""

    data = _finite_values(values)
    if sample and len(data) < 2:
        raise ValueError("sample variance requires at least two values")
    center = mean(data)
    denominator = len(data) - 1 if sample else len(data)
    return math.fsum((value - center) ** 2 for value in data) / denominator


def standard_deviation(values: Sequence[float], *, sample: bool = False) -> float:
    return math.sqrt(variance(values, sample=sample))


def covariance(left: Sequence[float], right: Sequence[float], *, sample: bool = False) -> float:
    x = _finite_values(left, name="left values")
    y = _finite_values(right, name="right values")
    if len(x) != len(y):
        raise ValueError("left and right values must have the same length")
    if sample and len(x) < 2:
        raise ValueError("sample covariance requires at least two paired values")

    mean_x = mean(x)
    mean_y = mean(y)
    denominator = len(x) - 1 if sample else len(x)
    return math.fsum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / denominator


def correlation(left: Sequence[float], right: Sequence[float]) -> float:
    x = _finite_values(left, name="left values")
    y = _finite_values(right, name="right values")
    if len(x) != len(y):
        raise ValueError("left and right values must have the same length")

    std_x = standard_deviation(x)
    std_y = standard_deviation(y)
    if std_x == 0.0 or std_y == 0.0:
        raise ValueError("correlation is undefined for a constant variable")
    result = covariance(x, y) / (std_x * std_y)
    return max(-1.0, min(1.0, result))


def normal_pdf(value: float, *, mean_value: float = 0.0, std: float = 1.0) -> float:
    x = float(value)
    mu = float(mean_value)
    sigma = float(std)
    if not all(math.isfinite(item) for item in (x, mu, sigma)):
        raise ValueError("normal distribution parameters must be finite")
    if sigma <= 0.0:
        raise ValueError("std must be greater than zero")
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def bernoulli_log_likelihood(outcomes: Sequence[int], probability: float) -> float:
    if len(outcomes) == 0:
        raise ValueError("outcomes must contain at least one observation")
    if any(outcome not in (0, 1) for outcome in outcomes):
        raise ValueError("Bernoulli outcomes must be 0 or 1")
    p = float(probability)
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be strictly between 0 and 1")
    successes = sum(outcomes)
    failures = len(outcomes) - successes
    return successes * math.log(p) + failures * math.log1p(-p)


@dataclass(frozen=True, slots=True)
class BootstrapEstimate:
    observed: float
    lower: float
    upper: float
    confidence: float
    resamples: int


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 42,
) -> BootstrapEstimate:
    """Estimate a percentile bootstrap confidence interval for the mean."""

    data = _finite_values(values)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 100:
        raise ValueError("resamples must be an integer of at least 100")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    generator = random.Random(seed)
    size = len(data)
    estimates = []
    for _ in range(resamples):
        sample = [data[generator.randrange(size)] for _ in range(size)]
        estimates.append(mean(sample))
    estimates.sort()

    alpha = (1.0 - confidence) / 2.0
    lower_index = max(0, min(resamples - 1, int(alpha * resamples)))
    upper_index = max(0, min(resamples - 1, int((1.0 - alpha) * resamples) - 1))

    return BootstrapEstimate(
        observed=mean(data),
        lower=estimates[lower_index],
        upper=estimates[upper_index],
        confidence=confidence,
        resamples=resamples,
    )
