"""Gradient-flow diagnostics for NeuralForge Part 019."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from .autograd import Value

GradientHealth = Literal["healthy", "zero", "vanishing", "exploding", "nonfinite"]


def _parameters(values: Iterable[Value]) -> tuple[Value, ...]:
    parameters = tuple(values)
    if not parameters:
        raise ValueError("parameters must contain at least one Value")
    if any(not isinstance(value, Value) for value in parameters):
        raise TypeError("parameters must be Value objects")
    return parameters


@dataclass(frozen=True, slots=True)
class GradientStats:
    count: int
    finite_count: int
    nonfinite_count: int
    zero_count: int
    mean_abs: float
    l2_norm: float
    max_abs: float

    @property
    def zero_fraction(self) -> float:
        return self.zero_count / self.count


@dataclass(frozen=True, slots=True)
class LayerGradientReport:
    name: str
    status: GradientHealth
    stats: GradientStats


@dataclass(frozen=True, slots=True)
class GradientFlowReport:
    layers: tuple[LayerGradientReport, ...]
    overall_status: GradientHealth

    @property
    def has_nonfinite(self) -> bool:
        return any(layer.stats.nonfinite_count for layer in self.layers)


def gradient_statistics(parameters: Iterable[Value]) -> GradientStats:
    values = _parameters(parameters)
    gradients = tuple(float(parameter.grad) for parameter in values)
    finite = tuple(gradient for gradient in gradients if math.isfinite(gradient))
    nonfinite_count = len(gradients) - len(finite)
    zero_count = sum(gradient == 0.0 for gradient in finite)

    if not finite:
        return GradientStats(
            count=len(values),
            finite_count=0,
            nonfinite_count=nonfinite_count,
            zero_count=0,
            mean_abs=math.inf,
            l2_norm=math.inf,
            max_abs=math.inf,
        )

    magnitudes = tuple(abs(gradient) for gradient in finite)
    return GradientStats(
        count=len(values),
        finite_count=len(finite),
        nonfinite_count=nonfinite_count,
        zero_count=zero_count,
        mean_abs=math.fsum(magnitudes) / len(magnitudes),
        l2_norm=math.sqrt(math.fsum(gradient * gradient for gradient in finite)),
        max_abs=max(magnitudes),
    )


def classify_gradient_health(
    stats: GradientStats,
    *,
    vanishing_threshold: float = 1e-8,
    exploding_threshold: float = 100.0,
) -> GradientHealth:
    vanishing = float(vanishing_threshold)
    exploding = float(exploding_threshold)
    if not math.isfinite(vanishing) or vanishing <= 0.0:
        raise ValueError("vanishing_threshold must be finite and greater than zero")
    if not math.isfinite(exploding) or exploding <= vanishing:
        raise ValueError("exploding_threshold must be finite and greater than vanishing_threshold")

    if stats.nonfinite_count:
        return "nonfinite"
    if stats.max_abs == 0.0:
        return "zero"
    if stats.max_abs < vanishing:
        return "vanishing"
    if stats.max_abs > exploding:
        return "exploding"
    return "healthy"


def gradient_flow_report(
    groups: Mapping[str, Iterable[Value]],
    *,
    vanishing_threshold: float = 1e-8,
    exploding_threshold: float = 100.0,
) -> GradientFlowReport:
    if not groups:
        raise ValueError("groups must contain at least one named parameter group")

    reports: list[LayerGradientReport] = []
    for name, parameters in groups.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("gradient group names must be non-empty strings")
        stats = gradient_statistics(parameters)
        reports.append(
            LayerGradientReport(
                name=name,
                status=classify_gradient_health(
                    stats,
                    vanishing_threshold=vanishing_threshold,
                    exploding_threshold=exploding_threshold,
                ),
                stats=stats,
            )
        )

    statuses = {report.status for report in reports}
    if "nonfinite" in statuses:
        overall: GradientHealth = "nonfinite"
    elif "exploding" in statuses:
        overall = "exploding"
    elif "vanishing" in statuses:
        overall = "vanishing"
    elif statuses == {"zero"}:
        overall = "zero"
    else:
        overall = "healthy"
    return GradientFlowReport(layers=tuple(reports), overall_status=overall)


def relative_update_ratio(parameters: Iterable[Value], learning_rate: float) -> float:
    """Return ``lr * ||grad|| / ||parameter||`` for a parameter group."""

    values = _parameters(parameters)
    rate = float(learning_rate)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("learning_rate must be finite and greater than zero")

    parameter_norm = math.sqrt(math.fsum(value.data * value.data for value in values))
    if not math.isfinite(parameter_norm):
        raise ValueError("parameter values must be finite")

    stats = gradient_statistics(values)
    if stats.nonfinite_count:
        return math.inf
    if parameter_norm == 0.0:
        return math.inf if stats.l2_norm > 0.0 else 0.0
    return rate * stats.l2_norm / parameter_norm


def mlp_parameter_groups(model: object) -> dict[str, tuple[Value, ...]]:
    """Extract layer parameter groups from a NeuralForge MLP-like object."""

    layers = getattr(model, "layers", None)
    if not isinstance(layers, tuple) or not layers:
        raise TypeError("model must expose a non-empty tuple of layers")
    groups: dict[str, tuple[Value, ...]] = {}
    for index, layer in enumerate(layers):
        parameters_method = getattr(layer, "parameters", None)
        if not callable(parameters_method):
            raise TypeError("each layer must expose a parameters() method")
        parameters = tuple(parameters_method())
        _parameters(parameters)
        groups[f"layer_{index}"] = parameters
    return groups
