"""Weight initialization and signal-propagation diagnostics for Part 018."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Initialization = Literal[
    "zeros",
    "xavier_uniform",
    "xavier_normal",
    "he_uniform",
    "he_normal",
    "lecun_normal",
]
ActivationName = Literal["linear", "tanh", "relu"]


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("seed must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class InitializationPlan:
    scheme: Initialization
    fan_in: int
    fan_out: int
    distribution: str
    scale: float


def initialization_plan(
    fan_in: int,
    fan_out: int,
    scheme: Initialization,
) -> InitializationPlan:
    incoming = _positive_int(fan_in, name="fan_in")
    outgoing = _positive_int(fan_out, name="fan_out")

    if scheme == "zeros":
        return InitializationPlan(scheme, incoming, outgoing, "constant", 0.0)
    if scheme == "xavier_uniform":
        return InitializationPlan(
            scheme,
            incoming,
            outgoing,
            "uniform_bound",
            math.sqrt(6.0 / (incoming + outgoing)),
        )
    if scheme == "xavier_normal":
        return InitializationPlan(
            scheme,
            incoming,
            outgoing,
            "normal_std",
            math.sqrt(2.0 / (incoming + outgoing)),
        )
    if scheme == "he_uniform":
        return InitializationPlan(
            scheme,
            incoming,
            outgoing,
            "uniform_bound",
            math.sqrt(6.0 / incoming),
        )
    if scheme == "he_normal":
        return InitializationPlan(
            scheme,
            incoming,
            outgoing,
            "normal_std",
            math.sqrt(2.0 / incoming),
        )
    if scheme == "lecun_normal":
        return InitializationPlan(
            scheme,
            incoming,
            outgoing,
            "normal_std",
            math.sqrt(1.0 / incoming),
        )
    raise ValueError(f"unsupported initialization scheme: {scheme!r}")


def initialize_matrix(
    output_size: int,
    input_size: int,
    *,
    scheme: Initialization = "xavier_uniform",
    seed: int = 42,
) -> tuple[tuple[float, ...], ...]:
    """Create an ``[output_size, input_size]`` weight matrix deterministically."""

    outputs = _positive_int(output_size, name="output_size")
    inputs = _positive_int(input_size, name="input_size")
    rng = random.Random(_seed(seed))
    plan = initialization_plan(inputs, outputs, scheme)

    if plan.distribution == "constant":
        return tuple(tuple(0.0 for _ in range(inputs)) for _ in range(outputs))
    if plan.distribution == "uniform_bound":
        return tuple(
            tuple(rng.uniform(-plan.scale, plan.scale) for _ in range(inputs))
            for _ in range(outputs)
        )
    return tuple(
        tuple(rng.gauss(0.0, plan.scale) for _ in range(inputs))
        for _ in range(outputs)
    )


def population_variance(values: Sequence[float | int]) -> float:
    numbers = tuple(float(value) for value in values)
    if not numbers:
        raise ValueError("values must contain at least one number")
    if any(not math.isfinite(value) for value in numbers):
        raise ValueError("values must be finite")
    mean = math.fsum(numbers) / len(numbers)
    return math.fsum((value - mean) ** 2 for value in numbers) / len(numbers)


def _activate(value: float, activation: ActivationName) -> float:
    if activation == "linear":
        return value
    if activation == "tanh":
        return math.tanh(value)
    if activation == "relu":
        return max(0.0, value)
    raise ValueError(f"unsupported activation: {activation!r}")


def _validate_batch(batch: Sequence[Sequence[float | int]]) -> tuple[tuple[float, ...], ...]:
    rows = tuple(tuple(float(value) for value in row) for row in batch)
    if not rows or not rows[0]:
        raise ValueError("batch must contain at least one non-empty row")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("batch rows must have the same width")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("batch values must be finite")
    return rows


def _linear_batch(
    batch: tuple[tuple[float, ...], ...],
    weights: tuple[tuple[float, ...], ...],
    activation: ActivationName,
) -> tuple[tuple[float, ...], ...]:
    output: list[tuple[float, ...]] = []
    for row in batch:
        transformed = tuple(
            _activate(math.fsum(weight * value for weight, value in zip(neuron, row)), activation)
            for neuron in weights
        )
        output.append(transformed)
    return tuple(output)


@dataclass(frozen=True, slots=True)
class SignalPropagationProfile:
    scheme: Initialization
    activation: ActivationName
    widths: tuple[int, ...]
    variances: tuple[float, ...]

    @property
    def variance_ratio(self) -> float:
        start = self.variances[0]
        if start == 0.0:
            return math.inf if self.variances[-1] != 0.0 else 1.0
        return self.variances[-1] / start


def signal_propagation_profile(
    batch: Sequence[Sequence[float | int]],
    layer_sizes: Sequence[int],
    *,
    scheme: Initialization,
    activation: ActivationName,
    seed: int = 42,
) -> SignalPropagationProfile:
    """Propagate a numeric batch and record population variance after each layer."""

    current = _validate_batch(batch)
    sizes = tuple(_positive_int(size, name="layer size") for size in layer_sizes)
    if not sizes:
        raise ValueError("layer_sizes must contain at least one layer")
    _activate(0.0, activation)  # validate activation even for an empty-valued path
    base_seed = _seed(seed)

    widths = [len(current[0])]
    variances = [population_variance([value for row in current for value in row])]
    input_width = len(current[0])

    for layer_index, output_width in enumerate(sizes):
        weights = initialize_matrix(
            output_width,
            input_width,
            scheme=scheme,
            seed=base_seed + layer_index,
        )
        current = _linear_batch(current, weights, activation)
        widths.append(output_width)
        variances.append(population_variance([value for row in current for value in row]))
        input_width = output_width

    return SignalPropagationProfile(
        scheme=scheme,
        activation=activation,
        widths=tuple(widths),
        variances=tuple(variances),
    )
