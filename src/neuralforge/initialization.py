"""Initialization and forward signal-propagation tools for NeuralForge Part 018."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

Initialization = Literal[
    "zeros",
    "uniform",
    "xavier_uniform",
    "xavier_normal",
    "he_uniform",
    "he_normal",
    "lecun_normal",
]
ActivationName = Literal["linear", "tanh", "relu", "sigmoid"]


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_positive(name: str, value: float) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return normalized


def initialization_scale(
    scheme: Initialization,
    fan_in: int,
    fan_out: int,
    *,
    uniform_scale: float = 0.05,
) -> float:
    """Return the bound (uniform) or standard deviation (normal) for a scheme."""

    incoming = _positive_int("fan_in", fan_in)
    outgoing = _positive_int("fan_out", fan_out)
    if scheme == "zeros":
        return 0.0
    if scheme == "uniform":
        return _finite_positive("uniform_scale", uniform_scale)
    if scheme == "xavier_uniform":
        return math.sqrt(6.0 / (incoming + outgoing))
    if scheme == "xavier_normal":
        return math.sqrt(2.0 / (incoming + outgoing))
    if scheme == "he_uniform":
        return math.sqrt(6.0 / incoming)
    if scheme == "he_normal":
        return math.sqrt(2.0 / incoming)
    if scheme == "lecun_normal":
        return math.sqrt(1.0 / incoming)
    raise ValueError(f"unsupported initialization scheme: {scheme!r}")


def recommend_initialization(activation: str) -> Initialization:
    normalized = activation.strip().lower()
    if normalized == "relu":
        return "he_normal"
    if normalized in {"tanh", "sigmoid", "linear"}:
        return "xavier_uniform"
    raise ValueError(f"unsupported activation: {activation!r}")


def initialize_matrix(
    fan_in: int,
    fan_out: int,
    *,
    scheme: Initialization = "xavier_uniform",
    seed: int = 42,
    uniform_scale: float = 0.05,
) -> tuple[tuple[float, ...], ...]:
    """Create a deterministic dense weight matrix shaped ``fan_out x fan_in``."""

    incoming = _positive_int("fan_in", fan_in)
    outgoing = _positive_int("fan_out", fan_out)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    scale = initialization_scale(scheme, incoming, outgoing, uniform_scale=uniform_scale)
    if scheme == "zeros":
        return tuple(tuple(0.0 for _ in range(incoming)) for _ in range(outgoing))

    rng = random.Random(seed)
    rows: list[tuple[float, ...]] = []
    for _ in range(outgoing):
        if scheme in {"uniform", "xavier_uniform", "he_uniform"}:
            row = tuple(rng.uniform(-scale, scale) for _ in range(incoming))
        else:
            row = tuple(rng.gauss(0.0, scale) for _ in range(incoming))
        rows.append(row)
    return tuple(rows)


def dense_forward(
    inputs: Sequence[float],
    weights: Sequence[Sequence[float]],
    biases: Sequence[float] | None = None,
) -> tuple[float, ...]:
    if not weights:
        raise ValueError("weights must contain at least one output row")
    width = len(inputs)
    if width == 0:
        raise ValueError("inputs must not be empty")
    if any(len(row) != width for row in weights):
        raise ValueError("every weight row must match the input width")
    if biases is None:
        bias_values = (0.0,) * len(weights)
    else:
        if len(biases) != len(weights):
            raise ValueError("bias count must match output rows")
        bias_values = tuple(float(value) for value in biases)

    output: list[float] = []
    for row, bias in zip(weights, bias_values):
        value = sum(float(weight) * float(item) for weight, item in zip(row, inputs)) + bias
        if not math.isfinite(value):
            raise ValueError("dense forward pass produced a non-finite value")
        output.append(value)
    return tuple(output)


def activate(values: Iterable[float], activation: ActivationName) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        item = float(value)
        if not math.isfinite(item):
            raise ValueError("activation inputs must be finite")
        if activation == "linear":
            activated = item
        elif activation == "tanh":
            activated = math.tanh(item)
        elif activation == "relu":
            activated = max(0.0, item)
        elif activation == "sigmoid":
            if item >= 0.0:
                activated = 1.0 / (1.0 + math.exp(-item))
            else:
                exponential = math.exp(item)
                activated = exponential / (1.0 + exponential)
        else:
            raise ValueError(f"unsupported activation: {activation!r}")
        result.append(activated)
    return tuple(result)


@dataclass(frozen=True)
class SignalStats:
    layer: int
    width: int
    mean: float
    variance: float
    minimum: float
    maximum: float
    zero_fraction: float


def _signal_stats(layer: int, batch: Sequence[Sequence[float]]) -> SignalStats:
    flattened = [float(value) for row in batch for value in row]
    if not flattened:
        raise ValueError("batch must contain values")
    mean = sum(flattened) / len(flattened)
    variance = sum((value - mean) ** 2 for value in flattened) / len(flattened)
    zero_fraction = sum(value == 0.0 for value in flattened) / len(flattened)
    return SignalStats(
        layer=layer,
        width=len(batch[0]),
        mean=mean,
        variance=variance,
        minimum=min(flattened),
        maximum=max(flattened),
        zero_fraction=zero_fraction,
    )


def propagate_signal(
    batch: Sequence[Sequence[float]],
    layer_sizes: Sequence[int],
    *,
    activation: ActivationName,
    scheme: Initialization | None = None,
    seed: int = 42,
) -> tuple[SignalStats, ...]:
    """Propagate a numeric batch through random dense layers and summarize signals."""

    if not batch or not batch[0]:
        raise ValueError("batch must be a non-empty rectangular matrix")
    width = len(batch[0])
    if any(len(row) != width for row in batch):
        raise ValueError("batch must be rectangular")
    if not layer_sizes:
        raise ValueError("layer_sizes must not be empty")
    if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in layer_sizes):
        raise ValueError("layer sizes must be positive integers")
    selected = scheme or recommend_initialization(activation)

    current = [tuple(float(value) for value in row) for row in batch]
    stats: list[SignalStats] = [_signal_stats(0, current)]
    fan_in = width
    for layer_index, fan_out in enumerate(layer_sizes, start=1):
        weights = initialize_matrix(
            fan_in,
            fan_out,
            scheme=selected,
            seed=seed + layer_index,
        )
        current = [activate(dense_forward(row, weights), activation) for row in current]
        stats.append(_signal_stats(layer_index, current))
        fan_in = fan_out
    return tuple(stats)
