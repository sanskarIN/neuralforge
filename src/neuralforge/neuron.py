"""Artificial-neuron primitives for NeuralForge Part 009."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

from .foundations import sigmoid

ActivationName = Literal["identity", "sigmoid", "tanh", "relu", "leaky_relu"]


def identity(value: float) -> float:
    return float(value)


def relu(value: float) -> float:
    return max(0.0, float(value))


def leaky_relu(value: float, *, negative_slope: float = 0.01) -> float:
    slope = float(negative_slope)
    if not math.isfinite(slope) or slope < 0.0:
        raise ValueError("negative_slope must be finite and non-negative")
    x = float(value)
    return x if x >= 0.0 else slope * x


def _activation_function(name: ActivationName) -> Callable[[float], float]:
    functions: dict[str, Callable[[float], float]] = {
        "identity": identity,
        "sigmoid": sigmoid,
        "tanh": math.tanh,
        "relu": relu,
        "leaky_relu": leaky_relu,
    }
    try:
        return functions[name]
    except KeyError as exc:
        supported = ", ".join(functions)
        raise ValueError(f"unsupported activation {name!r}; choose from {supported}") from exc


def _finite_vector(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one value")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values") from exc
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True, slots=True)
class NeuronTrace:
    """Expose the weighted contributions, pre-activation value, and output."""

    contributions: tuple[float, ...]
    weighted_sum: float
    output: float


@dataclass(frozen=True, slots=True)
class ArtificialNeuron:
    """One fully inspectable artificial neuron."""

    weights: tuple[float, ...]
    bias: float = 0.0
    activation: ActivationName = "identity"

    def __post_init__(self) -> None:
        normalized = _finite_vector(self.weights, name="weights")
        object.__setattr__(self, "weights", normalized)
        bias = float(self.bias)
        if not math.isfinite(bias):
            raise ValueError("bias must be finite")
        object.__setattr__(self, "bias", bias)
        _activation_function(self.activation)

    def trace(self, inputs: Sequence[float]) -> NeuronTrace:
        features = _finite_vector(inputs, name="inputs")
        if len(features) != len(self.weights):
            raise ValueError(
                f"expected {len(self.weights)} inputs, received {len(features)}"
            )
        contributions = tuple(
            weight * value for weight, value in zip(self.weights, features)
        )
        weighted_sum = math.fsum(contributions) + self.bias
        output = _activation_function(self.activation)(weighted_sum)
        return NeuronTrace(
            contributions=contributions,
            weighted_sum=weighted_sum,
            output=output,
        )

    def forward(self, inputs: Sequence[float]) -> float:
        return self.trace(inputs).output


def activation_derivative(name: ActivationName, value: float) -> float:
    """Return d activation(z) / dz at the supplied pre-activation value ``z``."""

    z = float(value)
    if not math.isfinite(z):
        raise ValueError("value must be finite")
    if name == "identity":
        return 1.0
    if name == "sigmoid":
        probability = sigmoid(z)
        return probability * (1.0 - probability)
    if name == "tanh":
        output = math.tanh(z)
        return 1.0 - output * output
    if name == "relu":
        return 1.0 if z > 0.0 else 0.0
    if name == "leaky_relu":
        return 1.0 if z >= 0.0 else 0.01
    _activation_function(name)  # raises a consistent error
    raise AssertionError("unreachable")
