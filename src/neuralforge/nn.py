"""Small neural-network modules backed by NeuralForge scalar autograd."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from .autograd import Value

Activation = Literal["linear", "tanh", "relu", "sigmoid"]


def _activate(value: Value, activation: Activation) -> Value:
    if activation == "linear":
        return value
    if activation == "tanh":
        return value.tanh()
    if activation == "relu":
        return value.relu()
    if activation == "sigmoid":
        return value.sigmoid()
    raise ValueError(f"unsupported activation: {activation!r}")


def _as_values(inputs: Sequence[Value | float | int]) -> tuple[Value, ...]:
    return tuple(item if isinstance(item, Value) else Value(item) for item in inputs)


class Module:
    """Minimal base class for modules that own trainable scalar parameters."""

    def parameters(self) -> tuple[Value, ...]:
        return ()

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.grad = 0.0


@dataclass(slots=True)
class Neuron(Module):
    weights: tuple[Value, ...]
    bias: Value
    activation: Activation = "tanh"

    @classmethod
    def random(
        cls,
        input_size: int,
        *,
        activation: Activation = "tanh",
        generator: random.Random | None = None,
        scale: float | None = None,
        label_prefix: str = "",
    ) -> "Neuron":
        if isinstance(input_size, bool) or not isinstance(input_size, int) or input_size <= 0:
            raise ValueError("input_size must be a positive integer")
        rng = generator or random.Random()
        init_scale = float(scale) if scale is not None else math.sqrt(1.0 / input_size)
        if not math.isfinite(init_scale) or init_scale <= 0.0:
            raise ValueError("scale must be finite and positive")
        _activate(Value(0.0), activation)  # validate activation name

        weights = tuple(
            Value(
                rng.uniform(-init_scale, init_scale),
                label=f"{label_prefix}w{index}",
            )
            for index in range(input_size)
        )
        bias = Value(0.0, label=f"{label_prefix}b")
        return cls(weights=weights, bias=bias, activation=activation)

    def __call__(self, inputs: Sequence[Value | float | int]) -> Value:
        values = _as_values(inputs)
        if len(values) != len(self.weights):
            raise ValueError(
                f"expected {len(self.weights)} inputs, received {len(values)}"
            )
        preactivation = sum(
            (weight * value for weight, value in zip(self.weights, values)),
            self.bias,
        )
        return _activate(preactivation, self.activation)

    def parameters(self) -> tuple[Value, ...]:
        return (*self.weights, self.bias)


@dataclass(slots=True)
class Layer(Module):
    neurons: tuple[Neuron, ...]

    @classmethod
    def random(
        cls,
        input_size: int,
        output_size: int,
        *,
        activation: Activation = "tanh",
        generator: random.Random | None = None,
        label_prefix: str = "layer.",
    ) -> "Layer":
        if isinstance(output_size, bool) or not isinstance(output_size, int) or output_size <= 0:
            raise ValueError("output_size must be a positive integer")
        rng = generator or random.Random()
        neurons = tuple(
            Neuron.random(
                input_size,
                activation=activation,
                generator=rng,
                label_prefix=f"{label_prefix}n{index}.",
            )
            for index in range(output_size)
        )
        return cls(neurons=neurons)

    def __call__(self, inputs: Sequence[Value | float | int]) -> tuple[Value, ...]:
        return tuple(neuron(inputs) for neuron in self.neurons)

    def parameters(self) -> tuple[Value, ...]:
        return tuple(
            parameter
            for neuron in self.neurons
            for parameter in neuron.parameters()
        )


class MLP(Module):
    """A fully connected multilayer perceptron built from scalar ``Value`` nodes."""

    def __init__(
        self,
        input_size: int,
        layer_sizes: Sequence[int],
        *,
        hidden_activation: Activation = "tanh",
        output_activation: Activation = "linear",
        seed: int = 42,
    ) -> None:
        if isinstance(input_size, bool) or not isinstance(input_size, int) or input_size <= 0:
            raise ValueError("input_size must be a positive integer")
        if not layer_sizes:
            raise ValueError("layer_sizes must contain at least one output size")
        if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in layer_sizes):
            raise ValueError("all layer sizes must be positive integers")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")

        rng = random.Random(seed)
        sizes = (input_size, *tuple(layer_sizes))
        layers: list[Layer] = []
        for index, (nin, nout) in enumerate(zip(sizes, sizes[1:])):
            activation = output_activation if index == len(layer_sizes) - 1 else hidden_activation
            layers.append(
                Layer.random(
                    nin,
                    nout,
                    activation=activation,
                    generator=rng,
                    label_prefix=f"L{index}.",
                )
            )
        self.layers = tuple(layers)

    def __call__(
        self, inputs: Sequence[Value | float | int]
    ) -> Value | tuple[Value, ...]:
        values = _as_values(inputs)
        current: tuple[Value, ...] = values
        for layer in self.layers:
            current = layer(current)
        return current[0] if len(current) == 1 else current

    def parameters(self) -> tuple[Value, ...]:
        return tuple(
            parameter
            for layer in self.layers
            for parameter in layer.parameters()
        )


def mean_squared_error(
    predictions: Iterable[Value], targets: Iterable[float | int]
) -> Value:
    prediction_values = tuple(predictions)
    target_values = tuple(float(target) for target in targets)
    if not prediction_values:
        raise ValueError("predictions must contain at least one Value")
    if len(prediction_values) != len(target_values):
        raise ValueError("predictions and targets must have the same length")
    losses = tuple((prediction - target) ** 2 for prediction, target in zip(prediction_values, target_values))
    return sum(losses, Value(0.0)) / len(losses)


def binary_cross_entropy_loss(
    probabilities: Iterable[Value], targets: Iterable[int], *, epsilon: float = 1e-12
) -> Value:
    probability_values = tuple(probabilities)
    target_values = tuple(targets)
    if not probability_values:
        raise ValueError("probabilities must contain at least one Value")
    if len(probability_values) != len(target_values):
        raise ValueError("probabilities and targets must have the same length")
    if any(target not in (0, 1) for target in target_values):
        raise ValueError("binary targets must be 0 or 1")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be between 0 and 0.5")

    terms: list[Value] = []
    for probability, target in zip(probability_values, target_values):
        # Sigmoid outputs are normally strictly inside (0, 1). The explicit
        # guard provides a clear error if another caller supplies an invalid node.
        if not 0.0 < probability.data < 1.0:
            raise ValueError("probability Values must be strictly between 0 and 1")
        if target == 1:
            terms.append(-probability.log())
        else:
            terms.append(-(1.0 - probability).log())
    return sum(terms, Value(0.0)) / len(terms)


def sgd_step(parameters: Iterable[Value], *, learning_rate: float) -> None:
    rate = float(learning_rate)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("learning_rate must be finite and greater than zero")
    for parameter in parameters:
        updated = parameter.data - rate * parameter.grad
        if not math.isfinite(updated):
            raise ValueError("SGD update produced a non-finite parameter")
        parameter.data = updated
