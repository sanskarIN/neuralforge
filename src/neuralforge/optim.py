"""Educational optimizers for NeuralForge Part 013."""

from __future__ import annotations

import math
from collections.abc import Iterable

from .autograd import Value


def _positive(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result


def _unit_interval(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"{name} must be in [0, 1)")
    return result


def _non_negative(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _parameters(values: Iterable[Value]) -> tuple[Value, ...]:
    parameters: list[Value] = []
    seen: set[int] = set()
    for value in values:
        if not isinstance(value, Value):
            raise TypeError("optimizer parameters must be Value objects")
        identity = id(value)
        if identity not in seen:
            seen.add(identity)
            parameters.append(value)
    if not parameters:
        raise ValueError("optimizer requires at least one parameter")
    return tuple(parameters)


class Optimizer:
    """Base class for stateful scalar-parameter optimizers."""

    def __init__(self, parameters: Iterable[Value]) -> None:
        self.parameters = _parameters(parameters)

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = 0.0

    @staticmethod
    def _checked_gradient(parameter: Value) -> float:
        gradient = float(parameter.grad)
        if not math.isfinite(gradient):
            raise ValueError("optimizer encountered a non-finite gradient")
        return gradient

    @staticmethod
    def _assign(parameter: Value, new_value: float) -> None:
        if not math.isfinite(new_value):
            raise ValueError("optimizer update produced a non-finite parameter")
        parameter.data = new_value

    def step(self) -> None:  # pragma: no cover - abstract behavior
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Value],
        *,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(parameters)
        self.learning_rate = _positive(learning_rate, name="learning_rate")
        self.weight_decay = _non_negative(weight_decay, name="weight_decay")

    def step(self) -> None:
        for parameter in self.parameters:
            gradient = self._checked_gradient(parameter)
            if self.weight_decay:
                gradient += self.weight_decay * parameter.data
            self._assign(
                parameter,
                parameter.data - self.learning_rate * gradient,
            )


class Momentum(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Value],
        *,
        learning_rate: float = 0.01,
        beta: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(parameters)
        self.learning_rate = _positive(learning_rate, name="learning_rate")
        self.beta = _unit_interval(beta, name="beta")
        self.weight_decay = _non_negative(weight_decay, name="weight_decay")
        self.velocity = [0.0] * len(self.parameters)

    def step(self) -> None:
        for index, parameter in enumerate(self.parameters):
            gradient = self._checked_gradient(parameter)
            if self.weight_decay:
                gradient += self.weight_decay * parameter.data
            self.velocity[index] = self.beta * self.velocity[index] + gradient
            self._assign(
                parameter,
                parameter.data - self.learning_rate * self.velocity[index],
            )


class RMSProp(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Value],
        *,
        learning_rate: float = 0.001,
        beta: float = 0.99,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(parameters)
        self.learning_rate = _positive(learning_rate, name="learning_rate")
        self.beta = _unit_interval(beta, name="beta")
        self.epsilon = _positive(epsilon, name="epsilon")
        self.weight_decay = _non_negative(weight_decay, name="weight_decay")
        self.average_squared = [0.0] * len(self.parameters)

    def step(self) -> None:
        for index, parameter in enumerate(self.parameters):
            gradient = self._checked_gradient(parameter)
            if self.weight_decay:
                gradient += self.weight_decay * parameter.data
            self.average_squared[index] = (
                self.beta * self.average_squared[index]
                + (1.0 - self.beta) * gradient * gradient
            )
            denominator = math.sqrt(self.average_squared[index]) + self.epsilon
            self._assign(
                parameter,
                parameter.data - self.learning_rate * gradient / denominator,
            )


class Adam(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Value],
        *,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(parameters)
        self.learning_rate = _positive(learning_rate, name="learning_rate")
        self.beta1 = _unit_interval(beta1, name="beta1")
        self.beta2 = _unit_interval(beta2, name="beta2")
        self.epsilon = _positive(epsilon, name="epsilon")
        self.weight_decay = _non_negative(weight_decay, name="weight_decay")
        self.first_moment = [0.0] * len(self.parameters)
        self.second_moment = [0.0] * len(self.parameters)
        self.timestep = 0

    def step(self) -> None:
        self.timestep += 1
        correction1 = 1.0 - self.beta1**self.timestep
        correction2 = 1.0 - self.beta2**self.timestep

        for index, parameter in enumerate(self.parameters):
            gradient = self._checked_gradient(parameter)
            if self.weight_decay:
                gradient += self.weight_decay * parameter.data

            self.first_moment[index] = (
                self.beta1 * self.first_moment[index]
                + (1.0 - self.beta1) * gradient
            )
            self.second_moment[index] = (
                self.beta2 * self.second_moment[index]
                + (1.0 - self.beta2) * gradient * gradient
            )
            mean_hat = self.first_moment[index] / correction1
            variance_hat = self.second_moment[index] / correction2
            update = self.learning_rate * mean_hat / (
                math.sqrt(variance_hat) + self.epsilon
            )
            self._assign(parameter, parameter.data - update)


def clip_grad_norm(parameters: Iterable[Value], max_norm: float) -> float:
    """Clip gradients by global L2 norm and return the pre-clipping norm."""

    values = _parameters(parameters)
    limit = _positive(max_norm, name="max_norm")
    gradients = []
    for parameter in values:
        gradient = float(parameter.grad)
        if not math.isfinite(gradient):
            raise ValueError("gradient clipping encountered a non-finite gradient")
        gradients.append(gradient)

    norm = math.sqrt(math.fsum(gradient * gradient for gradient in gradients))
    if norm > limit and norm > 0.0:
        scale = limit / norm
        for parameter in values:
            parameter.grad *= scale
    return norm
