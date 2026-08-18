"""Tiny scalar reverse-mode automatic differentiation engine for Part 011."""

from __future__ import annotations

import math
from typing import Callable, Iterable

Number = int | float


class Value:
    """A scalar value that records a computational graph and its gradient.

    The implementation is intentionally small and educational. Each operation
    creates a new ``Value`` with parent links and a local backward rule. Calling
    ``backward`` topologically traverses the graph in reverse and applies the
    chain rule.
    """

    def __init__(
        self,
        data: Number,
        *,
        _children: Iterable["Value"] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        normalized = float(data)
        if not math.isfinite(normalized):
            raise ValueError("Value data must be finite")
        self.data = normalized
        self.grad = 0.0
        self.label = label
        self._op = _op
        self._prev = tuple(_children)
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        suffix = f", label={self.label!r}" if self.label else ""
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}{suffix})"

    @staticmethod
    def _coerce(other: "Value | Number") -> "Value":
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other: "Value | Number") -> "Value":
        other = self._coerce(other)
        out = Value(self.data + other.data, _children=(self, other), _op="+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Number) -> "Value":
        return self + other

    def __neg__(self) -> "Value":
        return self * -1.0

    def __sub__(self, other: "Value | Number") -> "Value":
        return self + (-self._coerce(other))

    def __rsub__(self, other: Number) -> "Value":
        return self._coerce(other) - self

    def __mul__(self, other: "Value | Number") -> "Value":
        other = self._coerce(other)
        out = Value(self.data * other.data, _children=(self, other), _op="*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Number) -> "Value":
        return self * other

    def __pow__(self, exponent: Number) -> "Value":
        power = float(exponent)
        if not math.isfinite(power):
            raise ValueError("power must be finite")
        if self.data < 0.0 and not power.is_integer():
            raise ValueError("fractional powers of negative scalar Values are unsupported")
        if self.data == 0.0 and power <= 0.0:
            raise ValueError("zero cannot be raised to a non-positive power")

        result = self.data**power
        if not math.isfinite(result):
            raise ValueError("power operation produced a non-finite result")
        out = Value(result, _children=(self,), _op=f"**{power:g}")

        def _backward() -> None:
            if power == 0.0:
                local = 0.0
            else:
                local = power * (self.data ** (power - 1.0))
            self.grad += local * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other: "Value | Number") -> "Value":
        other = self._coerce(other)
        if other.data == 0.0:
            raise ZeroDivisionError("division by zero Value")
        return self * (other ** -1.0)

    def __rtruediv__(self, other: Number) -> "Value":
        if self.data == 0.0:
            raise ZeroDivisionError("division by zero Value")
        return self._coerce(other) * (self ** -1.0)

    def exp(self) -> "Value":
        try:
            result = math.exp(self.data)
        except OverflowError as exc:
            raise ValueError("exp operation overflowed") from exc
        out = Value(result, _children=(self,), _op="exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Value":
        if self.data <= 0.0:
            raise ValueError("log requires a positive Value")
        out = Value(math.log(self.data), _children=(self,), _op="log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        result = math.tanh(self.data)
        out = Value(result, _children=(self,), _op="tanh")

        def _backward() -> None:
            self.grad += (1.0 - out.data * out.data) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(max(0.0, self.data), _children=(self,), _op="relu")

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Value":
        if self.data >= 0.0:
            probability = 1.0 / (1.0 + math.exp(-self.data))
        else:
            exp_value = math.exp(self.data)
            probability = exp_value / (1.0 + exp_value)
        out = Value(probability, _children=(self,), _op="sigmoid")

        def _backward() -> None:
            self.grad += out.data * (1.0 - out.data) * out.grad

        out._backward = _backward
        return out

    def topological_order(self) -> tuple["Value", ...]:
        """Return graph nodes from leaves to this value without duplicates."""

        order: list[Value] = []
        visited: set[Value] = set()

        def visit(node: Value) -> None:
            if node in visited:
                return
            visited.add(node)
            for parent in node._prev:
                visit(parent)
            order.append(node)

        visit(self)
        return tuple(order)

    def zero_grad(self, *, graph: bool = True) -> None:
        """Clear this gradient, or every gradient in its graph by default."""

        nodes = self.topological_order() if graph else (self,)
        for node in nodes:
            node.grad = 0.0

    def backward(self, gradient: Number = 1.0, *, clear_grads: bool = True) -> None:
        """Run reverse-mode autodiff from this scalar output."""

        seed = float(gradient)
        if not math.isfinite(seed):
            raise ValueError("backward gradient must be finite")
        order = self.topological_order()
        if clear_grads:
            for node in order:
                node.grad = 0.0
        self.grad = seed
        for node in reversed(order):
            node._backward()


def graph_summary(root: Value) -> dict[str, int]:
    """Return simple node/edge/operation counts for an educational graph."""

    nodes = root.topological_order()
    return {
        "nodes": len(nodes),
        "edges": sum(len(node._prev) for node in nodes),
        "operations": sum(bool(node._op) for node in nodes),
    }
