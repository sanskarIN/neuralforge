"""Small dependency-free tensor-shape helpers for NeuralForge Part 002."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Any


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def infer_shape(value: Any) -> tuple[int, ...]:
    """Infer the rectangular shape of a nested Python sequence.

    Scalars have shape ``()``. Ragged nested sequences raise ``ValueError``
    because they cannot be represented as one dense rectangular tensor.
    """

    if not _is_sequence(value):
        return ()

    length = len(value)
    if length == 0:
        return (0,)

    child_shapes = [infer_shape(item) for item in value]
    first = child_shapes[0]
    if any(shape != first for shape in child_shapes[1:]):
        raise ValueError("ragged nested sequences do not have one dense tensor shape")
    return (length, *first)


def flatten(value: Any) -> list[Any]:
    """Return tensor elements in row-major order as a flat Python list."""

    infer_shape(value)  # validate rectangular structure before flattening
    if not _is_sequence(value):
        return [value]

    result: list[Any] = []
    for item in value:
        result.extend(flatten(item))
    return result


def numel(value: Any) -> int:
    """Return the number of scalar elements in a rectangular nested sequence."""

    shape = infer_shape(value)
    if not shape:
        return 1
    return prod(shape)


def _validate_new_shape(shape: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(shape)
    if any(isinstance(dim, bool) or not isinstance(dim, int) for dim in normalized):
        raise TypeError("shape dimensions must be integers")
    if any(dim <= 0 for dim in normalized):
        raise ValueError("reshape dimensions must be positive integers")
    return normalized


def reshape(value: Any, shape: Sequence[int]) -> Any:
    """Reshape rectangular data using row-major order.

    This educational helper supports positive dimensions and the scalar shape
    ``()``. It deliberately omits inferred ``-1`` dimensions so the element
    count remains explicit for early learners.
    """

    new_shape = _validate_new_shape(shape)
    flat = flatten(value)

    if not new_shape:
        if len(flat) != 1:
            raise ValueError("scalar reshape requires exactly one element")
        return flat[0]

    expected = prod(new_shape)
    if expected != len(flat):
        raise ValueError(
            f"cannot reshape {len(flat)} elements into shape {new_shape} "
            f"({expected} elements required)"
        )

    iterator = iter(flat)

    def build(dimensions: tuple[int, ...]) -> Any:
        if len(dimensions) == 1:
            return [next(iterator) for _ in range(dimensions[0])]
        return [build(dimensions[1:]) for _ in range(dimensions[0])]

    return build(new_shape)
