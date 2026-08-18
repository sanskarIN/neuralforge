"""Dependency-free linear algebra helpers for NeuralForge Part 004."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _vector(values: Sequence[float], *, name: str = "vector") -> tuple[float, ...]:
    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one value")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values") from exc
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _matrix(values: Sequence[Sequence[float]], *, name: str = "matrix") -> tuple[tuple[float, ...], ...]:
    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one row")
    rows = tuple(_vector(row, name=f"{name} row") for row in values)
    width = len(rows[0])
    if any(len(row) != width for row in rows[1:]):
        raise ValueError(f"{name} must be rectangular")
    return rows


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the dot product of two equal-length vectors."""

    a = _vector(left, name="left vector")
    b = _vector(right, name="right vector")
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    return sum(x * y for x, y in zip(a, b))


def l2_norm(values: Sequence[float]) -> float:
    """Return the Euclidean (L2) norm of a vector."""

    vector = _vector(values)
    return math.sqrt(sum(value * value for value in vector))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for two non-zero equal-length vectors."""

    a = _vector(left, name="left vector")
    b = _vector(right, name="right vector")
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    norm_a = l2_norm(a)
    norm_b = l2_norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return dot(a, b) / (norm_a * norm_b)


def transpose(values: Sequence[Sequence[float]]) -> list[list[float]]:
    """Transpose a rectangular matrix."""

    matrix = _matrix(values)
    return [list(column) for column in zip(*matrix)]


def matmul(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Multiply two compatible matrices using explicit loops."""

    a = _matrix(left, name="left matrix")
    b = _matrix(right, name="right matrix")
    if len(a[0]) != len(b):
        raise ValueError(
            f"incompatible matrix shapes: ({len(a)}, {len(a[0])}) and "
            f"({len(b)}, {len(b[0])})"
        )

    b_t = tuple(zip(*b))
    return [
        [sum(x * y for x, y in zip(row, column)) for column in b_t]
        for row in a
    ]


def outer(left: Sequence[float], right: Sequence[float]) -> list[list[float]]:
    """Return the outer product of two vectors."""

    a = _vector(left, name="left vector")
    b = _vector(right, name="right vector")
    return [[x * y for y in b] for x in a]
