"""Dependency-free 2D convolution and pooling primitives for NeuralForge Part 021.

The functions operate on single-channel rectangular matrices represented by
nested Python sequences. They intentionally favor explicit loops and validation
over performance so learners can inspect the mechanics before using tensor
frameworks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Number = int | float
Matrix = tuple[tuple[float, ...], ...]


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _pair(value: int | tuple[int, int], *, name: str, allow_zero: bool = False) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{name} must be an integer or a pair")
        first, second = value
    else:
        first = second = value
    validator = _non_negative_int if allow_zero else _positive_int
    return validator(first, name=f"{name}[0]"), validator(second, name=f"{name}[1]")


def as_matrix(values: Sequence[Sequence[Number]], *, name: str = "matrix") -> Matrix:
    """Validate a non-empty rectangular finite numeric matrix."""

    if not values:
        raise ValueError(f"{name} must contain at least one row")
    rows: list[tuple[float, ...]] = []
    width: int | None = None
    for row_index, row in enumerate(values):
        if not row:
            raise ValueError(f"{name} row {row_index} must not be empty")
        normalized = tuple(float(item) for item in row)
        if any(not math.isfinite(item) for item in normalized):
            raise ValueError(f"{name} must contain only finite values")
        if width is None:
            width = len(normalized)
        elif len(normalized) != width:
            raise ValueError(f"{name} must be rectangular")
        rows.append(normalized)
    return tuple(rows)


def spatial_output_size(
    input_size: int,
    kernel_size: int,
    *,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """Return one spatial output dimension for convolution/correlation."""

    n = _positive_int(input_size, name="input_size")
    k = _positive_int(kernel_size, name="kernel_size")
    s = _positive_int(stride, name="stride")
    p = _non_negative_int(padding, name="padding")
    d = _positive_int(dilation, name="dilation")
    effective_kernel = d * (k - 1) + 1
    numerator = n + 2 * p - effective_kernel
    if numerator < 0:
        raise ValueError("effective kernel is larger than the padded input")
    return numerator // s + 1


def output_shape_2d(
    input_shape: tuple[int, int],
    kernel_shape: tuple[int, int],
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> tuple[int, int]:
    """Return ``(height, width)`` after a 2D convolution/correlation."""

    if len(input_shape) != 2 or len(kernel_shape) != 2:
        raise ValueError("input_shape and kernel_shape must each contain two dimensions")
    sh, sw = _pair(stride, name="stride")
    ph, pw = _pair(padding, name="padding", allow_zero=True)
    dh, dw = _pair(dilation, name="dilation")
    return (
        spatial_output_size(input_shape[0], kernel_shape[0], stride=sh, padding=ph, dilation=dh),
        spatial_output_size(input_shape[1], kernel_shape[1], stride=sw, padding=pw, dilation=dw),
    )


def pad2d(
    image: Sequence[Sequence[Number]],
    padding: int | tuple[int, int],
    *,
    value: Number = 0.0,
) -> Matrix:
    """Pad equally on both sides of each spatial axis."""

    matrix = as_matrix(image, name="image")
    ph, pw = _pair(padding, name="padding", allow_zero=True)
    fill = float(value)
    if not math.isfinite(fill):
        raise ValueError("padding value must be finite")
    if ph == 0 and pw == 0:
        return matrix
    width = len(matrix[0]) + 2 * pw
    border = tuple(fill for _ in range(width))
    rows = [border for _ in range(ph)]
    side = (fill,) * pw
    rows.extend(side + row + side for row in matrix)
    rows.extend(border for _ in range(ph))
    return tuple(rows)


def cross_correlate2d(
    image: Sequence[Sequence[Number]],
    kernel: Sequence[Sequence[Number]],
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> Matrix:
    """Apply the operation commonly called convolution in deep-learning APIs.

    The kernel is *not* flipped; mathematically this is cross-correlation.
    """

    x = as_matrix(image, name="image")
    w = as_matrix(kernel, name="kernel")
    sh, sw = _pair(stride, name="stride")
    ph, pw = _pair(padding, name="padding", allow_zero=True)
    dh, dw = _pair(dilation, name="dilation")
    out_h, out_w = output_shape_2d(
        (len(x), len(x[0])),
        (len(w), len(w[0])),
        stride=(sh, sw),
        padding=(ph, pw),
        dilation=(dh, dw),
    )
    padded = pad2d(x, (ph, pw))
    result: list[tuple[float, ...]] = []
    for out_y in range(out_h):
        row: list[float] = []
        start_y = out_y * sh
        for out_x in range(out_w):
            start_x = out_x * sw
            total = 0.0
            for ky, kernel_row in enumerate(w):
                source_y = start_y + ky * dh
                for kx, weight in enumerate(kernel_row):
                    source_x = start_x + kx * dw
                    total += padded[source_y][source_x] * weight
            row.append(total)
        result.append(tuple(row))
    return tuple(result)


def convolution2d(
    image: Sequence[Sequence[Number]],
    kernel: Sequence[Sequence[Number]],
    **kwargs: object,
) -> Matrix:
    """Apply mathematical 2D convolution by flipping the kernel spatially."""

    w = as_matrix(kernel, name="kernel")
    flipped = tuple(tuple(reversed(row)) for row in reversed(w))
    return cross_correlate2d(image, flipped, **kwargs)


def _pool2d(
    image: Sequence[Sequence[Number]],
    *,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None,
    reducer: str,
) -> Matrix:
    x = as_matrix(image, name="image")
    kh, kw = _pair(kernel_size, name="kernel_size")
    sh, sw = _pair(kernel_size if stride is None else stride, name="stride")
    out_h, out_w = output_shape_2d(
        (len(x), len(x[0])),
        (kh, kw),
        stride=(sh, sw),
    )
    result: list[tuple[float, ...]] = []
    for out_y in range(out_h):
        row: list[float] = []
        for out_x in range(out_w):
            values = [
                x[out_y * sh + ky][out_x * sw + kx]
                for ky in range(kh)
                for kx in range(kw)
            ]
            row.append(max(values) if reducer == "max" else math.fsum(values) / len(values))
        result.append(tuple(row))
    return tuple(result)


def max_pool2d(
    image: Sequence[Sequence[Number]],
    *,
    kernel_size: int | tuple[int, int] = 2,
    stride: int | tuple[int, int] | None = None,
) -> Matrix:
    """Apply non-overlapping max pooling by default."""

    return _pool2d(image, kernel_size=kernel_size, stride=stride, reducer="max")


def average_pool2d(
    image: Sequence[Sequence[Number]],
    *,
    kernel_size: int | tuple[int, int] = 2,
    stride: int | tuple[int, int] | None = None,
) -> Matrix:
    """Apply non-overlapping average pooling by default."""

    return _pool2d(image, kernel_size=kernel_size, stride=stride, reducer="mean")
