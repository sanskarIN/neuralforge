"""Dependency-free convolution and pooling primitives for NeuralForge Part 021."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

Matrix = tuple[tuple[float, ...], ...]
PaddingMode = Literal["valid", "same"]


def _matrix(values: Sequence[Sequence[float | int]], *, name: str) -> Matrix:
    rows = tuple(tuple(float(value) for value in row) for row in values)
    if not rows or not rows[0]:
        raise ValueError(f"{name} must be a non-empty 2D matrix")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be rectangular")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError(f"{name} values must be finite")
    return rows


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def effective_kernel_size(kernel_size: int, dilation: int = 1) -> int:
    kernel = _positive_int(kernel_size, name="kernel_size")
    rate = _positive_int(dilation, name="dilation")
    return rate * (kernel - 1) + 1


def output_size(
    input_size: int,
    kernel_size: int,
    *,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    size = _positive_int(input_size, name="input_size")
    step = _positive_int(stride, name="stride")
    pad = _non_negative_int(padding, name="padding")
    effective = effective_kernel_size(kernel_size, dilation)
    numerator = size + 2 * pad - effective
    if numerator < 0:
        raise ValueError("effective kernel is larger than the padded input")
    return numerator // step + 1


def same_padding(input_size: int, kernel_size: int, *, stride: int = 1, dilation: int = 1) -> tuple[int, int]:
    size = _positive_int(input_size, name="input_size")
    step = _positive_int(stride, name="stride")
    effective = effective_kernel_size(kernel_size, dilation)
    output = math.ceil(size / step)
    total = max(0, (output - 1) * step + effective - size)
    before = total // 2
    return before, total - before


def pad2d(
    matrix: Sequence[Sequence[float | int]],
    padding: int | tuple[int, int, int, int],
    *,
    value: float = 0.0,
) -> Matrix:
    source = _matrix(matrix, name="matrix")
    fill = float(value)
    if not math.isfinite(fill):
        raise ValueError("padding value must be finite")

    if isinstance(padding, int) and not isinstance(padding, bool):
        amount = _non_negative_int(padding, name="padding")
        top = bottom = left = right = amount
    else:
        if not isinstance(padding, tuple) or len(padding) != 4:
            raise TypeError("padding must be an integer or (top, bottom, left, right)")
        top, bottom, left, right = (
            _non_negative_int(item, name="padding component") for item in padding
        )

    width = len(source[0]) + left + right
    padded: list[tuple[float, ...]] = [tuple(fill for _ in range(width)) for _ in range(top)]
    for row in source:
        padded.append(tuple([fill] * left + list(row) + [fill] * right))
    padded.extend(tuple(fill for _ in range(width)) for _ in range(bottom))
    return tuple(padded)


def conv2d(
    image: Sequence[Sequence[float | int]],
    kernel: Sequence[Sequence[float | int]],
    *,
    stride: int = 1,
    padding: int | PaddingMode = "valid",
    dilation: int = 1,
    bias: float = 0.0,
    flip_kernel: bool = False,
) -> Matrix:
    """Apply one 2D kernel to one 2D image.

    The default operation is cross-correlation, matching most deep-learning
    libraries. Set ``flip_kernel=True`` for mathematical convolution.
    """

    source = _matrix(image, name="image")
    weights = _matrix(kernel, name="kernel")
    step = _positive_int(stride, name="stride")
    rate = _positive_int(dilation, name="dilation")
    offset = float(bias)
    if not math.isfinite(offset):
        raise ValueError("bias must be finite")

    kernel_h, kernel_w = len(weights), len(weights[0])
    if flip_kernel:
        weights = tuple(tuple(reversed(row)) for row in reversed(weights))

    if padding == "valid":
        padded = source
    elif padding == "same":
        top, bottom = same_padding(len(source), kernel_h, stride=step, dilation=rate)
        left, right = same_padding(len(source[0]), kernel_w, stride=step, dilation=rate)
        padded = pad2d(source, (top, bottom, left, right))
    elif isinstance(padding, int) and not isinstance(padding, bool):
        padded = pad2d(source, _non_negative_int(padding, name="padding"))
    else:
        raise ValueError("padding must be 'valid', 'same', or a non-negative integer")

    effective_h = effective_kernel_size(kernel_h, rate)
    effective_w = effective_kernel_size(kernel_w, rate)
    if effective_h > len(padded) or effective_w > len(padded[0]):
        raise ValueError("effective kernel is larger than the padded image")

    out_h = (len(padded) - effective_h) // step + 1
    out_w = (len(padded[0]) - effective_w) // step + 1
    output: list[tuple[float, ...]] = []
    for out_y in range(out_h):
        row: list[float] = []
        origin_y = out_y * step
        for out_x in range(out_w):
            origin_x = out_x * step
            total = offset
            for kernel_y in range(kernel_h):
                image_y = origin_y + kernel_y * rate
                for kernel_x in range(kernel_w):
                    image_x = origin_x + kernel_x * rate
                    total += padded[image_y][image_x] * weights[kernel_y][kernel_x]
            row.append(total)
        output.append(tuple(row))
    return tuple(output)


def pool2d(
    image: Sequence[Sequence[float | int]],
    *,
    kernel_size: int = 2,
    stride: int | None = None,
    mode: Literal["max", "average"] = "max",
) -> Matrix:
    source = _matrix(image, name="image")
    kernel = _positive_int(kernel_size, name="kernel_size")
    step = kernel if stride is None else _positive_int(stride, name="stride")
    if mode not in {"max", "average"}:
        raise ValueError("mode must be 'max' or 'average'")
    if kernel > len(source) or kernel > len(source[0]):
        raise ValueError("pooling kernel is larger than the input")

    out_h = (len(source) - kernel) // step + 1
    out_w = (len(source[0]) - kernel) // step + 1
    output: list[tuple[float, ...]] = []
    for out_y in range(out_h):
        row: list[float] = []
        for out_x in range(out_w):
            values = [
                source[out_y * step + y][out_x * step + x]
                for y in range(kernel)
                for x in range(kernel)
            ]
            row.append(max(values) if mode == "max" else math.fsum(values) / len(values))
        output.append(tuple(row))
    return tuple(output)


def max_pool2d(image: Sequence[Sequence[float | int]], *, kernel_size: int = 2, stride: int | None = None) -> Matrix:
    return pool2d(image, kernel_size=kernel_size, stride=stride, mode="max")


def average_pool2d(image: Sequence[Sequence[float | int]], *, kernel_size: int = 2, stride: int | None = None) -> Matrix:
    return pool2d(image, kernel_size=kernel_size, stride=stride, mode="average")
