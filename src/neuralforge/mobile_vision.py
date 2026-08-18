"""Efficient/mobile vision cost models for NeuralForge Part 023."""

from __future__ import annotations

import math
from dataclasses import dataclass

Shape3D = tuple[int, int, int]


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result


def _shape(shape: Shape3D) -> Shape3D:
    if len(shape) != 3:
        raise ValueError("shape must be (height, width, channels)")
    return tuple(_positive_int(value, name="shape dimension") for value in shape)  # type: ignore[return-value]


def _same_output_dimension(size: int, stride: int) -> int:
    return math.ceil(size / stride)


@dataclass(frozen=True, slots=True)
class VisionCost:
    parameters: int
    multiply_accumulates: int
    output_shape: Shape3D

    def parameter_ratio(self, baseline: "VisionCost") -> float:
        if baseline.parameters <= 0:
            raise ValueError("baseline must contain parameters")
        return self.parameters / baseline.parameters

    def mac_ratio(self, baseline: "VisionCost") -> float:
        if baseline.multiply_accumulates <= 0:
            raise ValueError("baseline must contain MACs")
        return self.multiply_accumulates / baseline.multiply_accumulates


def standard_conv_cost(
    input_shape: Shape3D,
    out_channels: int,
    *,
    kernel_size: int = 3,
    stride: int = 1,
    bias: bool = False,
) -> VisionCost:
    height, width, in_channels = _shape(input_shape)
    outgoing = _positive_int(out_channels, name="out_channels")
    kernel = _positive_int(kernel_size, name="kernel_size")
    step = _positive_int(stride, name="stride")
    out_h = _same_output_dimension(height, step)
    out_w = _same_output_dimension(width, step)
    weights = kernel * kernel * in_channels * outgoing
    parameters = weights + (outgoing if bias else 0)
    macs = out_h * out_w * weights
    return VisionCost(parameters, macs, (out_h, out_w, outgoing))


def depthwise_separable_cost(
    input_shape: Shape3D,
    out_channels: int,
    *,
    kernel_size: int = 3,
    stride: int = 1,
    bias: bool = False,
) -> VisionCost:
    height, width, in_channels = _shape(input_shape)
    outgoing = _positive_int(out_channels, name="out_channels")
    kernel = _positive_int(kernel_size, name="kernel_size")
    step = _positive_int(stride, name="stride")
    out_h = _same_output_dimension(height, step)
    out_w = _same_output_dimension(width, step)

    depthwise_weights = kernel * kernel * in_channels
    pointwise_weights = in_channels * outgoing
    parameters = depthwise_weights + pointwise_weights
    if bias:
        parameters += in_channels + outgoing
    macs = out_h * out_w * (depthwise_weights + pointwise_weights)
    return VisionCost(parameters, macs, (out_h, out_w, outgoing))


def inverted_residual_cost(
    input_shape: Shape3D,
    out_channels: int,
    *,
    expand_ratio: float = 6.0,
    kernel_size: int = 3,
    stride: int = 1,
    bias: bool = False,
) -> VisionCost:
    height, width, in_channels = _shape(input_shape)
    outgoing = _positive_int(out_channels, name="out_channels")
    kernel = _positive_int(kernel_size, name="kernel_size")
    step = _positive_int(stride, name="stride")
    expansion = _positive_float(expand_ratio, name="expand_ratio")
    expanded = max(1, int(round(in_channels * expansion)))
    out_h = _same_output_dimension(height, step)
    out_w = _same_output_dimension(width, step)

    expansion_weights = 0 if expanded == in_channels else in_channels * expanded
    depthwise_weights = kernel * kernel * expanded
    projection_weights = expanded * outgoing
    parameters = expansion_weights + depthwise_weights + projection_weights
    if bias:
        parameters += (0 if expanded == in_channels else expanded) + expanded + outgoing

    expansion_macs = 0 if expanded == in_channels else height * width * expansion_weights
    depthwise_macs = out_h * out_w * depthwise_weights
    projection_macs = out_h * out_w * projection_weights
    return VisionCost(
        parameters,
        expansion_macs + depthwise_macs + projection_macs,
        (out_h, out_w, outgoing),
    )


def model_size_bytes(parameters: int, *, bits_per_parameter: int = 32) -> int:
    count = _positive_int(parameters, name="parameters")
    bits = _positive_int(bits_per_parameter, name="bits_per_parameter")
    return math.ceil(count * bits / 8)


def model_size_mebibytes(parameters: int, *, bits_per_parameter: int = 32) -> float:
    return model_size_bytes(parameters, bits_per_parameter=bits_per_parameter) / (1024 * 1024)


def width_scaled_channels(channels: int, multiplier: float, *, divisor: int = 8) -> int:
    base = _positive_int(channels, name="channels")
    factor = _positive_float(multiplier, name="multiplier")
    block = _positive_int(divisor, name="divisor")
    scaled = base * factor
    rounded = max(block, int(scaled + block / 2) // block * block)
    if rounded < 0.9 * scaled:
        rounded += block
    return rounded
