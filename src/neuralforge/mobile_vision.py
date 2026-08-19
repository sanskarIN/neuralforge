"""CNN efficiency and mobile-vision cost estimators for NeuralForge Part 023."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .convolution import output_shape_2d

FeatureShape = tuple[int, int, int]


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result


def _validate_shape(shape: FeatureShape) -> FeatureShape:
    if len(shape) != 3:
        raise ValueError("shape must be (channels, height, width)")
    return tuple(_positive_int(item, name="shape dimension") for item in shape)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class OperationCost:
    name: str
    input_shape: FeatureShape
    output_shape: FeatureShape
    parameters: int
    macs: int
    peak_output_elements: int
    notes: str = ""

    @property
    def parameter_bytes_fp32(self) -> int:
        return self.parameters * 4

    @property
    def output_bytes_fp32(self) -> int:
        return self.peak_output_elements * 4


@dataclass(frozen=True, slots=True)
class InvertedResidualCost:
    input_shape: FeatureShape
    expanded_shape: FeatureShape
    output_shape: FeatureShape
    parameters: int
    macs: int
    residual_connection: bool
    expansion_channels: int


def make_divisible(channels: float, *, divisor: int = 8, minimum: int | None = None) -> int:
    """Round a channel target to deployment-friendly divisibility.

    The result follows the MobileNet-style rule that avoids rounding down by
    more than 10 percent of the unrounded target.
    """

    target = _positive_float(channels, name="channels")
    div = _positive_int(divisor, name="divisor")
    floor = div if minimum is None else _positive_int(minimum, name="minimum")
    rounded = max(floor, int(target + div / 2) // div * div)
    if rounded < 0.9 * target:
        rounded += div
    return rounded


def scale_channels(channels: int, width_multiplier: float, *, divisor: int = 8) -> int:
    base = _positive_int(channels, name="channels")
    multiplier = _positive_float(width_multiplier, name="width_multiplier")
    return make_divisible(base * multiplier, divisor=divisor)


def scale_resolution(
    height: int,
    width: int,
    resolution_multiplier: float,
) -> tuple[int, int]:
    h = _positive_int(height, name="height")
    w = _positive_int(width, name="width")
    multiplier = _positive_float(resolution_multiplier, name="resolution_multiplier")
    return max(1, int(round(h * multiplier))), max(1, int(round(w * multiplier)))


def standard_conv_cost(
    input_shape: FeatureShape,
    out_channels: int,
    *,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    bias: bool = False,
) -> OperationCost:
    in_c, height, width = _validate_shape(input_shape)
    out_c = _positive_int(out_channels, name="out_channels")
    kernel = _positive_int(kernel_size, name="kernel_size")
    step = _positive_int(stride, name="stride")
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
        raise ValueError("padding must be a non-negative integer")
    out_h, out_w = output_shape_2d(
        (height, width), (kernel, kernel), stride=step, padding=padding
    )
    kernel_macs = in_c * kernel * kernel
    parameters = out_c * kernel_macs + (out_c if bias else 0)
    macs = out_h * out_w * out_c * kernel_macs
    output: FeatureShape = (out_c, out_h, out_w)
    return OperationCost(
        name="standard_conv2d",
        input_shape=(in_c, height, width),
        output_shape=output,
        parameters=parameters,
        macs=macs,
        peak_output_elements=out_c * out_h * out_w,
        notes=f"kernel={kernel}, stride={step}, padding={padding}",
    )


def depthwise_separable_cost(
    input_shape: FeatureShape,
    out_channels: int,
    *,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    bias: bool = False,
) -> OperationCost:
    """Estimate depthwise spatial convolution followed by 1x1 pointwise mixing."""

    in_c, height, width = _validate_shape(input_shape)
    out_c = _positive_int(out_channels, name="out_channels")
    kernel = _positive_int(kernel_size, name="kernel_size")
    step = _positive_int(stride, name="stride")
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
        raise ValueError("padding must be a non-negative integer")
    out_h, out_w = output_shape_2d(
        (height, width), (kernel, kernel), stride=step, padding=padding
    )
    depthwise_parameters = in_c * kernel * kernel
    pointwise_parameters = in_c * out_c
    if bias:
        depthwise_parameters += in_c
        pointwise_parameters += out_c
    depthwise_macs = out_h * out_w * in_c * kernel * kernel
    pointwise_macs = out_h * out_w * in_c * out_c
    output: FeatureShape = (out_c, out_h, out_w)
    return OperationCost(
        name="depthwise_separable_conv2d",
        input_shape=(in_c, height, width),
        output_shape=output,
        parameters=depthwise_parameters + pointwise_parameters,
        macs=depthwise_macs + pointwise_macs,
        peak_output_elements=max(in_c * out_h * out_w, out_c * out_h * out_w),
        notes="depthwise spatial convolution + 1x1 pointwise convolution",
    )


def inverted_residual_cost(
    input_shape: FeatureShape,
    out_channels: int,
    *,
    expansion: float = 6.0,
    stride: int = 1,
    kernel_size: int = 3,
    padding: int = 1,
) -> InvertedResidualCost:
    """Estimate a MobileNetV2-style expand-depthwise-project block."""

    in_c, height, width = _validate_shape(input_shape)
    out_c = _positive_int(out_channels, name="out_channels")
    factor = _positive_float(expansion, name="expansion")
    step = _positive_int(stride, name="stride")
    kernel = _positive_int(kernel_size, name="kernel_size")
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
        raise ValueError("padding must be a non-negative integer")

    expanded_c = max(1, int(round(in_c * factor)))
    out_h, out_w = output_shape_2d(
        (height, width), (kernel, kernel), stride=step, padding=padding
    )
    # Mobile blocks commonly pair these convolutions with normalization, so the
    # estimator intentionally counts bias-free convolution weights only.
    expand_parameters = in_c * expanded_c
    depthwise_parameters = expanded_c * kernel * kernel
    project_parameters = expanded_c * out_c
    expand_macs = height * width * in_c * expanded_c
    depthwise_macs = out_h * out_w * expanded_c * kernel * kernel
    project_macs = out_h * out_w * expanded_c * out_c
    output: FeatureShape = (out_c, out_h, out_w)
    return InvertedResidualCost(
        input_shape=(in_c, height, width),
        expanded_shape=(expanded_c, height, width),
        output_shape=output,
        parameters=expand_parameters + depthwise_parameters + project_parameters,
        macs=expand_macs + depthwise_macs + project_macs,
        residual_connection=step == 1 and in_c == out_c,
        expansion_channels=expanded_c,
    )


def efficiency_ratio(reference: OperationCost, candidate: OperationCost) -> dict[str, float]:
    """Return candidate/reference ratios for parameters and MACs."""

    if reference.parameters <= 0 or reference.macs <= 0:
        raise ValueError("reference cost must have positive parameters and MACs")
    return {
        "parameter_ratio": candidate.parameters / reference.parameters,
        "mac_ratio": candidate.macs / reference.macs,
        "parameter_reduction": 1.0 - candidate.parameters / reference.parameters,
        "mac_reduction": 1.0 - candidate.macs / reference.macs,
    }
