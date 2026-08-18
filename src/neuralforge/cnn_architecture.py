"""CNN architecture geometry and cost analysis for NeuralForge Part 022."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

from .convolution import effective_kernel_size, output_size

Shape3D: TypeAlias = tuple[int, int, int]


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class ConvLayerSpec:
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    groups: int = 1
    bias: bool = True
    name: str = "conv"


@dataclass(frozen=True, slots=True)
class PoolLayerSpec:
    kernel_size: int = 2
    stride: int = 2
    padding: int = 0
    name: str = "pool"


LayerSpec: TypeAlias = ConvLayerSpec | PoolLayerSpec


@dataclass(frozen=True, slots=True)
class LayerReport:
    name: str
    input_shape: Shape3D
    output_shape: Shape3D
    parameters: int
    multiply_accumulates: int
    receptive_field: int
    output_jump: int


@dataclass(frozen=True, slots=True)
class ArchitectureReport:
    input_shape: Shape3D
    output_shape: Shape3D
    layers: tuple[LayerReport, ...]
    total_parameters: int
    total_multiply_accumulates: int
    receptive_field: int


def conv_parameter_count(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    groups: int = 1,
    bias: bool = True,
) -> int:
    incoming = _positive_int(in_channels, name="in_channels")
    outgoing = _positive_int(out_channels, name="out_channels")
    kernel = _positive_int(kernel_size, name="kernel_size")
    group_count = _positive_int(groups, name="groups")
    if incoming % group_count != 0 or outgoing % group_count != 0:
        raise ValueError("groups must divide both input and output channels")
    weights = kernel * kernel * (incoming // group_count) * outgoing
    return weights + (outgoing if bias else 0)


def conv_mac_count(
    output_height: int,
    output_width: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    groups: int = 1,
) -> int:
    height = _positive_int(output_height, name="output_height")
    width = _positive_int(output_width, name="output_width")
    incoming = _positive_int(in_channels, name="in_channels")
    outgoing = _positive_int(out_channels, name="out_channels")
    kernel = _positive_int(kernel_size, name="kernel_size")
    group_count = _positive_int(groups, name="groups")
    if incoming % group_count != 0 or outgoing % group_count != 0:
        raise ValueError("groups must divide both input and output channels")
    return height * width * outgoing * kernel * kernel * (incoming // group_count)


def residual_compatible(main_shape: Shape3D, skip_shape: Shape3D) -> bool:
    return tuple(main_shape) == tuple(skip_shape)


def analyze_cnn_architecture(
    input_shape: Shape3D,
    layers: tuple[LayerSpec, ...] | list[LayerSpec],
) -> ArchitectureReport:
    if len(input_shape) != 3:
        raise ValueError("input_shape must be (height, width, channels)")
    height, width, channels = (
        _positive_int(value, name="input dimension") for value in input_shape
    )
    if not layers:
        raise ValueError("layers must contain at least one layer")

    current_shape: Shape3D = (height, width, channels)
    receptive_field = 1
    jump = 1
    reports: list[LayerReport] = []
    total_parameters = 0
    total_macs = 0

    for index, layer in enumerate(layers):
        input_stage_shape = current_shape
        in_h, in_w, in_channels = current_shape

        if isinstance(layer, ConvLayerSpec):
            out_channels = _positive_int(layer.out_channels, name="out_channels")
            kernel = _positive_int(layer.kernel_size, name="kernel_size")
            stride = _positive_int(layer.stride, name="stride")
            padding = _non_negative_int(layer.padding, name="padding")
            dilation = _positive_int(layer.dilation, name="dilation")
            groups = _positive_int(layer.groups, name="groups")
            out_h = output_size(in_h, kernel, stride=stride, padding=padding, dilation=dilation)
            out_w = output_size(in_w, kernel, stride=stride, padding=padding, dilation=dilation)
            parameters = conv_parameter_count(
                in_channels,
                out_channels,
                kernel,
                groups=groups,
                bias=layer.bias,
            )
            macs = conv_mac_count(
                out_h,
                out_w,
                in_channels,
                out_channels,
                kernel,
                groups=groups,
            )
            receptive_field += (effective_kernel_size(kernel, dilation) - 1) * jump
            jump *= stride
            current_shape = (out_h, out_w, out_channels)
            layer_name = layer.name or f"conv{index + 1}"
        elif isinstance(layer, PoolLayerSpec):
            kernel = _positive_int(layer.kernel_size, name="kernel_size")
            stride = _positive_int(layer.stride, name="stride")
            padding = _non_negative_int(layer.padding, name="padding")
            out_h = output_size(in_h, kernel, stride=stride, padding=padding)
            out_w = output_size(in_w, kernel, stride=stride, padding=padding)
            parameters = 0
            macs = 0
            receptive_field += (kernel - 1) * jump
            jump *= stride
            current_shape = (out_h, out_w, in_channels)
            layer_name = layer.name or f"pool{index + 1}"
        else:
            raise TypeError(f"unsupported layer specification: {type(layer).__name__}")

        total_parameters += parameters
        total_macs += macs
        reports.append(
            LayerReport(
                name=layer_name,
                input_shape=input_stage_shape,
                output_shape=current_shape,
                parameters=parameters,
                multiply_accumulates=macs,
                receptive_field=receptive_field,
                output_jump=jump,
            )
        )

    if not math.isfinite(float(total_macs)):
        raise ValueError("architecture cost overflowed")

    return ArchitectureReport(
        input_shape=(height, width, channels),
        output_shape=current_shape,
        layers=tuple(reports),
        total_parameters=total_parameters,
        total_multiply_accumulates=total_macs,
        receptive_field=receptive_field,
    )


def lenet_style_specs() -> tuple[LayerSpec, ...]:
    """Return a compact LeNet-style convolution/pooling feature extractor."""

    return (
        ConvLayerSpec(6, 5, name="conv1"),
        PoolLayerSpec(2, 2, name="pool1"),
        ConvLayerSpec(16, 5, name="conv2"),
        PoolLayerSpec(2, 2, name="pool2"),
    )
