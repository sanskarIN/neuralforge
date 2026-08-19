"""Framework-light CNN architecture modeling for NeuralForge Part 022.

This module propagates tensor shapes and counts trainable convolution/dense
parameters without allocating tensors. It makes architecture decisions visible
before learners move to framework model classes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .convolution import output_shape_2d

FeatureShape = tuple[int, int, int]  # channels, height, width
Shape = tuple[int, ...]


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _pair(value: int | tuple[int, int], *, name: str) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{name} must be an integer or pair")
        return (
            _positive_int(value[0], name=f"{name}[0]"),
            _positive_int(value[1], name=f"{name}[1]"),
        )
    item = _positive_int(value, name=name)
    return item, item


def _padding_pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("padding must be an integer or pair")
        items = value
    else:
        items = (value, value)
    result: list[int] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError("padding values must be non-negative integers")
        result.append(item)
    return result[0], result[1]


@dataclass(frozen=True, slots=True)
class LayerReport:
    name: str
    operation: str
    input_shape: Shape
    output_shape: Shape
    parameters: int = 0
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ArchitectureSummary:
    name: str
    input_shape: Shape
    output_shape: Shape
    layers: tuple[LayerReport, ...]

    @property
    def total_parameters(self) -> int:
        return sum(layer.parameters for layer in self.layers)

    @property
    def parameterized_layers(self) -> int:
        return sum(layer.parameters > 0 for layer in self.layers)


class ArchitectureBuilder:
    """Propagate shapes and parameter counts through an inspectable CNN spec."""

    def __init__(self, name: str, input_shape: FeatureShape) -> None:
        if not name.strip():
            raise ValueError("architecture name must not be empty")
        if len(input_shape) != 3:
            raise ValueError("CNN input_shape must be (channels, height, width)")
        self.name = name.strip()
        self.input_shape: Shape = tuple(
            _positive_int(item, name="input_shape dimension") for item in input_shape
        )
        self.shape: Shape = self.input_shape
        self.layers: list[LayerReport] = []

    def _feature_shape(self) -> FeatureShape:
        if len(self.shape) != 3:
            raise ValueError("operation requires a spatial (channels, height, width) tensor")
        channels, height, width = self.shape
        return channels, height, width

    def conv(
        self,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        groups: int = 1,
        bias: bool = True,
        name: str | None = None,
    ) -> "ArchitectureBuilder":
        in_channels, height, width = self._feature_shape()
        out_c = _positive_int(out_channels, name="out_channels")
        kh, kw = _pair(kernel_size, name="kernel_size")
        sh, sw = _pair(stride, name="stride")
        ph, pw = _padding_pair(padding)
        group_count = _positive_int(groups, name="groups")
        if in_channels % group_count != 0 or out_c % group_count != 0:
            raise ValueError("input and output channels must be divisible by groups")
        out_h, out_w = output_shape_2d(
            (height, width),
            (kh, kw),
            stride=(sh, sw),
            padding=(ph, pw),
        )
        parameters = out_c * (in_channels // group_count) * kh * kw
        if bias:
            parameters += out_c
        output: FeatureShape = (out_c, out_h, out_w)
        report = LayerReport(
            name=name or f"conv{len(self.layers) + 1}",
            operation="conv2d",
            input_shape=self.shape,
            output_shape=output,
            parameters=parameters,
            notes=f"kernel={kh}x{kw}, stride={sh}x{sw}, padding={ph}x{pw}, groups={group_count}",
        )
        self.layers.append(report)
        self.shape = output
        return self

    def pool(
        self,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        kind: str = "max",
        name: str | None = None,
    ) -> "ArchitectureBuilder":
        channels, height, width = self._feature_shape()
        kh, kw = _pair(kernel_size, name="kernel_size")
        sh, sw = _pair(kernel_size if stride is None else stride, name="stride")
        ph, pw = _padding_pair(padding)
        normalized_kind = kind.strip().lower()
        if normalized_kind not in {"max", "average"}:
            raise ValueError("pool kind must be 'max' or 'average'")
        out_h, out_w = output_shape_2d(
            (height, width), (kh, kw), stride=(sh, sw), padding=(ph, pw)
        )
        output: FeatureShape = (channels, out_h, out_w)
        self.layers.append(
            LayerReport(
                name=name or f"{normalized_kind}_pool{len(self.layers) + 1}",
                operation=f"{normalized_kind}_pool2d",
                input_shape=self.shape,
                output_shape=output,
                notes=f"kernel={kh}x{kw}, stride={sh}x{sw}, padding={ph}x{pw}",
            )
        )
        self.shape = output
        return self

    def residual_basic(
        self,
        out_channels: int,
        *,
        stride: int = 1,
        name: str | None = None,
    ) -> "ArchitectureBuilder":
        """Add a ResNet-style two-3x3-convolution basic block.

        A 1x1 projection shortcut is automatically counted when stride or channel
        count changes so the residual addition has matching shapes.
        """

        in_channels, height, width = self._feature_shape()
        out_c = _positive_int(out_channels, name="out_channels")
        step = _positive_int(stride, name="stride")
        out_h, out_w = output_shape_2d(
            (height, width), (3, 3), stride=step, padding=1
        )
        # Two bias-free 3x3 convolutions. Batch-normalization trainable gamma/beta
        # are included as 2*out_channels per normalization layer.
        first_conv = out_c * in_channels * 3 * 3
        second_conv = out_c * out_c * 3 * 3
        normalization = 4 * out_c
        projection = step != 1 or in_channels != out_c
        projection_parameters = 0
        if projection:
            projection_parameters = out_c * in_channels + 2 * out_c
        parameters = first_conv + second_conv + normalization + projection_parameters
        output: FeatureShape = (out_c, out_h, out_w)
        self.layers.append(
            LayerReport(
                name=name or f"residual{len(self.layers) + 1}",
                operation="residual_basic_block",
                input_shape=self.shape,
                output_shape=output,
                parameters=parameters,
                notes=(
                    f"stride={step}, projection={'yes' if projection else 'no'}, "
                    "two 3x3 convs + trainable BN affine terms"
                ),
            )
        )
        self.shape = output
        return self

    def flatten(self, *, name: str = "flatten") -> "ArchitectureBuilder":
        if len(self.shape) < 2:
            raise ValueError("flatten requires a tensor with at least two dimensions")
        features = 1
        for dimension in self.shape:
            features *= dimension
        output = (features,)
        self.layers.append(
            LayerReport(name, "flatten", self.shape, output, notes="preserves element count")
        )
        self.shape = output
        return self

    def global_average_pool(self, *, name: str = "global_average_pool") -> "ArchitectureBuilder":
        channels, _, _ = self._feature_shape()
        output = (channels,)
        self.layers.append(
            LayerReport(name, "global_average_pool2d", self.shape, output)
        )
        self.shape = output
        return self

    def dense(self, units: int, *, bias: bool = True, name: str | None = None) -> "ArchitectureBuilder":
        if len(self.shape) != 1:
            raise ValueError("dense requires a flat one-dimensional feature vector")
        input_units = self.shape[0]
        output_units = _positive_int(units, name="units")
        parameters = input_units * output_units + (output_units if bias else 0)
        output = (output_units,)
        self.layers.append(
            LayerReport(
                name=name or f"dense{len(self.layers) + 1}",
                operation="dense",
                input_shape=self.shape,
                output_shape=output,
                parameters=parameters,
            )
        )
        self.shape = output
        return self

    def build(self) -> ArchitectureSummary:
        return ArchitectureSummary(self.name, self.input_shape, self.shape, tuple(self.layers))


def lenet5(*, classes: int = 10) -> ArchitectureSummary:
    """Return an inspectable LeNet-5-style architecture for 32x32 grayscale input."""

    return (
        ArchitectureBuilder("LeNet-5", (1, 32, 32))
        .conv(6, 5, name="conv1")
        .pool(2, kind="average", name="pool1")
        .conv(16, 5, name="conv2")
        .pool(2, kind="average", name="pool2")
        .flatten()
        .dense(120, name="fc1")
        .dense(84, name="fc2")
        .dense(_positive_int(classes, name="classes"), name="classifier")
        .build()
    )


def alexnet(*, classes: int = 1000) -> ArchitectureSummary:
    """Return a compact architecture model of the canonical AlexNet shape flow."""

    return (
        ArchitectureBuilder("AlexNet", (3, 224, 224))
        .conv(64, 11, stride=4, padding=2, name="conv1")
        .pool(3, stride=2, name="pool1")
        .conv(192, 5, padding=2, name="conv2")
        .pool(3, stride=2, name="pool2")
        .conv(384, 3, padding=1, name="conv3")
        .conv(256, 3, padding=1, name="conv4")
        .conv(256, 3, padding=1, name="conv5")
        .pool(3, stride=2, name="pool5")
        .flatten()
        .dense(4096, name="fc6")
        .dense(4096, name="fc7")
        .dense(_positive_int(classes, name="classes"), name="classifier")
        .build()
    )


def vgg11(*, classes: int = 1000) -> ArchitectureSummary:
    """Return the VGG-11 convolutional pattern with 224x224 RGB input."""

    builder = ArchitectureBuilder("VGG-11", (3, 224, 224))
    for index, channels in enumerate((64, 128), start=1):
        builder.conv(channels, 3, padding=1, name=f"conv{index}_1").pool(2, name=f"pool{index}")
    for stage, channels in enumerate((256, 512, 512), start=3):
        builder.conv(channels, 3, padding=1, name=f"conv{stage}_1")
        builder.conv(channels, 3, padding=1, name=f"conv{stage}_2")
        builder.pool(2, name=f"pool{stage}")
    return (
        builder.flatten()
        .dense(4096, name="fc1")
        .dense(4096, name="fc2")
        .dense(_positive_int(classes, name="classes"), name="classifier")
        .build()
    )


def resnet18(*, classes: int = 1000) -> ArchitectureSummary:
    """Return a ResNet-18-style shape/parameter specification."""

    builder = (
        ArchitectureBuilder("ResNet-18", (3, 224, 224))
        .conv(64, 7, stride=2, padding=3, bias=False, name="stem_conv")
        .pool(3, stride=2, padding=1, name="stem_pool")
    )
    for stage, (channels, blocks) in enumerate(((64, 2), (128, 2), (256, 2), (512, 2)), start=1):
        for block in range(blocks):
            stride = 2 if stage > 1 and block == 0 else 1
            builder.residual_basic(
                channels,
                stride=stride,
                name=f"stage{stage}_block{block + 1}",
            )
    return (
        builder.global_average_pool()
        .dense(_positive_int(classes, name="classes"), name="classifier")
        .build()
    )
