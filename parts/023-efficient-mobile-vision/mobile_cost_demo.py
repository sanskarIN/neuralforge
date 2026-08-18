"""Compare standard and mobile-friendly convolution costs for Part 023."""

from __future__ import annotations

from neuralforge.mobile_vision import (
    depthwise_separable_cost,
    inverted_residual_cost,
    model_size_mebibytes,
    standard_conv_cost,
)


def main() -> None:
    input_shape = (32, 32, 32)
    standard = standard_conv_cost(input_shape, 64, kernel_size=3)
    depthwise = depthwise_separable_cost(input_shape, 64, kernel_size=3)
    block = inverted_residual_cost((16, 16, 16), 24, expand_ratio=6, stride=2)

    print("NeuralForge Part 023 — Efficient CNNs & Mobile Vision")
    print(f"standard conv: params={standard.parameters:,} MACs={standard.multiply_accumulates:,}")
    print(f"depthwise separable: params={depthwise.parameters:,} MACs={depthwise.multiply_accumulates:,}")
    print(f"parameter ratio: {depthwise.parameter_ratio(standard):.3f}")
    print(f"MAC ratio: {depthwise.mac_ratio(standard):.3f}")
    print(
        "inverted residual: "
        f"output={block.output_shape} params={block.parameters:,} MACs={block.multiply_accumulates:,}"
    )
    print(
        "1M parameters model-size estimate: "
        f"FP32={model_size_mebibytes(1_000_000, bits_per_parameter=32):.2f} MiB, "
        f"INT8={model_size_mebibytes(1_000_000, bits_per_parameter=8):.2f} MiB"
    )


if __name__ == "__main__":
    main()
