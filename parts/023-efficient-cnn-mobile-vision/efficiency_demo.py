"""Compare standard and mobile convolution costs for NeuralForge Part 023."""

from __future__ import annotations

from neuralforge.mobile_vision import (
    depthwise_separable_cost,
    efficiency_ratio,
    inverted_residual_cost,
    standard_conv_cost,
)


def main() -> None:
    input_shape = (32, 112, 112)
    standard = standard_conv_cost(input_shape, 64)
    separable = depthwise_separable_cost(input_shape, 64)
    ratios = efficiency_ratio(standard, separable)

    print("NeuralForge Part 023 — efficient CNNs and mobile vision")
    print(f"input shape: {input_shape}")
    print(
        f"standard conv: params={standard.parameters:,}, MACs={standard.macs:,}, "
        f"output={standard.output_shape}"
    )
    print(
        f"depthwise separable: params={separable.parameters:,}, MACs={separable.macs:,}, "
        f"output={separable.output_shape}"
    )
    print(f"parameter reduction: {ratios['parameter_reduction']:.1%}")
    print(f"MAC reduction: {ratios['mac_reduction']:.1%}")

    block = inverted_residual_cost((24, 56, 56), 24, expansion=6, stride=1)
    print("\nMobileNetV2-style inverted residual")
    print(f"expanded channels: {block.expansion_channels}")
    print(f"parameters: {block.parameters:,}")
    print(f"MACs: {block.macs:,}")
    print(f"residual connection: {block.residual_connection}")


if __name__ == "__main__":
    main()
