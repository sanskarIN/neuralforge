"""Compare classic CNN architecture specifications for NeuralForge Part 022."""

from __future__ import annotations

from neuralforge.cnn_architecture import ArchitectureSummary, lenet5, resnet18, vgg11


def print_summary(summary: ArchitectureSummary) -> None:
    print(f"\n{summary.name}: {summary.input_shape} -> {summary.output_shape}")
    print(f"trainable parameters: {summary.total_parameters:,}")
    print(f"parameterized layers/blocks: {summary.parameterized_layers}")
    for layer in summary.layers:
        if layer.operation in {"conv2d", "residual_basic_block", "flatten", "dense", "global_average_pool2d"}:
            print(
                f"  {layer.name:20} {layer.operation:24} "
                f"{str(layer.input_shape):16} -> {str(layer.output_shape):16} "
                f"params={layer.parameters:,}"
            )


def main() -> None:
    print("NeuralForge Part 022 — CNN architecture design")
    print_summary(lenet5(classes=10))
    print_summary(vgg11(classes=1000))
    print_summary(resnet18(classes=1000))


if __name__ == "__main__":
    main()
