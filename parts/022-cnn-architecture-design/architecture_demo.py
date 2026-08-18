"""Print a small CNN architecture report for Part 022."""

from __future__ import annotations

from neuralforge.cnn_architecture import analyze_cnn_architecture, lenet_style_specs


def main() -> None:
    report = analyze_cnn_architecture((32, 32, 1), lenet_style_specs())

    print("NeuralForge Part 022 — CNN architecture design")
    print(f"input: {report.input_shape}")
    print("layers:")
    for layer in report.layers:
        print(
            f"  {layer.name:8s} {layer.input_shape} -> {layer.output_shape} "
            f"params={layer.parameters:,} MACs={layer.multiply_accumulates:,} "
            f"RF={layer.receptive_field} jump={layer.output_jump}"
        )
    print(f"output: {report.output_shape}")
    print(f"total convolution parameters: {report.total_parameters:,}")
    print(f"total convolution MACs: {report.total_multiply_accumulates:,}")
    print(f"final receptive field: {report.receptive_field}x{report.receptive_field}")


if __name__ == "__main__":
    main()
