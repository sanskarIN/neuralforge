"""Compare initialization-driven signal propagation for NeuralForge Part 018."""

from __future__ import annotations

from neuralforge.initialization import propagate_signal


def print_report(name: str, report) -> None:
    print(name)
    for item in report:
        print(
            f"  layer={item.layer} width={item.width:02d} "
            f"mean={item.mean:+.5f} var={item.variance:.5f} "
            f"zeros={item.zero_fraction:.2%} range=[{item.minimum:+.4f}, {item.maximum:+.4f}]"
        )


def main() -> None:
    batch = [
        [-1.0, -0.5, 0.0, 0.5],
        [0.2, 0.4, 0.6, 0.8],
        [1.0, -1.0, 1.0, -1.0],
        [-0.3, 0.7, -0.9, 0.1],
        [0.9, 0.3, -0.2, -0.8],
        [-0.6, 0.1, 0.8, -0.4],
    ]

    print("NeuralForge Part 018 — initialization and signal propagation")
    print_report(
        "ReLU + He normal",
        propagate_signal(batch, [16, 16, 16, 8], activation="relu", scheme="he_normal", seed=42),
    )
    print()
    print_report(
        "Tanh + Xavier uniform",
        propagate_signal(batch, [16, 16, 16, 8], activation="tanh", scheme="xavier_uniform", seed=42),
    )


if __name__ == "__main__":
    main()
