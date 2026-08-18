"""Demonstrate first-principles convolution and pooling for Part 021."""

from __future__ import annotations

from neuralforge.convolution import conv2d, max_pool2d


def print_matrix(name: str, matrix: tuple[tuple[float, ...], ...]) -> None:
    print(name)
    for row in matrix:
        print("  " + " ".join(f"{value:6.2f}" for value in row))


def main() -> None:
    image = (
        (0, 0, 0, 0, 0),
        (0, 1, 1, 1, 0),
        (0, 1, 1, 1, 0),
        (0, 1, 1, 1, 0),
        (0, 0, 0, 0, 0),
    )
    vertical_edge_kernel = (
        (-1, 0, 1),
        (-1, 0, 1),
        (-1, 0, 1),
    )

    features = conv2d(image, vertical_edge_kernel, padding="same")
    pooled = max_pool2d(features, kernel_size=2, stride=2)

    print("NeuralForge Part 021 — CNN primitives from first principles")
    print_matrix("input", tuple(tuple(float(v) for v in row) for row in image))
    print_matrix("vertical-edge feature map", features)
    print_matrix("max-pooled feature map", pooled)
    print("\nDefault conv2d behavior is cross-correlation, matching common DL libraries.")


if __name__ == "__main__":
    main()
