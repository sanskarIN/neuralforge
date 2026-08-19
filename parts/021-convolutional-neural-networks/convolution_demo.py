"""Inspect convolution and pooling on a tiny image for NeuralForge Part 021."""

from __future__ import annotations

from neuralforge.convolution import average_pool2d, cross_correlate2d, max_pool2d


def print_matrix(name: str, matrix: tuple[tuple[float, ...], ...]) -> None:
    print(name)
    for row in matrix:
        print("  ", " ".join(f"{value:6.1f}" for value in row))


def main() -> None:
    image = (
        (0, 0, 0, 10, 10, 10),
        (0, 0, 0, 10, 10, 10),
        (0, 0, 0, 10, 10, 10),
        (0, 0, 0, 10, 10, 10),
    )
    vertical_edge_kernel = (
        (-1, 0, 1),
        (-1, 0, 1),
        (-1, 0, 1),
    )

    response = cross_correlate2d(image, vertical_edge_kernel, padding=1)
    maxima = max_pool2d(response, kernel_size=2, stride=2)
    averages = average_pool2d(response, kernel_size=2, stride=2)

    print("NeuralForge Part 021 — convolution from first principles")
    print_matrix("input", tuple(tuple(float(v) for v in row) for row in image))
    print_matrix("vertical-edge response", response)
    print_matrix("max pooled", maxima)
    print_matrix("average pooled", averages)


if __name__ == "__main__":
    main()
