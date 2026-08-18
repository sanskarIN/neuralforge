"""Explore tensor shape/reshape concepts using plain Python data."""

from __future__ import annotations

from neuralforge.tensor_basics import flatten, infer_shape, numel, reshape


def main() -> None:
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
    ]

    print("NeuralForge Part 002 — tensor shape demo")
    print(f"matrix: {matrix}")
    print(f"shape: {infer_shape(matrix)}")
    print(f"elements: {numel(matrix)}")
    print(f"flattened: {flatten(matrix)}")
    print(f"reshaped to (3, 2): {reshape(matrix, (3, 2))}")


if __name__ == "__main__":
    main()
