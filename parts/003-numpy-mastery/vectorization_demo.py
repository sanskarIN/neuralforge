"""Demonstrate vectorized feature transforms, dense layers, and softmax."""

from __future__ import annotations

from numpy_vectorization import dense_layer, softmax, standardize_features


def main() -> None:
    features = [
        [18.0, 120.0, 3.0],
        [24.0, 150.0, 5.0],
        [30.0, 180.0, 7.0],
        [36.0, 210.0, 9.0],
    ]

    standardized, mean, scale = standardize_features(features)

    weights = [
        [0.7, -0.3],
        [-0.2, 0.5],
        [0.4, 0.1],
    ]
    bias = [0.1, -0.2]

    logits = dense_layer(standardized, weights, bias)
    probabilities = softmax(logits)

    print("NeuralForge Part 003 — NumPy vectorization")
    print(f"mean: {mean}")
    print(f"scale: {scale}")
    print("standardized features:")
    print(standardized)
    print("logits:")
    print(logits)
    print("probabilities:")
    print(probabilities)


if __name__ == "__main__":
    main()
