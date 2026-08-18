"""Vectorized NumPy building blocks for NeuralForge Part 003."""

from __future__ import annotations

import numpy as np


def as_feature_matrix(values: object) -> np.ndarray:
    """Convert input to a finite two-dimensional float64 feature matrix."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2D feature matrix, received shape {matrix.shape}")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("feature matrix must have at least one row and one column")
    if not np.isfinite(matrix).all():
        raise ValueError("feature matrix must contain only finite values")
    return matrix


def standardize_features(values: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize columns and return (standardized, mean, scale).

    Constant columns use a scale of 1.0 so they map to zeros instead of
    producing division-by-zero values.
    """

    matrix = as_feature_matrix(values)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    scale = np.where(std == 0.0, 1.0, std)
    standardized = (matrix - mean) / scale
    return standardized, mean, scale


def dense_layer(inputs: object, weights: object, bias: object) -> np.ndarray:
    """Compute one fully connected layer as ``inputs @ weights + bias``."""

    x = as_feature_matrix(inputs)
    w = np.asarray(weights, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)

    if w.ndim != 2:
        raise ValueError("weights must be a 2D matrix")
    if b.ndim != 1:
        raise ValueError("bias must be a 1D vector")
    if x.shape[1] != w.shape[0]:
        raise ValueError(
            f"input width {x.shape[1]} does not match weight rows {w.shape[0]}"
        )
    if w.shape[1] != b.shape[0]:
        raise ValueError(
            f"weight outputs {w.shape[1]} do not match bias length {b.shape[0]}"
        )
    if not np.isfinite(w).all() or not np.isfinite(b).all():
        raise ValueError("weights and bias must contain only finite values")

    return x @ w + b


def softmax(logits: object) -> np.ndarray:
    """Compute a numerically stable row-wise softmax for a 2D matrix."""

    values = as_feature_matrix(logits)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)
