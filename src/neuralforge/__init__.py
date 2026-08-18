"""Shared utilities for the NeuralForge companion repository."""

from .calculus import GradientCheckResult, check_gradient, numerical_derivative, numerical_gradient
from .foundations import (
    LogisticNeuron,
    TrainingResult,
    binary_cross_entropy,
    sigmoid,
    train_logistic_neuron,
)
from .linear_algebra import cosine_similarity, dot, l2_norm, matmul, outer, transpose
from .reproducibility import SeedReport, set_global_seed
from .tensor_basics import flatten, infer_shape, numel, reshape

__all__ = [
    "GradientCheckResult",
    "LogisticNeuron",
    "SeedReport",
    "TrainingResult",
    "binary_cross_entropy",
    "check_gradient",
    "cosine_similarity",
    "dot",
    "flatten",
    "infer_shape",
    "l2_norm",
    "matmul",
    "numel",
    "numerical_derivative",
    "numerical_gradient",
    "outer",
    "reshape",
    "set_global_seed",
    "sigmoid",
    "train_logistic_neuron",
    "transpose",
]
__version__ = "0.1.0"
