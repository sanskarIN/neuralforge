"""Shared utilities for the NeuralForge companion repository."""

from .foundations import (
    LogisticNeuron,
    TrainingResult,
    binary_cross_entropy,
    sigmoid,
    train_logistic_neuron,
)
from .reproducibility import SeedReport, set_global_seed

__all__ = [
    "LogisticNeuron",
    "SeedReport",
    "TrainingResult",
    "binary_cross_entropy",
    "set_global_seed",
    "sigmoid",
    "train_logistic_neuron",
]
__version__ = "0.1.0"
