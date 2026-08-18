"""Shared utilities for the NeuralForge companion repository."""

from .calculus import (
    GradientCheckResult,
    check_gradient,
    numerical_derivative,
    numerical_gradient,
)
from .data_preparation import (
    SplitIndices,
    Standardizer,
    random_split_indices,
    select_rows,
    stratified_split_indices,
)
from .eda import (
    Histogram,
    NumericSummary,
    describe,
    histogram,
    iqr_outlier_mask,
    quantile,
    scatter_svg,
    write_scatter_svg,
)
from .foundations import (
    LogisticNeuron,
    TrainingResult,
    binary_cross_entropy,
    sigmoid,
    train_logistic_neuron,
)
from .linear_algebra import cosine_similarity, dot, l2_norm, matmul, outer, transpose
from .logistic_regression import (
    LogisticRegression,
    LogisticRegressionResult,
    train_logistic_regression,
)
from .neuron import (
    ArtificialNeuron,
    NeuronTrace,
    activation_derivative,
    identity,
    leaky_relu,
    relu,
)
from .perceptron import Perceptron, PerceptronResult, train_perceptron
from .reproducibility import SeedReport, set_global_seed
from .statistics import (
    BootstrapEstimate,
    bernoulli_log_likelihood,
    bootstrap_mean_interval,
    correlation,
    covariance,
    mean,
    normal_pdf,
    standard_deviation,
    variance,
)
from .tensor_basics import flatten, infer_shape, numel, reshape

__all__ = [
    "ArtificialNeuron",
    "BootstrapEstimate",
    "GradientCheckResult",
    "Histogram",
    "LogisticNeuron",
    "LogisticRegression",
    "LogisticRegressionResult",
    "NeuronTrace",
    "NumericSummary",
    "Perceptron",
    "PerceptronResult",
    "SeedReport",
    "SplitIndices",
    "Standardizer",
    "TrainingResult",
    "activation_derivative",
    "bernoulli_log_likelihood",
    "binary_cross_entropy",
    "bootstrap_mean_interval",
    "check_gradient",
    "correlation",
    "cosine_similarity",
    "covariance",
    "describe",
    "dot",
    "flatten",
    "histogram",
    "identity",
    "infer_shape",
    "iqr_outlier_mask",
    "l2_norm",
    "leaky_relu",
    "matmul",
    "mean",
    "normal_pdf",
    "numel",
    "numerical_derivative",
    "numerical_gradient",
    "outer",
    "quantile",
    "random_split_indices",
    "relu",
    "reshape",
    "scatter_svg",
    "select_rows",
    "set_global_seed",
    "sigmoid",
    "standard_deviation",
    "stratified_split_indices",
    "train_logistic_neuron",
    "train_logistic_regression",
    "train_perceptron",
    "transpose",
    "variance",
    "write_scatter_svg",
]
__version__ = "0.1.0"
