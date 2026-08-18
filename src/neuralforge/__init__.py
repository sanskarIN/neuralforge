"""Shared utilities for the NeuralForge companion repository."""

from .autograd import Value, graph_summary
from .calculus import GradientCheckResult, check_gradient, numerical_derivative, numerical_gradient
from .data_preparation import SplitIndices, Standardizer, random_split_indices, select_rows, stratified_split_indices
from .eda import Histogram, NumericSummary, describe, histogram, iqr_outlier_mask, quantile, scatter_svg, write_scatter_svg
from .foundations import LogisticNeuron, TrainingResult, binary_cross_entropy, sigmoid, train_logistic_neuron
from .linear_algebra import cosine_similarity, dot, l2_norm, matmul, outer, transpose
from .logistic_regression import LogisticRegression, LogisticRegressionResult, train_logistic_regression
from .neuron import ArtificialNeuron, NeuronTrace, activation_derivative, identity, leaky_relu, relu
from .nn import Layer, MLP, Module, Neuron, binary_cross_entropy_loss, mean_squared_error, sgd_step
from .normalization import (
    BatchNormalizationResult,
    LayerNormalizationResult,
    RunningMoments,
    batch_normalize,
    layer_normalize,
    normalization_parameters,
    stable_softmax,
)
from .optim import Adam, Momentum, Optimizer, RMSProp, SGD, clip_grad_norm
from .perceptron import Perceptron, PerceptronResult, train_perceptron
from .regularization import (
    DropoutResult,
    EarlyStopping,
    generalization_gap,
    inverted_dropout,
    l1_penalty,
    l2_penalty,
    parameter_l2_norm,
    regularized_loss,
)
from .reproducibility import SeedReport, set_global_seed
from .statistics import BootstrapEstimate, bernoulli_log_likelihood, bootstrap_mean_interval, correlation, covariance, mean, normal_pdf, standard_deviation, variance
from .tensor_basics import flatten, infer_shape, numel, reshape

__all__ = [
    "Adam",
    "ArtificialNeuron",
    "BatchNormalizationResult",
    "BootstrapEstimate",
    "DropoutResult",
    "EarlyStopping",
    "GradientCheckResult",
    "Histogram",
    "Layer",
    "LayerNormalizationResult",
    "LogisticNeuron",
    "LogisticRegression",
    "LogisticRegressionResult",
    "MLP",
    "Module",
    "Momentum",
    "Neuron",
    "NeuronTrace",
    "NumericSummary",
    "Optimizer",
    "Perceptron",
    "PerceptronResult",
    "RMSProp",
    "RunningMoments",
    "SGD",
    "SeedReport",
    "SplitIndices",
    "Standardizer",
    "TrainingResult",
    "Value",
    "activation_derivative",
    "batch_normalize",
    "bernoulli_log_likelihood",
    "binary_cross_entropy",
    "binary_cross_entropy_loss",
    "bootstrap_mean_interval",
    "check_gradient",
    "clip_grad_norm",
    "correlation",
    "cosine_similarity",
    "covariance",
    "describe",
    "dot",
    "flatten",
    "generalization_gap",
    "graph_summary",
    "histogram",
    "identity",
    "infer_shape",
    "inverted_dropout",
    "iqr_outlier_mask",
    "l1_penalty",
    "l2_norm",
    "l2_penalty",
    "layer_normalize",
    "leaky_relu",
    "matmul",
    "mean",
    "mean_squared_error",
    "normal_pdf",
    "normalization_parameters",
    "numel",
    "numerical_derivative",
    "numerical_gradient",
    "outer",
    "parameter_l2_norm",
    "quantile",
    "random_split_indices",
    "regularized_loss",
    "relu",
    "reshape",
    "scatter_svg",
    "select_rows",
    "set_global_seed",
    "sgd_step",
    "sigmoid",
    "stable_softmax",
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
