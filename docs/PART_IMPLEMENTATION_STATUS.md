# Part-by-Part Implementation Status

This file tracks the open-source companion implementation for the 120-part NeuralForge curriculum. It tracks repository code only; it is not a statement about whether the corresponding book chapter exists.

## Status legend

- **Implemented** — runnable companion code, tests, and a part README are present.
- **Queued** — identified as the next implementation batch.
- **Planned** — part belongs to the 120-part roadmap but companion code has not been added yet.

## Current implementation

| Part | Topic | Status | Primary companion material |
|---:|---|---|---|
| 001 | Foundations, Installation, Tooling, and Your First Neural Networks | Implemented | `src/neuralforge/foundations.py`, OR-gate lab, unit tests |
| 002 | Python Essentials for Tensor Programs | Implemented | `src/neuralforge/tensor_basics.py`, tensor-shape demo, unit tests |
| 003 | NumPy Mastery for Vectorized Computation | Implemented | isolated NumPy 2.5.1 lab, tests, dedicated CI job |
| 004 | Linear Algebra for Neural Networks | Implemented | `src/neuralforge/linear_algebra.py`, demo, unit tests |
| 005 | Calculus for Neural Networks | Implemented | `src/neuralforge/calculus.py`, gradient-check demo, unit tests |
| 006 | Probability and Statistics for Learning Systems | Implemented | descriptive statistics, covariance/correlation, likelihood, bootstrap intervals, demo, tests |
| 007 | Data Preparation and Leakage-Resistant Evaluation | Implemented | disjoint/stratified splits, train-only standardization, leakage demo, tests |
| 008 | Visualization and Exploratory Data Analysis | Implemented | quantiles, summaries, outliers, histograms, SVG scatter plots, demo, tests |
| 009 | From a Biological Metaphor to an Artificial Neuron | Implemented | inspectable artificial neuron, activations/derivatives, trace demo, tests |
| 010 | Build a Perceptron and Logistic Classifier from Scratch | Implemented | perceptron, logistic regression, comparison demo, separate unit tests |
| 011 | Computational Graphs and Automatic Differentiation | Queued | Next implementation batch |
| 012 | Backpropagation Through Multilayer Perceptrons | Queued | Next implementation batch |
| 013 | Optimization Algorithms for Deep Neural Networks | Queued | Next implementation batch |
| 014 | Regularization and Generalization in Deep Neural Networks | Queued | Next implementation batch |
| 015 | Normalization and Training Stabilization in Deep Neural Networks | Queued | Next implementation batch |
| 016–120 | Remaining NeuralForge curriculum | Planned | Add part-specific code, tests, labs, and READMEs incrementally |

## Repository-wide implementation requirements

A part should not be marked **Implemented** until it has, where applicable:

1. a concise part README with run/test instructions;
2. runnable code or a meaningful executable lab;
3. automated tests for reusable or correctness-sensitive logic;
4. dependency declarations isolated to the part when third-party packages are required;
5. reproducibility notes for experiments that report metrics;
6. licensing/provenance notes for third-party data, models, or assets;
7. CI coverage or a documented reason why automated execution is impractical.

## Next batch

Parts **011–015** will move the repository from single-neuron/classifier foundations into computational graphs, automatic differentiation, multilayer backpropagation, optimization algorithms, regularization, and normalization/training stabilization.

That batch should remain framework-light where possible so learners can inspect the underlying mechanics before later framework-specific implementations.
