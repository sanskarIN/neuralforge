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
| 011 | Computational Graphs and Automatic Differentiation | Implemented | scalar reverse-mode autodiff, graph summary, gradient tests, demo |
| 012 | Backpropagation Through Multilayer Perceptrons | Implemented | `Neuron`/`Layer`/`MLP`, losses, multilayer training demo, tests |
| 013 | Optimization Algorithms for Deep Neural Networks | Implemented | SGD, Momentum, RMSProp, Adam, weight decay, gradient clipping, tests |
| 014 | Regularization and Generalization in Deep Neural Networks | Implemented | L1/L2 penalties, inverted dropout, early stopping, generalization helpers, tests |
| 015 | Normalization and Training Stabilization in Deep Neural Networks | Implemented | batch/layer normalization, running moments, stable softmax, tests |
| 016 | Loss Functions and Output-Layer Design | Implemented | robust regression losses, BCE/CE from logits, output/loss pairing guide, tests |
| 017 | Learning-Rate Schedules and Optimization Control | Implemented | constant/step/exponential/cosine/warmup schedules, plateau controller, tests |
| 018 | Initialization and Deep Signal Propagation | Implemented | Xavier/He/LeCun initialization, seeded matrices, signal-variance profiling, tests |
| 019 | Gradient Flow and Deep-Network Stability | Implemented | layerwise gradient statistics/health, update ratios, MLP diagnostics, tests |
| 020 | Training-Loop Engineering & Reproducible Experiment Runners | Implemented | deterministic experiment config/fingerprints, schedules, clipping, validation/early-stop history, tests |
| 021 | Convolutional Neural Networks from First Principles | Queued | Next implementation batch |
| 022 | CNN Architecture Design: From LeNet to ResNet | Queued | Next implementation batch |
| 023 | Efficient CNNs & Mobile Vision | Queued | Next implementation batch |
| 024 | Object Detection from First Principles to Real-Time Systems | Queued | Next implementation batch |
| 025 | Image Segmentation: From Pixels to Panoptic Understanding | Queued | Next implementation batch |
| 026–120 | Remaining NeuralForge curriculum | Planned | Add part-specific code, tests, labs, and READMEs incrementally |

## Repository-wide implementation requirements

A part should not be marked **Implemented** until it has, where applicable:

1. a concise part README with run/test instructions;
2. runnable code or a meaningful executable lab;
3. automated tests for reusable or correctness-sensitive logic;
4. dependency declarations isolated to the part when third-party packages are required;
5. reproducibility notes for experiments that report metrics;
6. licensing/provenance notes for third-party data, models, or assets;
7. CI coverage or a documented reason why automated execution is impractical;
8. the canonical Gumroad storefront footer on GitHub-facing Part documentation.

## Next batch

Parts **021–025** begin NeuralForge's computer-vision sequence: convolution from first principles, CNN architecture patterns, efficient/mobile CNN design, object detection, and image segmentation.

## Official storefront

Available Ram Sandesh digital publications and storefront releases are linked through the canonical Gumroad destination:

**https://ramsandesh.gumroad.com**
