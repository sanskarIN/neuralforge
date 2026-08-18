# NeuralForge

[![Repository Quality](https://github.com/sanskarIN/neuralforge/actions/workflows/repository-quality.yml/badge.svg)](https://github.com/sanskarIN/neuralforge/actions/workflows/repository-quality.yml)
[![Get NeuralForge on Gumroad](https://img.shields.io/badge/Gumroad-Ram%20Sandesh-FF90E8?style=for-the-badge&logo=gumroad&logoColor=000000)](https://ramsandesh.gumroad.com)

**Deep Learning from Zero to Mastery — Companion Repository**

NeuralForge is the open-source companion repository for the **Complete 120-Part Master Edition** of *NeuralForge - Deep Learning from Zero to Mastery* by **Ram Sandesh**.

> **Official store:** [Ram Sandesh on Gumroad — ebooks, publication releases, and digital products](https://ramsandesh.gumroad.com)

The repository is being built as a practical learning system: runnable code, tests, labs, reference implementations, reproducibility utilities, release engineering, and concise part-specific documentation.

## Publication

- Author: **Ram Sandesh**
- Edition: **Complete 120-Part Master Edition**
- Publication build: **August 2026, Version 1.0**
- Canonical repository: `https://github.com/sanskarIN/neuralforge`
- **Official Gumroad store:** **https://ramsandesh.gumroad.com**

## Current implementation status

Parts **001–025** now have runnable companion implementations and automated tests:

| Part | Topic | Companion focus |
|---:|---|---|
| 001 | Foundations | Logistic neuron, stable sigmoid/BCE, gradient-descent OR-gate lab |
| 002 | Python Essentials | Tensor shapes, flattening, element counts, reshape validation |
| 003 | NumPy Mastery | Vectorization, standardization, dense layers, stable softmax |
| 004 | Linear Algebra | Dot products, norms, cosine similarity, transpose, matmul, outer products |
| 005 | Calculus | Numerical derivatives, gradients, and analytical gradient checking |
| 006 | Probability & Statistics | Descriptive statistics, covariance/correlation, likelihood, bootstrap intervals |
| 007 | Data Preparation | Disjoint/stratified splits and training-only standardization |
| 008 | Visualization & EDA | Quantiles, summaries, outliers, histograms, self-contained SVG scatter plots |
| 009 | Artificial Neuron | Weighted contribution traces, activations, and activation derivatives |
| 010 | Perceptron & Logistic Regression | From-scratch binary models, convergence histories, comparison demo |
| 011 | Computational Graphs & Autodiff | Scalar reverse-mode autodiff, graph traversal, chain-rule gradient tests |
| 012 | Multilayer Backpropagation | `Neuron`/`Layer`/`MLP`, differentiable losses, end-to-end training |
| 013 | Optimization | SGD, Momentum, RMSProp, Adam, weight decay, global gradient clipping |
| 014 | Regularization | L1/L2, inverted dropout, early stopping, generalization monitoring |
| 015 | Normalization & Stability | Batch/layer normalization, running moments, trainable affine terms, stable softmax |
| 016 | Loss & Output Design | logits-first BCE/categorical CE, MSE/MAE/Huber, output-objective recommendations |
| 017 | Learning-Rate Control | constant/step/exponential/cosine/warmup schedules and plateau reduction |
| 018 | Initialization & Signal Propagation | Xavier/He/LeCun schemes and seeded layerwise activation statistics |
| 019 | Gradient Flow | finite checks, global/per-layer statistics, gradient/parameter ratio, health diagnostics |
| 020 | Training-Loop Engineering | deterministic experiment configs, optimizer/schedule integration, gradient telemetry, JSON run records |
| 021 | CNNs from First Principles | cross-correlation/convolution, padding, stride, dilation, max/average pooling |
| 022 | CNN Architecture Design | shapes, parameter/MAC estimates, receptive fields, grouped convolution, residual compatibility |
| 023 | Efficient & Mobile Vision | depthwise-separable/inverted-residual cost models, precision/model-size estimates, width scaling |
| 024 | Object Detection | box geometry, IoU, class-aware NMS, matching, precision/recall |
| 025 | Image Segmentation | confusion matrices, pixel accuracy, IoU/Dice, ignore labels, panoptic quality |

See [`docs/PART_IMPLEMENTATION_STATUS.md`](docs/PART_IMPLEMENTATION_STATUS.md) for the 120-part implementation tracker.

The reusable vision APIs for Parts 021–025 are grouped under **`neuralforge.vision`** while the lower-level modules remain importable individually.

## Quick start

Clone the repository and run the dependency-free test suite:

```bash
git clone https://github.com/sanskarIN/neuralforge.git
cd neuralforge
python -m unittest discover -s tests -p "test_*.py" -v
```

Run the first-principles CNN demo:

```bash
PYTHONPATH=src python parts/021-convolutional-neural-networks/cnn_primitives_demo.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python parts/021-convolutional-neural-networks/cnn_primitives_demo.py
```

The core companion package supports Python **3.10+**. Part 003 intentionally isolates its current NumPy dependency and uses Python **3.12+**.

## Quality and automation

The `Repository Quality` workflow checks:

- repository invariants, canonical GitHub destination, Gumroad visibility, and durable-link policy;
- whitespace and Python compilation;
- dependency-free unit tests on Python 3.10, 3.11, and 3.12;
- the isolated Part 003 NumPy test suite on Python 3.12.

The repository validator requires every implemented Part README to retain the canonical Gumroad store link while keeping promotional content out of Python source/test logic. Dependabot monitors GitHub Actions and the Part 003 Python dependency. A separate non-destructive Release Readiness workflow validates the repository, runs tests, generates SHA-256 checksums, and builds a source archive without automatically publishing a release.

## Repository layout

```text
.github/       GitHub workflows, templates, CODEOWNERS, Dependabot
src/           Shared neuralforge Python package
parts/         Part-by-part companion implementations
labs/          Cross-part practical labs and future capstones
examples/      Reusable focused examples
tests/         Dependency-free package tests
tools/         Repository validation utilities
docs/          Roadmap, policies, metadata, publishing and QA docs
```

## Documentation

- [`docs/CONTRIBUTOR_SETUP.md`](docs/CONTRIBUTOR_SETUP.md) — contributor environment and workflow
- [`docs/PART_IMPLEMENTATION_STATUS.md`](docs/PART_IMPLEMENTATION_STATUS.md) — Parts 001–120 implementation tracker
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — experiment reproducibility policy
- [`docs/DEPENDENCY_POLICY.md`](docs/DEPENDENCY_POLICY.md) — dependency and licensing rules
- [`docs/VERSIONING.md`](docs/VERSIONING.md) — repository versioning policy
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — development phases
- [`docs/ERRATA_POLICY.md`](docs/ERRATA_POLICY.md) — correction workflow
- [`docs/GUMROAD.md`](docs/GUMROAD.md) — official Gumroad/store-link policy and reusable badge
- [`SUPPORT.md`](SUPPORT.md) — support routing
- [`CHANGELOG.md`](CHANGELOG.md) — notable repository changes

## Official Gumroad store

[![Visit Ram Sandesh on Gumroad](https://img.shields.io/badge/Visit%20Store-ramsandesh.gumroad.com-FF90E8?style=for-the-badge&logo=gumroad&logoColor=000000)](https://ramsandesh.gumroad.com)

The official store destination for NeuralForge publication releases and other Ram Sandesh digital products is **https://ramsandesh.gumroad.com**. Repository documentation uses this canonical store-home URL so readers have a durable storefront destination.

## Licensing

Original companion **source code** in this repository uses the **MIT License** unless a file or third-party dependency states otherwise.

The **book manuscript, PDF/EPUB/DOCX publication files, book layout, exercises, explanations, cover, and publication assets are not automatically licensed under MIT**. See [`BOOK_LICENSE.md`](BOOK_LICENSE.md) for the book-content rights policy.

Third-party libraries, datasets, model weights, and assets retain their own licenses and attribution requirements.

## Durable links

Permanent NeuralForge publication artifacts intentionally do **not** embed an X/Twitter profile URL. Social handles can change after readers purchase a copy, so durable project metadata uses the canonical repository and the official Gumroad storefront instead.

## Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`docs/CONTRIBUTOR_SETUP.md`](docs/CONTRIBUTOR_SETUP.md) before contributing. Pull requests run the repository-quality checks automatically.

## Citation

Machine-readable citation metadata is available in [`CITATION.cff`](CITATION.cff).

---

**Official store:** **[https://ramsandesh.gumroad.com](https://ramsandesh.gumroad.com)**
