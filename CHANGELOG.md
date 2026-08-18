# Changelog

All notable changes to the NeuralForge companion repository will be documented here.

The companion repository uses its own semantic-style versioning. Book/publication Version 1.0 is separate from the Python companion package version.

## [Unreleased]

### Added
- GitHub-native collaboration layer with CODEOWNERS, pull-request template, feature/documentation issue templates, and structured issue intake.
- Repository invariant validator that checks required project structure, canonical repository metadata, durable-link policy, and canonical Gumroad visibility.
- Repository Quality CI with dependency-free unit tests on Python 3.10, 3.11, and 3.12.
- Isolated NumPy Part 003 CI job and weekly dependency monitoring.
- Non-destructive Release Readiness workflow that validates, tests, creates SHA-256 manifests, builds a source archive, and uploads temporary artifacts.
- Contributor setup, versioning, reproducibility, dependency, support, and Gumroad/store-link policies.
- `CITATION.cff`, `.editorconfig`, Python package metadata, and shared `neuralforge` package foundation.
- Cross-framework reproducibility helper with optional NumPy, PyTorch, and TensorFlow seeding.
- Part 001 companion: stable sigmoid/BCE, logistic neuron, gradient-descent trainer, OR-gate demo, and tests.
- Part 002 companion: tensor-shape inference, flattening, element counting, reshape validation, demo, and tests.
- Part 003 companion: NumPy vectorization, feature standardization, dense layer, stable softmax, demo, tests, and isolated dependency file.
- Part 004 companion: dependency-free vector/matrix operations, demo, and tests.
- Part 005 companion: numerical derivatives, gradients, gradient checking, demo, and tests.
- Part 006 companion: descriptive statistics, covariance/correlation, normal density, Bernoulli log-likelihood, reproducible bootstrap intervals, demo, and tests.
- Part 007 companion: disjoint and stratified splits, row selection, training-only standardization, leakage demonstration, and tests.
- Part 008 companion: quantiles, descriptive summaries, IQR outlier detection, histograms, self-contained SVG scatter plots, demo, and tests.
- Part 009 companion: inspectable artificial-neuron traces, multiple activations, activation derivatives, demo, and tests.
- Part 010 companion: from-scratch perceptron and logistic-regression modules, convergence/loss histories, comparison demo, and separate tests.
- Part 011 companion: scalar reverse-mode automatic differentiation, computational-graph traversal, nonlinear operations, graph summaries, demo, and derivative tests.
- Part 012 companion: trainable `Neuron`, `Layer`, and `MLP` modules backed by scalar autodiff, differentiable losses, end-to-end training demo, and tests.
- Part 013 companion: SGD, Momentum, RMSProp, Adam, weight decay, global gradient clipping, optimizer-state demo, and exact update-equation tests.
- Part 014 companion: differentiable L1/L2 penalties, inverted dropout, early stopping, generalization-gap/norm monitoring, demo, and tests.
- Part 015 companion: differentiable batch/layer normalization, trainable affine parameters, running evaluation moments, stable softmax, demo, and tests.
- Part 016 companion: MSE/MAE/Huber, stable softplus/log-sum-exp, BCE/categorical cross-entropy from logits, output-design recommendations, demo, and tests.
- Part 017 companion: constant, step, exponential, cosine, warmup/cosine schedules, validation plateau control, demo, and tests.
- Part 018 companion: Xavier/Glorot, He/Kaiming, LeCun and educational initialization schemes, deterministic dense matrices, forward signal propagation statistics, demo, and tests.
- Part 019 companion: global/per-layer gradient statistics, finite guards, gradient-to-parameter ratios, configurable gradient-health classification, demo, and tests.
- Part 020 companion: deterministic experiment configuration fingerprints, optimizer/schedule integration, gradient telemetry, clipping, JSON experiment records, demo, and tests.
- Part-by-part implementation status tracker for the 120-part curriculum.
- Highlighted Gumroad badges and canonical store links across the repository homepage, support/publishing/metadata/release/contribution docs, issue/PR surfaces, and every implemented Part README.

### Changed
- Modernized GitHub workflows to `actions/checkout@v7`, `actions/setup-python@v7`, and `actions/upload-artifact@v7`.
- Expanded the README from repository planning notes into a runnable quick-start, implementation-status, automation, documentation, licensing, and official-store guide.
- Advanced the README, roadmap, package API, validator, and implementation tracker through Parts 001–020.
- Strengthened repository validation to require all implemented Part 001–020 directories, shared modules, milestone tests, and the canonical `https://ramsandesh.gumroad.com` link on reader-facing surfaces.
- Queued Parts 021–025 as the next visual-deep-learning milestone.
- Clarified that the companion package/repository version is independent of the book publication version.
- Kept Gumroad promotional links out of Python source/test logic while exposing the store consistently in reader-facing documentation.

### Verified
- Pull request #5 exercised the original complete Repository Quality workflow successfully before merge.
- Pull request #7 merged Parts 006–010 after 65 dependency-free tests passed on Python 3.10, 3.11, and 3.12, the isolated NumPy Part 003 job passed, and repository invariant validation passed.
- Pull request #8 merged Parts 011–015 after all five Repository Quality jobs passed: validation, NumPy Part 003, and dependency-free tests on Python 3.10, 3.11, and 3.12.
- Parts 016–020 and the repository-wide Gumroad rollout are being delivered on a dedicated feature branch and will merge only after the exact PR head passes the same CI gate.

## [0.1.0] - 2026-08-18

Initial repository/documentation and Python package foundation for the NeuralForge Complete 120-Part Master Edition companion project.

---

**Official store:** **[https://ramsandesh.gumroad.com](https://ramsandesh.gumroad.com)**
