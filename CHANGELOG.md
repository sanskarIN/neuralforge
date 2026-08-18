# Changelog

All notable changes to the NeuralForge companion repository will be documented here.

The companion repository uses its own semantic-style versioning. Book/publication Version 1.0 is separate from the Python companion package version.

## [Unreleased]

### Added
- GitHub-native collaboration layer with CODEOWNERS, pull-request template, feature/documentation issue templates, and structured issue intake.
- Repository invariant validator that checks required project structure, canonical repository metadata, and durable social-link policy.
- Repository Quality CI with dependency-free unit tests on Python 3.10, 3.11, and 3.12.
- Isolated NumPy Part 003 CI job and weekly dependency monitoring.
- Non-destructive Release Readiness workflow that validates, tests, creates SHA-256 manifests, builds a source archive, and uploads temporary artifacts.
- Contributor setup, versioning, reproducibility, dependency, and support policies.
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
- Part-by-part implementation status tracker for the 120-part curriculum.

### Changed
- Modernized GitHub workflows to `actions/checkout@v7`, `actions/setup-python@v7`, and `actions/upload-artifact@v7`.
- Expanded the README from repository planning notes into a runnable quick-start, implementation-status, automation, documentation, and licensing guide.
- Advanced the README, roadmap, package API, validator, and implementation tracker through Parts 001–015.
- Strengthened repository validation to require all implemented Part 001–015 directories, shared modules, and Part 011–015 milestone tests.
- Queued Parts 016–020 as the next training-system engineering milestone.
- Clarified that the companion package/repository version is independent of the book publication version.

### Verified
- Pull request #5 exercised the original complete Repository Quality workflow successfully before merge.
- Pull request #7 merged Parts 006–010 after 65 dependency-free tests passed on Python 3.10, 3.11, and 3.12, the isolated NumPy Part 003 job passed, and repository invariant validation passed.
- The Parts 011–015 milestone is being delivered through a dedicated feature branch and will be merged only after its exact PR head passes the same multi-version CI gate.

## [0.1.0] - 2026-08-18

Initial repository/documentation and Python package foundation for the NeuralForge Complete 120-Part Master Edition companion project.
