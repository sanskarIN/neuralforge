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
- Part 016 companion: MSE/MAE/Huber losses, stable softplus, binary/multiclass cross-entropy from logits, output/loss pairing guidance, demo, and tests.
- Part 017 companion: constant/step/exponential/cosine/warmup learning-rate schedules, optimizer LR application, reduce-on-plateau control, demo, and tests.
- Part 018 companion: Xavier/He/LeCun initialization plans, deterministic matrix generation, signal-variance propagation profiling, demo, and tests.
- Part 019 companion: gradient statistics, zero/vanishing/exploding/non-finite classification, layerwise MLP reports, relative update ratios, demo, and tests.
- Part 020 companion: immutable experiment configs/epoch records, SHA-256 config/data/run fingerprints, optimizer/schedule orchestration, clipping, validation, early stopping, demo, and tests.
- Part-by-part implementation status tracker for the 120-part curriculum.
- Official Ram Sandesh Gumroad storefront integration using canonical URL `https://ramsandesh.gumroad.com`.
- Custom clickable repository storefront asset at `assets/gumroad-storefront.svg`.
- Dedicated `STORE.md` documenting the repository/storefront boundary.
- `.github/FUNDING.yml` custom link for the Gumroad storefront.
- Gumroad storefront visibility in the issue chooser, issue templates, PR template, support/contribution docs, canonical metadata, publishing guide, durable-link policy, release checklist, roadmap, implemented Part READMEs, and project metadata.

### Changed
- Modernized GitHub workflows to `actions/checkout@v7`, `actions/setup-python@v7`, and `actions/upload-artifact@v7`.
- Expanded the README into a runnable quick-start, implementation-status, automation, documentation, licensing, and storefront guide.
- Advanced the README, roadmap, package API, validator, and implementation tracker through Parts 001–020.
- Strengthened repository validation to require all implemented Part 001–020 directories, shared modules, milestone tests, and canonical Gumroad surfaces.
- Queued Parts 021–025 as the first computer-vision/CNN milestone.
- Clarified that the companion package/repository version is independent of the book publication version.
- Promoted `https://ramsandesh.gumroad.com` to a canonical durable storefront destination on GitHub-facing NeuralForge surfaces.

### Verified
- Pull request #5 exercised the original complete Repository Quality workflow successfully before merge.
- Pull request #7 merged Parts 006–010 after 65 dependency-free tests passed on Python 3.10, 3.11, and 3.12, the isolated NumPy Part 003 job passed, and repository invariant validation passed.
- Pull request #8 merged Parts 011–015 after repository validation, Python 3.10/3.11/3.12 tests, and the isolated NumPy Part 003 job all passed on the exact PR head.
- Pull request #10 merged the official Gumroad storefront rollout after canonical storefront validation, NumPy Part 003, and Python 3.10/3.11/3.12 tests all passed on the exact PR head.
- Parts 016–020 are being delivered on a dedicated feature branch and will merge only after the expanded Repository Quality gate passes on the exact milestone head.

## [0.1.0] - 2026-08-18

Initial repository/documentation and Python package foundation for the NeuralForge Complete 120-Part Master Edition companion project.

---

**Official Gumroad Storefront:** **https://ramsandesh.gumroad.com**
