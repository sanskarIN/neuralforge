# Changelog

All notable changes to the NeuralForge companion repository will be documented here.

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
- Part-by-part implementation status tracker for the 120-part curriculum.

### Changed
- Modernized GitHub workflows to `actions/checkout@v7`, `actions/setup-python@v7`, and `actions/upload-artifact@v7`.
- Expanded the README from repository planning notes into a runnable quick-start, implementation-status, automation, documentation, and licensing guide.
- Updated the roadmap to mark Phase 1 complete, Phase 2/4 active, and release-readiness foundations established.

### Verified
- Pull request #5 exercised the complete Repository Quality workflow successfully before merge.
- Dependency-free tests passed across Python 3.10, 3.11, and 3.12.
- Part 003 NumPy tests passed in their isolated Python 3.12 CI environment.
- Repository invariant validation passed.

## [1.0.0] - 2026-08-18

Initial repository/documentation foundation for the NeuralForge Complete 120-Part Master Edition companion project.
