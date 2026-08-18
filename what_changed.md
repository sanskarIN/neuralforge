# What Changed

## 2026-08-18 publication/repository foundation

- Initialized the NeuralForge companion repository.
- Confirmed canonical repository: `https://github.com/sanskarIN/neuralforge`.
- Added separate licensing boundaries for MIT-licensed companion code and copyrighted book/publication assets.
- Added contribution, security, conduct, release, metadata, publishing, repository-structure, roadmap, errata, testing, issue-template, support, dependency, versioning, reproducibility, and citation documentation.
- Added workspace guides for Parts 1-120, labs, examples, and tests.
- Adopted a durable-link policy: no X/Twitter URL is embedded in immutable publication files because social handles can change after purchase.
- Prepared Version 1.0 publication assets: master PDF, editable DOCX, reflowable EPUB, storefront cover, free preview, detailed TOC, store metadata, product descriptions, publishing guide, source archive, QA reports, and checksums.
- Created purpose-specific release packages: `NeuralForge_Storefront_Package_v1.0.zip`, `NeuralForge_Publication_Package_v1.0.zip`, `NeuralForge_Author_Source_Package_v1.0.zip`, and a complete bundle containing all three.
- Publication source audit confirms Parts 1-120 are represented; the recovered source set required reconstruction of Part 59, documented separately in the publication change log.
- The final fixed-layout PDF, EPUB, reconstructed Part 59, preview, and 120-part source archive contain no X/Twitter URL. The canonical NeuralForge repository is used for durable project linking.

## 2026-08-18 companion-code implementation continuation

- Added GitHub-native CODEOWNERS, pull-request template, feature/documentation issue templates, and issue intake configuration.
- Added a repository invariant validator and GitHub Actions Repository Quality workflow.
- Added a Python package foundation under `src/neuralforge/`.
- Added reproducibility helpers and dependency-free educational utilities.
- Implemented companion material for Parts 001-015 with runnable demos and automated tests.
- Parts 001-005 cover the dependency-free foundations: neurons/classification, Python tensor structure, NumPy vectorization, linear algebra, and calculus/gradient checking.
- Parts 006-010 add statistics, leakage-resistant data preparation, EDA/visualization, inspectable artificial neurons, perceptron training, and logistic regression.
- Part 011 adds a scalar reverse-mode autodiff engine with computational-graph tracking, topological backward traversal, shared-subgraph accumulation, nonlinear operations, and graph summaries.
- Part 012 builds `Neuron`, `Layer`, and `MLP` modules directly on that autodiff engine, with differentiable losses and end-to-end multilayer training.
- Part 013 adds stateful SGD, Momentum, RMSProp, Adam, optional weight decay, global gradient clipping, and exact optimizer-state/update tests.
- Part 014 adds differentiable L1/L2 penalties, deterministic inverted dropout, early stopping, generalization-gap monitoring, and parameter-norm monitoring.
- Part 015 adds differentiable batch/layer normalization, trainable affine normalization parameters, running evaluation moments, and numerically stable scalar softmax.
- Every Part 011-015 has a dedicated README, runnable demo, reusable source module, and automated tests.
- Advanced the 120-part implementation tracker: Parts 001-015 are implemented and Parts 016-020 are queued next.
- Parts 016-020 are queued for loss/output-layer design, learning-rate schedules, initialization/signal propagation, gradient-flow diagnostics, and reproducible training-loop engineering.
- Strengthened repository validation so every implemented Part 001-015 must retain a README and runnable Python material; all shared source modules and Part 011-015 milestone tests are required as well.
- Added Dependabot monitoring for GitHub Actions and the active Part 003 Python dependency.
- Added a non-destructive Release Readiness workflow that validates, tests, generates SHA-256 manifests, and creates source archives without automatically publishing a GitHub Release.
- PR #5 exercised the original pull-request CI path successfully on Python 3.10/3.11/3.12 plus the NumPy Part 003 job before merge.
- PR #7 merged the completed Parts 006-010 milestone after 65 dependency-free tests passed on Python 3.10/3.11/3.12, the isolated NumPy Part 003 job passed, and the strengthened repository validator passed.
- Modernized workflow dependencies to `actions/checkout@v7`, `actions/setup-python@v7`, and `actions/upload-artifact@v7` after CI exposed old Node-runtime deprecation warnings.
- Refreshed README, roadmap, changelog, implementation tracker, validator, and package API through Part 015.
- The companion Python/repository baseline remains version `0.1.0`; the book's Publication Version 1.0 remains a separate lifecycle.
- The Parts 011-015 feature branch will be merged only after its exact PR head passes the full Repository Quality matrix.

## Commit identity

Repository commit metadata was checked directly during the continuation work. Git-authored commits record:

- Git author name: `Sanskar`
- Git author email: `sanskarin@outlook.in`

GitHub-created merge commits may use GitHub's own committer identity, while preserving the authored commits and their requested author email in history.

For local clones, keep the same email with:

```bash
git config user.email "sanskarin@outlook.in"
```
