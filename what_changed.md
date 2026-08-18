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
- Implemented companion material for Parts 001-005 with runnable demos and automated tests.
- Part 001 now includes a logistic neuron, numerically stable sigmoid/BCE, full-batch gradient descent, and OR-gate training demo.
- Part 002 now includes tensor-shape inference, flattening, element counting, and reshape validation.
- Part 003 now includes an isolated NumPy 2.5.1 environment, standardization, dense-layer vectorization, stable softmax, demo, and tests.
- Part 004 now includes dependency-free dot products, norms, cosine similarity, transpose, matrix multiplication, and outer products.
- Part 005 now includes numerical derivatives, numerical gradients, and analytical gradient checking.
- Added a 120-part implementation tracker; Parts 006-010 are queued as the next implementation batch.
- Added Dependabot monitoring for GitHub Actions and the active Part 003 Python dependency.
- Added a non-destructive Release Readiness workflow that validates, tests, generates SHA-256 manifests, and creates source archives without automatically publishing a GitHub Release.
- Opened PR #5 specifically to exercise the pull-request CI path. Repository validation, Python 3.10/3.11/3.12 unit-test jobs, and the NumPy Part 003 job all passed before the PR was merged.
- The Python 3.11 CI job executed 31 dependency-free tests successfully; equivalent core suites passed on Python 3.10 and 3.12.
- Modernized workflow dependencies to `actions/checkout@v7`, `actions/setup-python@v7`, and `actions/upload-artifact@v7` after CI exposed old Node-runtime deprecation warnings.
- Refreshed README, roadmap, and changelog so the public repository reflects implemented code rather than only planned work.

## Commit identity

Repository commit metadata was checked directly after the continuation work. The live `main` branch commit metadata records:

- Git author/committer name: `Sanskar`
- Git author/committer email: `sanskarin@outlook.in`

This confirms the requested commit email is being recorded on the GitHub commits created through the connected repository workflow.

For local clones, keep the same email with:

```bash
git config user.email "sanskarin@outlook.in"
```
