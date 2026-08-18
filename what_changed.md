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
- Implemented companion material for Parts 001-010 with runnable demos and automated tests.
- Part 001 includes a logistic neuron, numerically stable sigmoid/BCE, full-batch gradient descent, and OR-gate training demo.
- Part 002 includes tensor-shape inference, flattening, element counting, and reshape validation.
- Part 003 includes an isolated NumPy 2.5.1 environment, standardization, dense-layer vectorization, stable softmax, demo, and tests.
- Part 004 includes dependency-free dot products, norms, cosine similarity, transpose, matrix multiplication, and outer products.
- Part 005 includes numerical derivatives, numerical gradients, and analytical gradient checking.
- Part 006 adds descriptive statistics, covariance/correlation, normal density, Bernoulli likelihood, and reproducible bootstrap intervals.
- Part 007 adds disjoint/stratified train-validation-test splitting and training-only feature standardization to demonstrate leakage-resistant evaluation.
- Part 008 adds quantiles, numeric summaries, IQR outlier detection, histograms, and self-contained SVG scatter-plot generation without another plotting dependency.
- Part 009 adds an inspectable artificial neuron with weighted-contribution traces, common activations, and activation derivatives.
- Part 010 adds separate from-scratch perceptron and logistic-regression implementations with convergence/loss histories and a comparison demo.
- Advanced the 120-part implementation tracker: Parts 001-010 are implemented and Parts 011-015 are queued next.
- Strengthened repository validation so every implemented Part 001-010 must retain a README and runnable Python material, and the required shared modules must exist.
- Added Dependabot monitoring for GitHub Actions and the active Part 003 Python dependency.
- Added a non-destructive Release Readiness workflow that validates, tests, generates SHA-256 manifests, and creates source archives without automatically publishing a GitHub Release.
- PR #5 exercised the original pull-request CI path successfully on Python 3.10/3.11/3.12 plus the NumPy Part 003 job before merge.
- The Python 3.11 PR #5 job executed 31 dependency-free tests successfully; equivalent core suites passed on Python 3.10 and 3.12.
- Modernized workflow dependencies to `actions/checkout@v7`, `actions/setup-python@v7`, and `actions/upload-artifact@v7` after CI exposed old Node-runtime deprecation warnings.
- PR #7 is the validation/merge gate for the completed Parts 006-010 milestone and the upgraded v7 Actions workflow.
- A concurrent PR #6 merged the earlier Part 006 slice while the feature branch continued; PR #7 contains the remaining milestone work and avoids reimplementing that already-merged slice.
- Refreshed README, roadmap, changelog, implementation tracker, and package API to reflect the current repository state.
- Aligned the companion Python/repository baseline at version `0.1.0`; the book's Publication Version 1.0 remains a separate lifecycle.

## Commit identity

Repository commit metadata was checked directly during the continuation work. Git-authored commits record:

- Git author name: `Sanskar`
- Git author email: `sanskarin@outlook.in`

GitHub-created merge commits may use GitHub's own committer identity, while preserving the authored commits and their requested author email in history.

For local clones, keep the same email with:

```bash
git config user.email "sanskarin@outlook.in"
```
