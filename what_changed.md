# What Changed

## 2026-08-18 publication/repository foundation

- Initialized the NeuralForge companion repository.
- Confirmed canonical repository: `https://github.com/sanskarIN/neuralforge`.
- Confirmed official Ram Sandesh Gumroad storefront: `https://ramsandesh.gumroad.com`.
- Added separate licensing boundaries for MIT-licensed companion code and copyrighted book/publication assets.
- Added contribution, security, conduct, release, metadata, publishing, repository-structure, roadmap, errata, testing, issue-template, support, dependency, versioning, reproducibility, and citation documentation.
- Added workspace guides for Parts 1-120, labs, examples, and tests.
- Adopted a durable-link policy: no X/Twitter URL is embedded in immutable publication files because social handles can change after purchase.
- Prepared Version 1.0 publication assets: master PDF, editable DOCX, reflowable EPUB, storefront cover, free preview, detailed TOC, store metadata, product descriptions, publishing guide, source archive, QA reports, and checksums.
- Created purpose-specific release packages: `NeuralForge_Storefront_Package_v1.0.zip`, `NeuralForge_Publication_Package_v1.0.zip`, `NeuralForge_Author_Source_Package_v1.0.zip`, and a complete bundle containing all three.
- Publication source audit confirms Parts 1-120 are represented; the recovered source set required reconstruction of Part 59, documented separately in the publication change log.
- The canonical NeuralForge repository and official Gumroad storefront are used as durable project/store destinations.

## 2026-08-18 companion-code implementation continuation

- Added GitHub-native CODEOWNERS, pull-request template, feature/documentation issue templates, and issue intake configuration.
- Added a repository invariant validator and GitHub Actions Repository Quality workflow.
- Added a Python package foundation under `src/neuralforge/`.
- Added reproducibility helpers and dependency-free educational utilities.
- Implemented companion material for Parts 001-020 with runnable demos and automated tests.
- Parts 001-005 cover the dependency-free foundations: neurons/classification, Python tensor structure, NumPy vectorization, linear algebra, and calculus/gradient checking.
- Parts 006-010 add statistics, leakage-resistant data preparation, EDA/visualization, inspectable artificial neurons, perceptron training, and logistic regression.
- Part 011 adds a scalar reverse-mode autodiff engine with computational-graph tracking, topological backward traversal, shared-subgraph accumulation, nonlinear operations, and graph summaries.
- Part 012 builds `Neuron`, `Layer`, and `MLP` modules directly on that autodiff engine, with differentiable losses and end-to-end multilayer training.
- Part 013 adds stateful SGD, Momentum, RMSProp, Adam, optional weight decay, global gradient clipping, and exact optimizer-state/update tests.
- Part 014 adds differentiable L1/L2 penalties, deterministic inverted dropout, early stopping, generalization-gap monitoring, and parameter-norm monitoring.
- Part 015 adds differentiable batch/layer normalization, trainable affine normalization parameters, running evaluation moments, and numerically stable scalar softmax.
- Part 016 adds MSE, MAE, Huber loss, numerically stable softplus, binary cross-entropy from logits, multiclass cross-entropy from logits, and explicit output/loss pairing guidance.
- Part 017 adds constant, step, exponential, cosine, linear-warmup, warmup+cosine schedules, optimizer learning-rate updates, and validation-driven reduce-on-plateau control.
- Part 018 adds Xavier/Glorot, He/Kaiming, LeCun, and zero initialization plans, deterministic matrix generation, and multi-layer signal-variance propagation diagnostics.
- Part 019 adds layerwise gradient statistics, zero/vanishing/healthy/exploding/non-finite classification, MLP parameter grouping, and relative update ratios.
- Part 020 adds immutable experiment configuration and epoch records, SHA-256 configuration/data/run fingerprints, optimizer/schedule orchestration, gradient clipping/health capture, validation tracking, and early stopping.
- Every Part 001-020 has a dedicated README with the canonical Gumroad storefront footer and runnable Python material.
- Advanced the 120-part implementation tracker: Parts 001-020 are implemented and Parts 021-025 are queued next.
- Parts 021-025 begin the computer-vision sequence: convolution from first principles, CNN architecture design, efficient/mobile vision, object detection, and image segmentation.
- Strengthened repository validation so every implemented Part 001-020 must retain a README, runnable Python material, and the exact canonical Gumroad storefront; all shared modules and milestone tests are required as well.
- Added Dependabot monitoring for GitHub Actions and the active Part 003 Python dependency.
- Added a non-destructive Release Readiness workflow that validates, tests, generates SHA-256 manifests, and creates source archives without automatically publishing a GitHub Release.
- PR #5 exercised the original pull-request CI path successfully on Python 3.10/3.11/3.12 plus the NumPy Part 003 job before merge.
- PR #7 merged the completed Parts 006-010 milestone after 65 dependency-free tests passed on Python 3.10/3.11/3.12, the isolated NumPy Part 003 job passed, and the strengthened repository validator passed.
- PR #8 merged the completed Parts 011-015 milestone after all five Repository Quality jobs passed on the exact PR head.
- PR #10 merged the full GitHub-facing Gumroad storefront integration after the new canonical storefront validator, NumPy Part 003 job, and Python 3.10/3.11/3.12 tests all passed.
- Modernized workflow dependencies to `actions/checkout@v7`, `actions/setup-python@v7`, and `actions/upload-artifact@v7` after CI exposed old Node-runtime deprecation warnings.
- Refreshed README, roadmap, changelog, implementation tracker, validator, and package API through Part 020.
- The companion Python/repository baseline remains version `0.1.0`; the book's Publication Version 1.0 remains a separate lifecycle.

## 2026-08-18 Gumroad storefront integration

- Added the canonical storefront URL everywhere it is appropriate on GitHub-facing project surfaces: `https://ramsandesh.gumroad.com`.
- Added `assets/gumroad-storefront.svg`, a custom NeuralForge storefront badge with a direct Gumroad link when embedded in Markdown.
- Added `STORE.md` as the central repository/storefront boundary document.
- Added `.github/FUNDING.yml` with the Gumroad storefront as a custom GitHub funding/store link.
- Highlighted the Gumroad storefront near the top of `README.md`, in publication metadata, durable-link documentation, support/contribution docs, publishing guidance, release QA, repository structure, roadmap, project metadata, and all implemented Part READMEs.
- Added the Gumroad destination to the GitHub issue chooser, bug-report template, documentation template, feature-request template, and pull-request template.
- Added a release-check requirement that the storefront URL remain exactly `https://ramsandesh.gumroad.com` on canonical GitHub-facing surfaces.
- The repository validator now requires the storefront badge, `STORE.md`, `.github/FUNDING.yml`, the canonical URL across 19 key public surfaces, and the canonical Gumroad footer in every implemented Part README.
- The storefront badge is explicitly documented as a custom NeuralForge repository graphic rather than an official Gumroad corporate logo.
- PR #10 merged the storefront rollout only after all Repository Quality jobs passed on its exact head.

## Commit identity

Repository commit metadata was checked directly during the continuation work. Git-authored commits record:

- Git author name: `Sanskar`
- Git author email: `sanskarin@outlook.in`

GitHub-created merge commits may use GitHub's own committer identity, while preserving the authored commits and their requested author email in history.

For local clones, keep the same email with:

```bash
git config user.email "sanskarin@outlook.in"
```

---

**Official Gumroad Storefront:** **https://ramsandesh.gumroad.com**
