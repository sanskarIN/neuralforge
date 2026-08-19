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
- PR #8 merged the completed Parts 011-015 milestone after all five Repository Quality jobs passed on the exact PR head.
- Modernized workflow dependencies to `actions/checkout@v7`, `actions/setup-python@v7`, and `actions/upload-artifact@v7` after CI exposed old Node-runtime deprecation warnings.
- Refreshed README, roadmap, changelog, implementation tracker, validator, and package API through Part 015.
- The companion Python/repository baseline remains version `0.1.0`; the book's Publication Version 1.0 remains a separate lifecycle.

## 2026-08-18 Gumroad storefront integration

- Added the canonical storefront URL everywhere it is appropriate on GitHub-facing project surfaces: `https://ramsandesh.gumroad.com`.
- Added `assets/gumroad-storefront.svg`, a custom NeuralForge storefront badge with a direct Gumroad link when embedded in Markdown.
- Added `STORE.md` as the central repository/storefront boundary document.
- Added `.github/FUNDING.yml` with the Gumroad storefront as a custom GitHub funding/store link.
- Highlighted the Gumroad storefront near the top of `README.md`, in publication metadata, durable-link documentation, support/contribution docs, publishing guidance, release QA, repository structure, and roadmap.
- Added the Gumroad destination to the GitHub issue chooser, bug-report template, documentation template, feature-request template, and pull-request template.
- Added a release-check requirement that the storefront URL remain exactly `https://ramsandesh.gumroad.com` on canonical GitHub-facing surfaces.
- Repository validation requires the storefront badge, `STORE.md`, `.github/FUNDING.yml`, and canonical Gumroad URL on key public files.
- The storefront badge is explicitly documented as a custom NeuralForge repository graphic rather than an official Gumroad corporate logo.
- PR #10 merged the Gumroad rollout after the complete Repository Quality matrix passed on its exact head.

## 2026-08-19 Parts 016-020 clean milestone

- Implemented Part 016 with logits-first binary/categorical cross-entropy, regression losses, Huber loss, stable log-sum-exp, output-layer design recommendations, demo, and tests.
- Implemented Part 017 with constant, step, exponential, cosine, warmup-cosine learning-rate schedules and stateful plateau reduction, with demo and tests.
- Implemented Part 018 with Xavier/Glorot, He/Kaiming, LeCun, uniform/zero initialization helpers plus deterministic forward-signal propagation summaries, demo, and tests.
- Implemented Part 019 with global/per-layer gradient statistics, finite-gradient checks, gradient/parameter ratios, configurable health classification, demo, and tests.
- Implemented Part 020 with reproducible experiment configuration, deterministic fingerprints, optimizer/schedule integration, gradient diagnostics, optional clipping, epoch metrics, JSON run records, demo, and tests.
- Exposed Parts 016-020 through the shared `neuralforge` package API.
- Advanced the 120-part implementation tracker to Parts 001-020 implemented and Parts 021-025 queued.
- Updated README and roadmap to the visual-deep-learning milestone: convolution mechanics, CNN architectures, mobile/efficient vision, object detection, and segmentation.
- Expanded repository invariants from 15 to 20 implemented Part directories, from 17 to 22 required shared source modules, and from 5 to 10 milestone test modules.
- Extended CI validation so every implemented Part README must retain `https://ramsandesh.gumroad.com` while technical source/test files remain free of promotional-link requirements.
- Detected that the earlier combined branch diverged from `main` because the Gumroad rollout had already merged separately in PR #10.
- Rebuilt Parts 016-020 on `feat/parts-016-020-clean`, based directly on current `main`, reusing the already-reviewed Git blobs so code/tests/demos were preserved byte-for-byte while duplicated Gumroad history was removed.
- Closed redundant PR #13 without merging and opened clean replacement PR #16 from a branch that is 0 commits behind `main`.
- PR #16 remains gated on its exact-head Repository Quality run before merge.

## 2026-08-19 Parts 021-025 staged vision milestone

- Created Issue #17 as the formal visual-deep-learning milestone with per-Part scope and merge criteria.
- Created `staging/parts-021-025` from the frozen PR #16 head so vision development could continue without changing PR #16's tested commit.
- Implemented Part 021 with dependency-free single-channel 2D cross-correlation/convolution, symmetric padding, stride, dilation, output-shape helpers, max/average pooling, an edge-response demo, and hand-computed tests.
- Implemented Part 022 with architecture-as-data shape propagation and exact parameter counting for LeNet-5, AlexNet, VGG-11, and ResNet-18-style networks, including grouped convolution and automatic residual projection accounting.
- Implemented Part 023 with standard/depthwise-separable convolution parameter/MAC estimates, FP32 memory estimates, MobileNet-style channel divisibility, width/resolution scaling, inverted-residual cost models, and efficiency-ratio tests.
- Implemented Part 024 with bounding-box coordinate conversion, clipping, IoU, class-aware/class-agnostic NMS, score limits, one-to-one greedy matching, precision/recall metrics, and a post-processing demo.
- Implemented Part 025 with binary/multiclass segmentation confusion metrics, IoU/Dice, absent-class handling, ignore labels, probability thresholding, deterministic majority filtering, and a segmentation evaluation demo.
- Added dedicated README/demo/test material for every Part 021-025; every README retains `https://ramsandesh.gumroad.com`.
- Exposed the complete vision utility set through `src/neuralforge/__init__.py`.
- Advanced the staging implementation tracker and README to Parts 001-025 implemented.
- Expanded staging repository validation to 25 implemented Part directories, 27 shared source modules, and 15 milestone test modules.
- Kept technical source and tests free of promotional-link requirements while preserving the Gumroad requirement on reader-facing Part READMEs and canonical public surfaces.
- The staging branch will not be merged directly. After PR #16 merges, its new vision blobs will be moved onto updated `main` as a clean Parts 021-025 branch, then validated through its own pull request.

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
