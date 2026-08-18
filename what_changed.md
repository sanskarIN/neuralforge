# What Changed

## 2026-08-18 publication/repository foundation

- Initialized the NeuralForge companion repository.
- Confirmed canonical repository: `https://github.com/sanskarIN/neuralforge`.
- Added separate licensing boundaries for MIT-licensed companion code and copyrighted book/publication assets.
- Added contribution, security, conduct, release, metadata, publishing, repository-structure, roadmap, errata, testing, issue-template, support, dependency, versioning, reproducibility, citation, and Gumroad/store-link documentation.
- Added workspace guides for Parts 1-120, labs, examples, and tests.
- Adopted a durable-link policy: no X/Twitter URL is embedded in immutable publication files because social handles can change after purchase.
- Canonical reader-facing store destination: `https://ramsandesh.gumroad.com`.
- Prepared Version 1.0 publication assets: master PDF, editable DOCX, reflowable EPUB, storefront cover, free preview, detailed TOC, store metadata, product descriptions, publishing guide, source archive, QA reports, and checksums.
- Publication source audit confirms Parts 1-120 are represented; the recovered source set required reconstruction of Part 59, documented separately in the publication change log.

## 2026-08-18 companion-code implementation continuation

- Added GitHub-native CODEOWNERS, pull-request template, feature/documentation issue templates, issue intake configuration, CI, Dependabot, release-readiness automation, and repository validation.
- Added the shared `src/neuralforge/` Python package and dependency-free educational implementations.
- PR #7 merged Parts 006-010 after the full Repository Quality gate passed.
- PR #8 merged Parts 011-015 after validation, NumPy Part 003, and Python 3.10/3.11/3.12 jobs all passed.
- Implemented Parts 001-020 with dedicated reader-facing READMEs, runnable demos, reusable implementations, and automated tests.
- Part 016 adds regression losses, Huber loss, stable softplus/log-sum-exp, BCE/categorical cross-entropy from logits, and output-layer/objective design recommendations.
- Part 017 adds constant, step, exponential, cosine, linear warmup, warmup+cosine schedules, and stateful validation plateau control.
- Part 018 adds Xavier/Glorot, He/Kaiming, LeCun and educational initialization schemes plus deterministic forward signal-propagation reports.
- Part 019 adds gradient statistics, finite guards, layer grouping, gradient-to-parameter ratios, and configurable gradient-health diagnosis.
- Part 020 adds deterministic experiment configurations and fingerprints, optimizer/schedule integration, finite checks, clipping, epoch telemetry, final-state capture, and JSON experiment records.

## 2026-08-18 Gumroad GitHub rollout

- Added a prominent Gumroad badge at the top of the repository README and a dedicated official-store section.
- Added `docs/GUMROAD.md` as the canonical store-link and reusable-badge policy.
- Added/highlighted `https://ramsandesh.gumroad.com` in README, support, publishing, metadata, release, contribution, durable-link, PR-template, and GitHub issue-chooser surfaces.
- Added the canonical Gumroad footer to every implemented Part README.
- Kept promotional/store links out of Python implementation modules, tests, dependency files, and numerical output where they have no technical purpose.
- Strengthened repository validation so key reader-facing surfaces and every implemented Part README must retain the canonical Gumroad URL.
- The reusable GitHub visual uses a shields.io Gumroad badge rather than checking a copied third-party logo binary into the repository.

## 2026-08-18 first vision milestone — Parts 021-025

- Implemented Part 021 with dependency-free 2D cross-correlation/convolution, mathematical kernel flipping, SAME/explicit padding, stride, dilation, output geometry, max pooling, and average pooling.
- Implemented Part 022 with architecture-level shape propagation, convolution parameter/MAC estimates, grouped convolution validation, receptive-field/output-jump tracking, residual compatibility, and a LeNet-style feature-extractor report.
- Implemented Part 023 with standard versus depthwise-separable convolution cost models, inverted-residual cost analysis, model-size estimates across parameter precision, and width-multiplier channel rounding.
- Implemented Part 024 with validated bounding boxes, center-coordinate conversion, IoU, IoU matrices, class-aware NMS, score-ordered one-to-one matching, and precision/recall evaluation.
- Implemented Part 025 with semantic-segmentation confusion matrices, ignored labels, pixel accuracy, per-class/mean IoU and Dice, binary-mask metrics, and panoptic PQ/SQ/RQ decomposition.
- Fixed an ignore-label inference edge case before CI so predictions at ignored target pixels do not expand the inferred segmentation class space.
- Added runnable demos and dedicated Gumroad-linked READMEs for Parts 021-025.
- Added automated tests for all five new vision modules.
- Added `src/neuralforge/vision.py` as the public vision namespace for reusable Parts 021-025 APIs.
- Advanced the implementation tracker, README, roadmap, changelog, and repository validator through Part 025.
- Repository validation now requires all 25 implemented Part directories, all new vision source modules, all new vision tests, and canonical Gumroad links in each implemented Part README.
- Opened Issue #14 as the formal milestone tracker for Parts 021-025.
- Parts 026-030 remain planned until their exact titles are synchronized from the finalized canonical title inventory; obsolete draft numbering is intentionally not copied back into GitHub.

## Current merge gates

- PR #13 is the formal review/CI gate for the combined Gumroad + Parts 016-020 milestone on `feat/gumroad-and-parts-016-020`.
- GitHub has not yet exposed a Repository Quality Actions run for PR #13's exact head despite valid PR-open, reopen, and synchronize events; PR #13 remains open and unmerged rather than bypassing CI policy.
- Parts 021-025 continue on the stacked `feat/parts-021-025` branch based on PR #13's exact feature head.
- The Parts 021-025 branch must pass its own repository validation and full test matrix before it can be merged after its base milestone is resolved.

## Commit identity

Repository Git-authored commits use:

- Git author name: `Sanskar`
- Git author email: `sanskarin@outlook.in`

GitHub-created merge commits may use GitHub's committer identity while preserving authored commits and their author metadata.

For local clones:

```bash
git config user.email "sanskarin@outlook.in"
```

---

**Official store:** **[https://ramsandesh.gumroad.com](https://ramsandesh.gumroad.com)**
