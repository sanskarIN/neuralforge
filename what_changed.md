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
- Advanced the implementation tracker to Parts 001-020 Implemented and Parts 021-025 Queued.
- Exposed the Part 016-020 APIs through `neuralforge.__init__` while retaining the companion package baseline version `0.1.0`.

## 2026-08-18 Gumroad GitHub rollout

- Added a prominent Gumroad badge at the top of the repository README and a dedicated official-store section.
- Added `docs/GUMROAD.md` as the canonical store-link and reusable-badge policy.
- Added/highlighted `https://ramsandesh.gumroad.com` in README, support, publishing, metadata, release, contribution, durable-link, PR-template, and GitHub issue-chooser surfaces.
- Added the canonical Gumroad footer to every implemented Part README, Parts 001-020.
- Kept promotional/store links out of Python implementation modules, tests, dependency files, and numerical output where they have no technical purpose.
- Strengthened repository validation so key reader-facing surfaces and every implemented Part README must retain the canonical Gumroad URL.
- The reusable GitHub visual uses a shields.io Gumroad badge rather than checking a copied third-party logo binary into the repository.

## Current merge gate

- The combined Gumroad + Parts 016-020 milestone is developed on `feat/gumroad-and-parts-016-020`.
- It must pass repository validation, dependency-free tests on Python 3.10, 3.11, and 3.12, and the isolated NumPy Part 003 job before merge.

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
