# NeuralForge Companion Repository Roadmap

## Phase 1 — Repository foundation — Complete

Completed foundation work includes:

- core documentation and licensing boundaries;
- contribution, support, security, and conduct policies;
- canonical publication metadata and durable-link policy;
- CODEOWNERS, pull-request template, issue templates, and issue intake configuration;
- citation metadata, editor configuration, dependency policy, and contributor setup;
- official Ram Sandesh Gumroad storefront integration across GitHub-facing surfaces;
- custom clickable storefront badge, `STORE.md`, and `.github/FUNDING.yml`.

Official storefront: **https://ramsandesh.gumroad.com**

## Phase 2 — Part-by-part companion code — In progress

Goal: create organized companion material for Parts 001–120 with runnable examples, dependency declarations where needed, tests, and concise READMEs.

Current progress:

- **Parts 001–025 implemented** with runnable code, demos, and automated tests.
- **Parts 026–120 planned** for incremental implementation after the next canonical title batch is selected.

See `docs/PART_IMPLEMENTATION_STATUS.md` for the detailed tracker.

## Phase 3 — Labs and capstones — Planned

Add practical cross-part labs, reference implementations, legally redistributable sample generators/data, experiment templates, and final capstone scaffolding.

## Phase 4 — Reproducibility and CI — In progress

Completed/in-progress work includes:

- repository invariant validator that enforces Parts 001–025 implementation structure;
- canonical GitHub repository and Gumroad storefront checks on key public surfaces;
- Python compilation checks;
- dependency-free unit-test matrix on Python 3.10, 3.11, and 3.12;
- isolated NumPy Part 003 tests on Python 3.12;
- reproducibility policy and shared seeding utility;
- scalar autodiff, multilayer backpropagation, optimizer-state, regularization, normalization, loss, schedule, initialization, gradient-flow, and training-runner tests;
- first-principles convolution/pooling and CNN architecture tests;
- mobile-vision parameter/MAC cost tests;
- object-detection geometry/NMS/matching tests;
- segmentation confusion/IoU/Dice/mask-processing tests;
- deterministic experiment configuration fingerprints and JSON experiment records;
- Dependabot monitoring for GitHub Actions and active Python dependencies;
- pull-request validation before milestone merges.

Future additions should include framework-specific smoke tests, static analysis, formatting/linting where useful, compatibility matrices, and heavier integration tests only when their maintenance cost is justified.

## Phase 5 — Release engineering — Foundation complete, releases pending

Current release-readiness automation:

- validates repository invariants;
- checks canonical repository/storefront metadata;
- compiles repository Python;
- runs dependency-free tests;
- creates SHA-256 manifests;
- builds a source archive;
- uploads temporary workflow artifacts;
- does **not** publish a GitHub Release automatically.

Tagged public releases should begin only after a deliberate version decision and an appropriate companion-code milestone.

## Phase 6 — Maintenance — Ongoing

Maintain `CHANGELOG.md`, errata, dependency updates, security fixes, tests, compatibility notes, durable project metadata, and the canonical Gumroad storefront destination. Permanent ebook copies should continue to avoid changeable social-media profile links.

## Completed visual-deep-learning milestone — Parts 021–025

The repository now includes framework-light foundations for:

1. Convolutional Neural Networks from First Principles
2. CNN Architecture Design: From LeNet to ResNet
3. Efficient CNNs & Mobile Vision
4. Object Detection from First Principles to Real-Time Systems
5. Image Segmentation: From Pixels to Panoptic Understanding

The milestone progresses from explicit convolution/pooling mechanics into architecture-level shape reasoning, mobile efficiency, box-based detection, and per-pixel segmentation while keeping core logic dependency-free and testable.

## Next implementation milestone

Select the exact Part **026 onward** titles from the canonical 120-Part NeuralForge title inventory before opening the next code branch. This prevents topic-name drift or guessed chapter titles while preserving the established incremental implementation workflow.

---

**Official Gumroad Storefront:** **https://ramsandesh.gumroad.com**
