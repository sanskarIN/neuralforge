# NeuralForge Companion Repository Roadmap

## Phase 1 — Repository foundation — Complete

Completed foundation work includes:

- core documentation and licensing boundaries;
- contribution, support, security, and conduct policies;
- canonical publication metadata and durable-link policy;
- CODEOWNERS, pull-request template, issue templates, and issue intake configuration;
- citation metadata, editor configuration, dependency policy, and contributor setup.

## Phase 2 — Part-by-part companion code — In progress

Goal: create organized companion material for Parts 001–120 with runnable examples, dependency declarations where needed, tests, and concise READMEs.

Current progress:

- **Parts 001–005 implemented** with runnable code and automated tests.
- **Parts 006–010 queued** as the next dependency-light foundations batch.
- **Parts 011–120 planned** for incremental implementation.

See `docs/PART_IMPLEMENTATION_STATUS.md` for the detailed tracker.

## Phase 3 — Labs and capstones — Planned

Add practical cross-part labs, reference implementations, legally redistributable sample generators/data, experiment templates, and final capstone scaffolding.

## Phase 4 — Reproducibility and CI — In progress

Completed/in-progress work includes:

- repository invariant validator;
- Python compilation checks;
- dependency-free unit-test matrix on Python 3.10, 3.11, and 3.12;
- isolated NumPy Part 003 tests on Python 3.12;
- reproducibility policy and shared seeding utility;
- Dependabot monitoring for GitHub Actions and active Python dependencies.

Future additions should include framework-specific smoke tests, static analysis, formatting/linting where useful, compatibility matrices, and heavier integration tests only when their maintenance cost is justified.

## Phase 5 — Release engineering — Foundation complete, releases pending

Current release-readiness automation:

- validates repository invariants;
- compiles repository Python;
- runs dependency-free tests;
- creates SHA-256 manifests;
- builds a source archive;
- uploads temporary workflow artifacts;
- does **not** publish a GitHub Release automatically.

Tagged public releases should begin only after a deliberate version decision and an appropriate companion-code milestone.

## Phase 6 — Maintenance — Ongoing

Maintain `CHANGELOG.md`, errata, dependency updates, security fixes, tests, compatibility notes, and durable project metadata. Permanent ebook copies should continue to avoid changeable social-media profile links.

## Next implementation milestone

Implement Parts **006–010**:

1. Probability and Statistics for Learning Systems
2. Data Preparation and Leakage-Resistant Evaluation
3. Visualization and Exploratory Data Analysis
4. From a Biological Metaphor to an Artificial Neuron
5. Build a Perceptron and Logistic Classifier from Scratch

The milestone should preserve the current rule: every implemented part should have meaningful runnable material, validation/tests where applicable, and clear dependency/provenance documentation.
