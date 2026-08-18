# Repository Structure

Recommended long-term layout:

```text
neuralforge/
├─ README.md
├─ LICENSE
├─ BOOK_LICENSE.md
├─ CONTRIBUTING.md
├─ SECURITY.md
├─ CHANGELOG.md
├─ docs/
├─ parts/
│  ├─ part-001/
│  ├─ part-002/
│  └─ ... part-120/
├─ labs/
├─ examples/
├─ tests/
├─ scripts/
└─ assets/
```

Each part folder should contain a concise README, runnable source files, dependency notes, tests or verification steps where useful, and references to the relevant book part without redistributing the paid manuscript.

Generated models, large datasets, credentials, publication binaries, and private manuscript source files should not be committed by default.
