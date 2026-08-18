# Repository Structure

Recommended long-term layout:

```text
neuralforge/
├─ README.md
├─ STORE.md
├─ LICENSE
├─ BOOK_LICENSE.md
├─ CONTRIBUTING.md
├─ SECURITY.md
├─ CHANGELOG.md
├─ .github/
│  ├─ FUNDING.yml
│  ├─ PULL_REQUEST_TEMPLATE.md
│  └─ ISSUE_TEMPLATE/
├─ assets/
│  └─ gumroad-storefront.svg
├─ docs/
├─ src/
├─ parts/
│  ├─ part-001/
│  ├─ part-002/
│  └─ ... part-120/
├─ labs/
├─ examples/
├─ tests/
└─ tools/
```

Each part folder should contain a concise README, runnable source files, dependency notes, tests or verification steps where useful, and references to the relevant book part without redistributing the paid manuscript.

## Storefront surfaces

The official Ram Sandesh Gumroad storefront is:

**https://ramsandesh.gumroad.com**

Repository-facing storefront integration is intentionally centralized around:

- `STORE.md` for the repository/storefront boundary;
- `assets/gumroad-storefront.svg` for a custom clickable storefront badge;
- `.github/FUNDING.yml` for GitHub's custom funding/store link surface;
- `README.md`, publishing metadata, support docs, and GitHub templates for discoverability.

The custom SVG is a NeuralForge repository storefront graphic and is not represented as an official Gumroad corporate logo.

Generated models, large datasets, credentials, publication binaries, and private manuscript source files should not be committed by default.

---

**Official Gumroad Storefront:** **https://ramsandesh.gumroad.com**
