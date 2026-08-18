# Versioning Policy

NeuralForge uses semantic-style versioning for the companion repository while keeping book-edition metadata explicit.

## Repository versions

Use `MAJOR.MINOR.PATCH` tags when code releases begin:

- **MAJOR**: incompatible repository/API/layout changes or a major curriculum generation change.
- **MINOR**: backward-compatible labs, examples, features, or substantial documentation additions.
- **PATCH**: fixes, errata, small documentation corrections, CI maintenance, and compatible refinements.

Example tags: `v1.0.0`, `v1.1.0`, `v1.1.1`.

## Book edition versus repository release

The publication and repository evolve independently. A book may remain **Complete 120-Part Master Edition, Version 1.0** while the companion repository receives later patch/minor releases.

When a repository change requires a new book build, record the relationship in `CHANGELOG.md` and publication metadata rather than silently changing previously published files.

## Pre-release versions

Use suffixes such as `-alpha.1`, `-beta.1`, or `-rc.1` only when a release is intentionally not final.

## Tagging rules

Before tagging:

1. Run repository validation and tests.
2. Update `CHANGELOG.md`.
3. Confirm licensing and attribution for new dependencies/assets.
4. Confirm no secrets or unstable social-profile URLs are present.
5. Confirm the canonical storefront remains `https://ramsandesh.gumroad.com` on GitHub-facing release surfaces.
6. Run the Release Readiness workflow and review its checksum/source artifacts.

Tags should point to reviewed commits on `main`.

---

**Official Gumroad Storefront:** **https://ramsandesh.gumroad.com**
