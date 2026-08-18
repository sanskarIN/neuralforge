# Dependency Policy

Dependencies should be added deliberately and kept as small as practical for each NeuralForge part, lab, or example.

## Principles

- Prefer maintained, widely used packages when they materially simplify the learning objective.
- Do not add a dependency for functionality that the Python standard library can provide clearly and safely.
- Keep framework-specific dependencies local to the relevant part or lab instead of forcing one giant environment across all 120 parts.
- Record minimum/maximum or tested versions when compatibility matters.
- Avoid abandoned, unlicensed, suspicious, or unnecessary packages.

## Security and updates

GitHub Actions dependencies are monitored through Dependabot. Future language/package ecosystems should be added to `.github/dependabot.yml` only after their manifest files exist.

Security-sensitive dependency updates should be prioritized. Breaking upgrades should include migration notes and tests.

## Licensing

Before introducing a dependency, confirm that its license is compatible with the intended use. Third-party code, datasets, model weights, and assets retain their own licenses and attribution requirements.

## Lock files

Commit lock files when they improve reproducibility for an executable application or lab. For reusable examples/libraries, use dependency bounds appropriate to that ecosystem rather than blindly pinning every transitive dependency.

## Large AI dependencies

GPU frameworks, model runtimes, and accelerator toolkits may have platform-specific installation requirements. Document those requirements alongside the relevant part instead of assuming one environment works everywhere.

---

**Official Gumroad Storefront:** **https://ramsandesh.gumroad.com**
