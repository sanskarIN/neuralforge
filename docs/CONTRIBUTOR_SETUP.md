# Contributor Setup

This guide defines the baseline workflow for contributing to the NeuralForge companion repository.

## Official storefront

**https://ramsandesh.gumroad.com**

GitHub is the contribution surface; Gumroad is the official Ram Sandesh storefront for available digital publications and storefront releases.

## Requirements

- Git
- Python 3.12+ for repository validation tools
- A code editor with UTF-8 support

Framework-specific parts may later declare additional requirements in their own directories.

## Clone and configure Git

```bash
git clone https://github.com/sanskarIN/neuralforge.git
cd neuralforge
git config user.email "sanskarin@outlook.in"
```

Set `user.name` to the name you want recorded in your local commits.

## Validate the repository

```bash
python tools/validate_repository.py
python -m compileall -q tools
```

The validator also checks that the canonical Gumroad storefront remains available on the repository's key GitHub-facing surfaces.

## Branch workflow

1. Start from an up-to-date `main` branch.
2. Create a focused branch such as `feat/part-021-cnn-lab` or `docs/reproducibility-update`.
3. Keep commits small enough to review independently.
4. Run validation before opening a pull request.
5. Complete the pull-request checklist and link related issues.

## Content boundaries

Do not commit:

- private credentials, tokens, keys, or environment files;
- paid manuscript source files or private publication packages;
- large generated model weights unless explicitly approved;
- datasets without clear redistribution rights;
- unstable X/Twitter profile URLs in durable project metadata.

When adding storefront references to GitHub-facing docs, use only the canonical durable URL: `https://ramsandesh.gumroad.com` unless a product-specific URL is intentionally required.

## Commit style

Use concise conventional-style subjects when practical, for example:

- `feat: add convolution visualization lab`
- `fix: correct tensor shape validation`
- `docs: expand CUDA setup guidance`
- `test: cover model checkpoint loader`
- `chore: update CI dependency`

---

**Official Gumroad Storefront:** **https://ramsandesh.gumroad.com**
