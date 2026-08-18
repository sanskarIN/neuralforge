#!/usr/bin/env python3
"""Validate stable repository invariants for NeuralForge.

This script intentionally uses only the Python standard library so it can run
in a fresh GitHub Actions runner without installing project dependencies.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_REPO = "https://github.com/sanskarIN/neuralforge"

REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "BOOK_LICENSE.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "CHANGELOG.md",
    "what_changed.md",
    "CITATION.cff",
    "pyproject.toml",
    "docs/METADATA.md",
    "docs/PART_IMPLEMENTATION_STATUS.md",
    "docs/RELEASE_CHECKLIST.md",
    "docs/SOCIAL_LINK_POLICY.md",
    ".github/CODEOWNERS",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/workflows/repository-quality.yml",
    ".github/workflows/release-readiness.yml",
)

REQUIRED_DIRS = (
    "parts",
    "src/neuralforge",
    "labs",
    "examples",
    "tests",
    "tools",
    "docs",
    ".github/ISSUE_TEMPLATE",
)

IMPLEMENTED_PART_DIRS = (
    "parts/001-foundations",
    "parts/002-python-essentials",
    "parts/003-numpy-mastery",
    "parts/004-linear-algebra",
    "parts/005-calculus",
    "parts/006-probability-statistics",
    "parts/007-data-preparation",
    "parts/008-visualization-eda",
    "parts/009-artificial-neuron",
    "parts/010-perceptron-logistic",
)

REQUIRED_SOURCE_MODULES = (
    "src/neuralforge/__init__.py",
    "src/neuralforge/reproducibility.py",
    "src/neuralforge/foundations.py",
    "src/neuralforge/tensor_basics.py",
    "src/neuralforge/linear_algebra.py",
    "src/neuralforge/calculus.py",
    "src/neuralforge/statistics.py",
    "src/neuralforge/data_preparation.py",
    "src/neuralforge/eda.py",
    "src/neuralforge/neuron.py",
    "src/neuralforge/perceptron.py",
    "src/neuralforge/logistic_regression.py",
)

TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".yml",
    ".yaml",
    ".py",
    ".toml",
    ".json",
    ".cff",
}

SOCIAL_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:x\.com|twitter\.com)/[^\s)>\]}]+",
    re.IGNORECASE,
)


def iter_text_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if (
            path.name in {"LICENSE", ".gitignore", ".editorconfig"}
            or path.suffix.lower() in TEXT_SUFFIXES
        ):
            files.append(path)
    return files


def validate_implemented_parts(errors: list[str]) -> None:
    for relative in IMPLEMENTED_PART_DIRS:
        directory = ROOT / relative
        if not directory.is_dir():
            errors.append(f"missing implemented part directory: {relative}")
            continue
        if not (directory / "README.md").is_file():
            errors.append(f"implemented part is missing README.md: {relative}")
        python_files = list(directory.glob("*.py"))
        if not python_files:
            errors.append(f"implemented part has no runnable Python material: {relative}")


def main() -> int:
    errors: list[str] = []

    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    for relative in REQUIRED_DIRS:
        if not (ROOT / relative).is_dir():
            errors.append(f"missing required directory: {relative}")

    for relative in REQUIRED_SOURCE_MODULES:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required source module: {relative}")

    validate_implemented_parts(errors)

    readme = ROOT / "README.md"
    if readme.is_file() and CANONICAL_REPO not in readme.read_text(encoding="utf-8"):
        errors.append("README.md does not contain the canonical repository URL")

    for path in iter_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for match in SOCIAL_URL_RE.finditer(text):
            relative = path.relative_to(ROOT)
            errors.append(
                f"unstable X/Twitter URL found in {relative}: {match.group(0)}"
            )

    if errors:
        print("NeuralForge repository validation FAILED:")
        for error in errors:
            print(f" - {error}")
        return 1

    print("NeuralForge repository validation passed.")
    print(f"Canonical repository: {CANONICAL_REPO}")
    print(f"Checked {len(REQUIRED_FILES)} required files.")
    print(f"Checked {len(REQUIRED_DIRS)} required directories.")
    print(f"Checked {len(REQUIRED_SOURCE_MODULES)} shared source modules.")
    print(f"Checked {len(IMPLEMENTED_PART_DIRS)} implemented part directories.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
