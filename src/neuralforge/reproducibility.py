"""Reproducibility helpers shared by NeuralForge examples.

The functions in this module avoid hard dependencies on NumPy, PyTorch, or
TensorFlow. If one of those frameworks is installed, its random generator is
seeded too; otherwise the helper continues with the available runtimes.
"""

from __future__ import annotations

import importlib
import os
import random
from dataclasses import dataclass

_MIN_SEED = 0
_MAX_SEED = 2**32 - 1


@dataclass(frozen=True, slots=True)
class SeedReport:
    """Describe which random-number generators were configured."""

    seed: int
    python: bool
    numpy: bool
    torch: bool
    tensorflow: bool
    deterministic_requested: bool
    notes: tuple[str, ...] = ()


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not _MIN_SEED <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be between {_MIN_SEED} and {_MAX_SEED}")
    return seed


def _optional_import(module_name: str):
    try:
        return importlib.import_module(module_name), None
    except ModuleNotFoundError:
        return None, None
    except Exception as exc:  # pragma: no cover - depends on local installations
        return None, f"{module_name} could not be initialized: {exc.__class__.__name__}"


def set_global_seed(seed: int, *, deterministic: bool = False) -> SeedReport:
    """Seed common Python/ML random generators when their frameworks exist.

    Setting ``PYTHONHASHSEED`` affects child processes started after this call;
    it cannot retroactively change hash randomization in the current process.
    Deterministic framework execution is requested only when ``deterministic``
    is true. Hardware/runtime limitations can still prevent bit-for-bit
    reproducibility, so callers should record their environment as well.
    """

    seed = _validate_seed(seed)
    notes: list[str] = []

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    numpy_seeded = False
    numpy, note = _optional_import("numpy")
    if note:
        notes.append(note)
    if numpy is not None:
        numpy.random.seed(seed)
        numpy_seeded = True

    torch_seeded = False
    torch, note = _optional_import("torch")
    if note:
        notes.append(note)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.use_deterministic_algorithms(True)
                if getattr(torch.backends, "cudnn", None) is not None:
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
            torch_seeded = True
        except Exception as exc:  # pragma: no cover - framework/environment specific
            notes.append(f"torch seeding was incomplete: {exc.__class__.__name__}")

    tensorflow_seeded = False
    tensorflow, note = _optional_import("tensorflow")
    if note:
        notes.append(note)
    if tensorflow is not None:
        try:
            tensorflow.random.set_seed(seed)
            if deterministic:
                enable_determinism = getattr(
                    getattr(tensorflow, "config", None),
                    "experimental",
                    None,
                )
                enable_determinism = getattr(
                    enable_determinism,
                    "enable_op_determinism",
                    None,
                )
                if enable_determinism is not None:
                    enable_determinism()
            tensorflow_seeded = True
        except Exception as exc:  # pragma: no cover - framework/environment specific
            notes.append(
                f"tensorflow seeding was incomplete: {exc.__class__.__name__}"
            )

    return SeedReport(
        seed=seed,
        python=True,
        numpy=numpy_seeded,
        torch=torch_seeded,
        tensorflow=tensorflow_seeded,
        deterministic_requested=deterministic,
        notes=tuple(notes),
    )
