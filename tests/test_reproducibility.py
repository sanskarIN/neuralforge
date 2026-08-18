from __future__ import annotations

import os
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge import SeedReport, set_global_seed  # noqa: E402


class ReproducibilityTests(unittest.TestCase):
    def test_python_random_sequence_repeats(self) -> None:
        first_report = set_global_seed(12345)
        first = [random.random() for _ in range(5)]

        second_report = set_global_seed(12345)
        second = [random.random() for _ in range(5)]

        self.assertEqual(first, second)
        self.assertIsInstance(first_report, SeedReport)
        self.assertEqual(first_report.seed, second_report.seed)
        self.assertTrue(first_report.python)

    def test_hash_seed_environment_is_recorded(self) -> None:
        set_global_seed(7)
        self.assertEqual(os.environ.get("PYTHONHASHSEED"), "7")

    def test_boolean_seed_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            set_global_seed(True)

    def test_non_integer_seed_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            set_global_seed(1.5)  # type: ignore[arg-type]

    def test_seed_range_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            set_global_seed(-1)
        with self.assertRaises(ValueError):
            set_global_seed(2**32)


if __name__ == "__main__":
    unittest.main()
