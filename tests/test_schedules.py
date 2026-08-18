from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.schedules import (  # noqa: E402
    ReduceLROnPlateau,
    constant_lr,
    cosine_decay,
    exponential_decay,
    linear_warmup,
    step_decay,
    warmup_cosine_decay,
)


class ScheduleTests(unittest.TestCase):
    def test_constant_lr(self) -> None:
        self.assertEqual(constant_lr(0.1, 0), 0.1)
        self.assertEqual(constant_lr(0.1, 100), 0.1)

    def test_step_decay(self) -> None:
        values = [step_decay(0.1, step, step_size=3, gamma=0.5) for step in range(7)]
        self.assertEqual(values, [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025])

    def test_exponential_decay(self) -> None:
        self.assertAlmostEqual(exponential_decay(0.2, 3, gamma=0.5), 0.025)

    def test_cosine_decay_endpoints(self) -> None:
        self.assertAlmostEqual(cosine_decay(0.1, 0, total_steps=10, min_lr=0.01), 0.1)
        self.assertAlmostEqual(cosine_decay(0.1, 10, total_steps=10, min_lr=0.01), 0.01)
        self.assertAlmostEqual(cosine_decay(0.1, 100, total_steps=10, min_lr=0.01), 0.01)

    def test_linear_warmup(self) -> None:
        self.assertAlmostEqual(linear_warmup(0.1, 0, warmup_steps=4), 0.025)
        self.assertAlmostEqual(linear_warmup(0.1, 3, warmup_steps=4), 0.1)
        self.assertAlmostEqual(linear_warmup(0.1, 10, warmup_steps=4), 0.1)

    def test_warmup_cosine_is_continuous_at_boundary(self) -> None:
        before = warmup_cosine_decay(0.1, 3, warmup_steps=4, total_steps=12, min_lr=0.01)
        boundary = warmup_cosine_decay(0.1, 4, warmup_steps=4, total_steps=12, min_lr=0.01)
        self.assertAlmostEqual(before, 0.1)
        self.assertAlmostEqual(boundary, 0.1)
        self.assertAlmostEqual(warmup_cosine_decay(0.1, 12, warmup_steps=4, total_steps=12, min_lr=0.01), 0.01)

    def test_plateau_scheduler_reduces_after_patience(self) -> None:
        scheduler = ReduceLROnPlateau(0.1, factor=0.5, patience=2, min_lr=0.01)
        first = scheduler.update(1.0)
        self.assertFalse(first.reduced)
        self.assertEqual(first.learning_rate, 0.1)
        self.assertFalse(scheduler.update(1.1).reduced)
        reduced = scheduler.update(1.2)
        self.assertTrue(reduced.reduced)
        self.assertEqual(reduced.learning_rate, 0.05)

    def test_plateau_improvement_resets_counter(self) -> None:
        scheduler = ReduceLROnPlateau(0.1, patience=2, min_delta=0.01)
        scheduler.update(1.0)
        scheduler.update(1.005)
        improved = scheduler.update(0.98)
        self.assertEqual(improved.bad_epochs, 0)
        self.assertAlmostEqual(improved.best_metric, 0.98)

    def test_plateau_mode_max(self) -> None:
        scheduler = ReduceLROnPlateau(0.2, factor=0.5, patience=1, mode="max")
        scheduler.update(0.5)
        improved = scheduler.update(0.6)
        self.assertFalse(improved.reduced)
        self.assertAlmostEqual(improved.best_metric, 0.6)
        reduced = scheduler.update(0.59)
        self.assertTrue(reduced.reduced)
        self.assertAlmostEqual(reduced.learning_rate, 0.1)

    def test_invalid_schedule_inputs(self) -> None:
        with self.assertRaises(ValueError):
            constant_lr(0.0)
        with self.assertRaises(ValueError):
            step_decay(0.1, 0, step_size=0)
        with self.assertRaises(ValueError):
            exponential_decay(0.1, 1, gamma=1.1)
        with self.assertRaises(ValueError):
            cosine_decay(0.1, 0, total_steps=0)
        with self.assertRaises(ValueError):
            linear_warmup(0.1, 0, warmup_steps=0)
        with self.assertRaises(ValueError):
            warmup_cosine_decay(0.1, 0, warmup_steps=5, total_steps=5)
        with self.assertRaises(ValueError):
            ReduceLROnPlateau(0.1, factor=1.0)
        with self.assertRaises(ValueError):
            ReduceLROnPlateau(0.1, patience=0)
        with self.assertRaises(ValueError):
            ReduceLROnPlateau(0.1, mode="median")
        with self.assertRaises(ValueError):
            ReduceLROnPlateau(0.1).update(math.inf)


if __name__ == "__main__":
    unittest.main()
