from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.autograd import Value  # noqa: E402
from neuralforge.optim import SGD  # noqa: E402
from neuralforge.schedules import (  # noqa: E402
    ReduceLROnPlateau,
    apply_learning_rate,
    constant_learning_rate,
    cosine_decay,
    exponential_decay,
    linear_warmup,
    step_decay,
    warmup_cosine_decay,
)


class ScheduleTests(unittest.TestCase):
    def test_constant_schedule(self) -> None:
        self.assertEqual(constant_learning_rate(0.1, 100), 0.1)

    def test_step_decay(self) -> None:
        self.assertAlmostEqual(step_decay(0.1, 0, step_size=5, gamma=0.5), 0.1)
        self.assertAlmostEqual(step_decay(0.1, 5, step_size=5, gamma=0.5), 0.05)
        self.assertAlmostEqual(step_decay(0.1, 12, step_size=5, gamma=0.5), 0.025)

    def test_exponential_decay(self) -> None:
        self.assertAlmostEqual(exponential_decay(0.1, 3, decay_rate=0.9), 0.1 * 0.9**3)

    def test_cosine_endpoints_and_clamping(self) -> None:
        self.assertAlmostEqual(cosine_decay(0.1, 0, total_steps=10, min_lr=0.01), 0.1)
        self.assertAlmostEqual(cosine_decay(0.1, 10, total_steps=10, min_lr=0.01), 0.01)
        self.assertAlmostEqual(cosine_decay(0.1, 999, total_steps=10, min_lr=0.01), 0.01)

    def test_linear_warmup(self) -> None:
        self.assertAlmostEqual(linear_warmup(0.1, 0, warmup_steps=4, start_lr=0.02), 0.02)
        self.assertAlmostEqual(linear_warmup(0.1, 2, warmup_steps=4, start_lr=0.02), 0.06)
        self.assertAlmostEqual(linear_warmup(0.1, 4, warmup_steps=4, start_lr=0.02), 0.1)

    def test_warmup_cosine_is_continuous_at_peak(self) -> None:
        self.assertAlmostEqual(
            warmup_cosine_decay(0.1, 4, warmup_steps=4, total_steps=12),
            0.1,
        )
        after = warmup_cosine_decay(0.1, 5, warmup_steps=4, total_steps=12)
        self.assertLess(after, 0.1)
        self.assertGreater(after, 0.0)
        self.assertAlmostEqual(
            warmup_cosine_decay(0.1, 12, warmup_steps=4, total_steps=12, min_lr=0.01),
            0.01,
        )

    def test_apply_learning_rate_updates_optimizer(self) -> None:
        optimizer = SGD([Value(1.0)], learning_rate=0.1)
        returned = apply_learning_rate(optimizer, 0.025)
        self.assertEqual(returned, 0.025)
        self.assertEqual(optimizer.learning_rate, 0.025)
        with self.assertRaises(TypeError):
            apply_learning_rate(object(), 0.01)

    def test_plateau_controller_reduces_after_patience(self) -> None:
        controller = ReduceLROnPlateau(factor=0.5, patience=2, min_lr=0.01, min_delta=0.01)
        rate = 0.1
        rate = controller.update(1.0, rate)
        self.assertEqual(rate, 0.1)
        rate = controller.update(0.995, rate)
        self.assertEqual(rate, 0.1)
        rate = controller.update(0.994, rate)
        self.assertEqual(rate, 0.05)
        self.assertEqual(controller.reductions, 1)
        rate = controller.update(0.8, rate)
        self.assertEqual(rate, 0.05)
        self.assertAlmostEqual(controller.best or math.nan, 0.8)

    def test_plateau_controller_respects_floor(self) -> None:
        controller = ReduceLROnPlateau(factor=0.1, patience=1, min_lr=0.01)
        rate = controller.update(1.0, 0.02)
        rate = controller.update(1.1, rate)
        self.assertEqual(rate, 0.01)
        rate = controller.update(1.2, rate)
        self.assertEqual(rate, 0.01)

    def test_invalid_schedule_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            step_decay(0.1, -1, step_size=10)
        with self.assertRaises(ValueError):
            step_decay(0.1, 1, step_size=0)
        with self.assertRaises(ValueError):
            cosine_decay(0.1, 1, total_steps=0)
        with self.assertRaises(ValueError):
            linear_warmup(0.1, 1, warmup_steps=0)
        with self.assertRaises(ValueError):
            warmup_cosine_decay(0.1, 1, warmup_steps=4, total_steps=4)
        with self.assertRaises(ValueError):
            ReduceLROnPlateau(factor=1.0)


if __name__ == "__main__":
    unittest.main()
