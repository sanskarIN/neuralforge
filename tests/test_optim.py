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
from neuralforge.optim import Adam, Momentum, RMSProp, SGD, clip_grad_norm  # noqa: E402


class OptimizerTests(unittest.TestCase):
    def test_sgd_update(self) -> None:
        parameter = Value(1.0)
        parameter.grad = 0.5
        optimizer = SGD([parameter], learning_rate=0.1)
        optimizer.step()
        self.assertAlmostEqual(parameter.data, 0.95)

    def test_sgd_weight_decay(self) -> None:
        parameter = Value(2.0)
        parameter.grad = 0.0
        SGD([parameter], learning_rate=0.1, weight_decay=0.5).step()
        self.assertAlmostEqual(parameter.data, 1.9)

    def test_momentum_accumulates_velocity(self) -> None:
        parameter = Value(1.0)
        optimizer = Momentum([parameter], learning_rate=0.1, beta=0.9)

        parameter.grad = 0.5
        optimizer.step()
        self.assertAlmostEqual(parameter.data, 0.95)
        self.assertAlmostEqual(optimizer.velocity[0], 0.5)

        parameter.grad = 0.5
        optimizer.step()
        self.assertAlmostEqual(optimizer.velocity[0], 0.95)
        self.assertAlmostEqual(parameter.data, 0.855)

    def test_rmsprop_tracks_squared_gradient(self) -> None:
        parameter = Value(1.0)
        parameter.grad = 0.5
        optimizer = RMSProp(
            [parameter],
            learning_rate=0.1,
            beta=0.9,
            epsilon=1e-8,
        )
        optimizer.step()
        self.assertAlmostEqual(optimizer.average_squared[0], 0.025)
        self.assertLess(parameter.data, 1.0)

    def test_adam_first_update_is_bias_corrected(self) -> None:
        parameter = Value(1.0)
        parameter.grad = 0.5
        optimizer = Adam(
            [parameter],
            learning_rate=0.1,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-12,
        )
        optimizer.step()
        self.assertEqual(optimizer.timestep, 1)
        self.assertAlmostEqual(parameter.data, 0.9, places=10)

    def test_zero_grad(self) -> None:
        first = Value(1.0)
        second = Value(2.0)
        first.grad = 3.0
        second.grad = -4.0
        optimizer = SGD([first, second])
        optimizer.zero_grad()
        self.assertEqual(first.grad, 0.0)
        self.assertEqual(second.grad, 0.0)

    def test_duplicate_parameters_are_updated_once(self) -> None:
        parameter = Value(1.0)
        parameter.grad = 1.0
        optimizer = SGD([parameter, parameter], learning_rate=0.1)
        self.assertEqual(len(optimizer.parameters), 1)
        optimizer.step()
        self.assertAlmostEqual(parameter.data, 0.9)

    def test_global_gradient_clipping(self) -> None:
        first = Value(0.0)
        second = Value(0.0)
        first.grad = 3.0
        second.grad = 4.0
        original_norm = clip_grad_norm([first, second], 1.0)
        self.assertAlmostEqual(original_norm, 5.0)
        self.assertAlmostEqual(first.grad, 0.6)
        self.assertAlmostEqual(second.grad, 0.8)
        self.assertAlmostEqual(math.hypot(first.grad, second.grad), 1.0)

    def test_invalid_optimizer_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SGD([], learning_rate=0.1)
        with self.assertRaises(ValueError):
            SGD([Value(1.0)], learning_rate=0.0)
        with self.assertRaises(ValueError):
            Momentum([Value(1.0)], beta=1.0)
        with self.assertRaises(ValueError):
            Adam([Value(1.0)], epsilon=0.0)

    def test_non_finite_gradient_is_rejected(self) -> None:
        parameter = Value(1.0)
        parameter.grad = math.inf
        with self.assertRaises(ValueError):
            SGD([parameter]).step()


if __name__ == "__main__":
    unittest.main()
