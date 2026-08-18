from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.autograd import Value, graph_summary  # noqa: E402


class AutogradTests(unittest.TestCase):
    def test_add_multiply_and_power_gradients(self) -> None:
        x = Value(2.0, label="x")
        y = Value(-3.0, label="y")
        output = x * y + x**2
        output.backward()

        self.assertAlmostEqual(output.data, -2.0)
        self.assertAlmostEqual(x.grad, 1.0)  # y + 2x = -3 + 4
        self.assertAlmostEqual(y.grad, 2.0)

    def test_shared_subgraph_accumulates_gradient(self) -> None:
        x = Value(3.0)
        shared = x * x
        output = shared + shared
        output.backward()
        self.assertAlmostEqual(x.grad, 12.0)

    def test_nonlinear_chain_matches_closed_form(self) -> None:
        x = Value(0.7)
        output = (x * 2.0 - 0.3).tanh()
        output.backward()
        expected = 2.0 * (1.0 - math.tanh(1.1) ** 2)
        self.assertAlmostEqual(x.grad, expected, places=10)

    def test_exp_log_and_sigmoid_gradients(self) -> None:
        x = Value(1.2)
        output = x.exp().log() + x.sigmoid()
        output.backward()
        probability = 1.0 / (1.0 + math.exp(-1.2))
        expected = 1.0 + probability * (1.0 - probability)
        self.assertAlmostEqual(output.data, 1.2 + probability, places=10)
        self.assertAlmostEqual(x.grad, expected, places=10)

    def test_division_gradient(self) -> None:
        x = Value(6.0)
        y = Value(2.0)
        output = x / y
        output.backward()
        self.assertAlmostEqual(x.grad, 0.5)
        self.assertAlmostEqual(y.grad, -1.5)

    def test_backward_can_clear_existing_gradients(self) -> None:
        x = Value(2.0)
        output = x**2
        output.backward()
        self.assertAlmostEqual(x.grad, 4.0)
        output.backward(clear_grads=True)
        self.assertAlmostEqual(x.grad, 4.0)
        output.zero_grad()
        self.assertEqual(x.grad, 0.0)

    def test_relu_gradient(self) -> None:
        positive = Value(2.0)
        positive.relu().backward()
        self.assertEqual(positive.grad, 1.0)

        negative = Value(-2.0)
        negative.relu().backward()
        self.assertEqual(negative.grad, 0.0)

    def test_graph_summary_counts_structure(self) -> None:
        x = Value(2.0)
        y = Value(4.0)
        output = (x * y + 1.0).sigmoid()
        summary = graph_summary(output)
        self.assertGreaterEqual(summary["nodes"], 6)
        self.assertGreaterEqual(summary["edges"], 5)
        self.assertGreaterEqual(summary["operations"], 3)

    def test_invalid_operations_are_rejected(self) -> None:
        with self.assertRaises(ZeroDivisionError):
            _ = Value(1.0) / Value(0.0)
        with self.assertRaises(ValueError):
            Value(-1.0) ** 0.5
        with self.assertRaises(ValueError):
            Value(0.0).log()


if __name__ == "__main__":
    unittest.main()
