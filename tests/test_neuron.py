from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.neuron import (  # noqa: E402
    ArtificialNeuron,
    activation_derivative,
    leaky_relu,
    relu,
)


class NeuronTests(unittest.TestCase):
    def test_trace_exposes_weighted_contributions(self) -> None:
        neuron = ArtificialNeuron((0.5, -1.0, 2.0), bias=0.25, activation="identity")
        trace = neuron.trace([2.0, 3.0, 0.5])
        self.assertEqual(trace.contributions, (1.0, -3.0, 1.0))
        self.assertEqual(trace.weighted_sum, -0.75)
        self.assertEqual(trace.output, -0.75)

    def test_sigmoid_neuron_output_is_probability_like(self) -> None:
        neuron = ArtificialNeuron((2.0, -1.0), bias=0.1, activation="sigmoid")
        output = neuron.forward([1.0, 0.5])
        self.assertGreater(output, 0.0)
        self.assertLess(output, 1.0)

    def test_relu_and_leaky_relu(self) -> None:
        self.assertEqual(relu(-4.0), 0.0)
        self.assertEqual(relu(4.0), 4.0)
        self.assertAlmostEqual(leaky_relu(-4.0), -0.04)

    def test_activation_derivatives(self) -> None:
        self.assertEqual(activation_derivative("identity", 2.0), 1.0)
        self.assertEqual(activation_derivative("relu", -1.0), 0.0)
        self.assertEqual(activation_derivative("relu", 1.0), 1.0)
        self.assertAlmostEqual(activation_derivative("tanh", 0.0), 1.0)
        self.assertAlmostEqual(activation_derivative("sigmoid", 0.0), 0.25)

    def test_neuron_validates_input_width(self) -> None:
        neuron = ArtificialNeuron((1.0, 2.0))
        with self.assertRaises(ValueError):
            neuron.forward([1.0])

    def test_invalid_activation_and_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ArtificialNeuron((1.0,), activation="unknown")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ArtificialNeuron((math.inf,))
        with self.assertRaises(ValueError):
            leaky_relu(-1.0, negative_slope=-0.1)


if __name__ == "__main__":
    unittest.main()
