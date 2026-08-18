from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.autograd import Value  # noqa: E402
from neuralforge.nn import (  # noqa: E402
    Layer,
    MLP,
    Neuron,
    binary_cross_entropy_loss,
    mean_squared_error,
    sgd_step,
)


class NeuralNetworkTests(unittest.TestCase):
    def test_neuron_forward_and_parameter_count(self) -> None:
        # Use explicit parameters so the expected value is deterministic.
        neuron = Neuron(
            weights=(Value(0.5), Value(-1.0), Value(2.0)),
            bias=Value(0.25),
            activation="linear",
        )
        output = neuron([2.0, 3.0, 0.5])
        self.assertAlmostEqual(output.data, -0.75)
        self.assertEqual(len(neuron.parameters()), 4)

    def test_layer_and_mlp_output_shapes(self) -> None:
        layer = Layer.random(2, 3, activation="relu")
        outputs = layer([1.0, -2.0])
        self.assertEqual(len(outputs), 3)

        multi_output = MLP(2, [4, 2], seed=7)
        result = multi_output([0.5, -1.0])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        scalar_output = MLP(2, [3, 1], seed=7)
        scalar = scalar_output([0.5, -1.0])
        self.assertIsInstance(scalar, Value)

    def test_mlp_parameter_count(self) -> None:
        model = MLP(2, [3, 1], seed=42)
        # 2->3: 3 * (2 weights + bias) = 9
        # 3->1: 1 * (3 weights + bias) = 4
        self.assertEqual(len(model.parameters()), 13)

    def test_backprop_reaches_all_parameters(self) -> None:
        model = MLP(2, [3, 1], output_activation="sigmoid", seed=3)
        prediction = model([1.0, 0.5])
        self.assertIsInstance(prediction, Value)
        loss = (prediction - 1.0) ** 2
        loss.backward()
        self.assertTrue(any(abs(parameter.grad) > 0.0 for parameter in model.parameters()))
        model.zero_grad()
        self.assertTrue(all(parameter.grad == 0.0 for parameter in model.parameters()))

    def test_mean_squared_error(self) -> None:
        loss = mean_squared_error([Value(1.0), Value(3.0)], [0.0, 1.0])
        self.assertAlmostEqual(loss.data, 2.5)

    def test_binary_cross_entropy_loss(self) -> None:
        probability = Value(0.0).sigmoid()
        loss = binary_cross_entropy_loss([probability], [1])
        self.assertAlmostEqual(loss.data, 0.6931471805599453)
        loss.backward()
        self.assertAlmostEqual(probability.grad, -2.0)

    def test_training_reduces_tiny_regression_loss(self) -> None:
        features = [[-1.0], [0.0], [1.0]]
        targets = [-1.5, 0.5, 2.5]
        model = MLP(
            1,
            [3, 1],
            hidden_activation="tanh",
            output_activation="linear",
            seed=11,
        )

        def batch_loss() -> Value:
            predictions = []
            for row in features:
                prediction = model(row)
                self.assertIsInstance(prediction, Value)
                predictions.append(prediction)
            return mean_squared_error(predictions, targets)

        initial = batch_loss().data
        for _ in range(400):
            loss = batch_loss()
            loss.backward()
            sgd_step(model.parameters(), learning_rate=0.05)
        final = batch_loss().data

        self.assertLess(final, initial * 0.1)

    def test_invalid_network_shapes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MLP(0, [1])
        with self.assertRaises(ValueError):
            MLP(2, [])
        with self.assertRaises(ValueError):
            Neuron.random(0)


if __name__ == "__main__":
    unittest.main()
