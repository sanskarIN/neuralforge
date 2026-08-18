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
from neuralforge.normalization import (  # noqa: E402
    RunningMoments,
    batch_normalize,
    layer_normalize,
    normalization_parameters,
    stable_softmax,
)


class NormalizationTests(unittest.TestCase):
    def test_batch_normalization_centers_features(self) -> None:
        result = batch_normalize(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            epsilon=1e-8,
        )
        self.assertAlmostEqual(result.means[0].data, 2.0)
        self.assertAlmostEqual(result.means[1].data, 20.0)
        self.assertAlmostEqual(result.variances[0].data, 2.0 / 3.0)
        self.assertAlmostEqual(result.variances[1].data, 200.0 / 3.0)

        for column in range(2):
            values = [row[column].data for row in result.outputs]
            self.assertAlmostEqual(sum(values) / len(values), 0.0, places=10)
            variance = sum(value * value for value in values) / len(values)
            self.assertAlmostEqual(variance, 1.0, places=6)

    def test_batch_norm_affine_parameters_receive_gradients(self) -> None:
        gamma, beta = normalization_parameters(2)
        result = batch_normalize(
            [[1.0, 2.0], [3.0, 4.0]],
            gamma=gamma,
            beta=beta,
        )
        selected = result.outputs[0][0] + result.outputs[0][1]
        selected.backward()
        self.assertAlmostEqual(beta[0].grad, 1.0)
        self.assertAlmostEqual(beta[1].grad, 1.0)
        self.assertNotEqual(gamma[0].grad, 0.0)
        self.assertNotEqual(gamma[1].grad, 0.0)

    def test_layer_normalization_centers_one_example(self) -> None:
        result = layer_normalize([1.0, 2.0, 3.0], epsilon=1e-8)
        self.assertAlmostEqual(result.mean.data, 2.0)
        self.assertAlmostEqual(result.variance.data, 2.0 / 3.0)
        values = [value.data for value in result.outputs]
        self.assertAlmostEqual(sum(values) / len(values), 0.0, places=10)

    def test_stable_softmax_handles_large_logits(self) -> None:
        logits = [Value(1000.0), Value(1001.0), Value(1002.0)]
        probabilities = stable_softmax(logits)
        self.assertTrue(all(math.isfinite(value.data) for value in probabilities))
        self.assertAlmostEqual(sum(value.data for value in probabilities), 1.0)
        self.assertGreater(probabilities[2].data, probabilities[1].data)
        self.assertGreater(probabilities[1].data, probabilities[0].data)

        loss = -probabilities[2].log()
        loss.backward()
        self.assertTrue(all(math.isfinite(logit.grad) for logit in logits))
        self.assertAlmostEqual(sum(logit.grad for logit in logits), 0.0, places=10)

    def test_running_moments_update_and_evaluation(self) -> None:
        running = RunningMoments(1, momentum=0.5)
        running.update([[1.0], [2.0], [3.0]])
        self.assertAlmostEqual(running.means[0], 2.0)
        self.assertAlmostEqual(running.variances[0], 2.0 / 3.0)

        running.update([[4.0], [6.0]])
        self.assertAlmostEqual(running.means[0], 3.5)
        self.assertAlmostEqual(running.variances[0], 5.0 / 6.0)

        output = running.normalize([3.5])
        self.assertAlmostEqual(output[0].data, 0.0)

    def test_running_moments_requires_training_statistics(self) -> None:
        running = RunningMoments(2)
        with self.assertRaises(RuntimeError):
            running.normalize([1.0, 2.0])

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            batch_normalize([])
        with self.assertRaises(ValueError):
            batch_normalize([[1.0, 2.0], [3.0]])
        with self.assertRaises(ValueError):
            layer_normalize([])
        with self.assertRaises(ValueError):
            stable_softmax([])
        with self.assertRaises(ValueError):
            normalization_parameters(0)
        with self.assertRaises(ValueError):
            RunningMoments(1, momentum=0.0)


if __name__ == "__main__":
    unittest.main()
