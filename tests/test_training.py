from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.schedules import cosine_decay  # noqa: E402
from neuralforge.training import ExperimentConfig, run_regression_experiment, write_experiment_json  # noqa: E402


class TrainingTests(unittest.TestCase):
    FEATURES = [[-1.0], [-0.5], [0.0], [0.5], [1.0]]
    TARGETS = [-1.5, -0.5, 0.5, 1.5, 2.5]

    def test_config_fingerprint_is_stable(self) -> None:
        first = ExperimentConfig(epochs=10, learning_rate=0.01, optimizer="adam", seed=7)
        second = ExperimentConfig(epochs=10, learning_rate=0.01, optimizer="adam", seed=7)
        different = ExperimentConfig(epochs=10, learning_rate=0.01, optimizer="adam", seed=8)
        self.assertEqual(first.fingerprint(), second.fingerprint())
        self.assertNotEqual(first.fingerprint(), different.fingerprint())
        self.assertEqual(len(first.fingerprint()), 16)

    def test_regression_experiment_is_reproducible(self) -> None:
        config = ExperimentConfig(epochs=80, learning_rate=0.03, optimizer="adam", seed=11)
        first = run_regression_experiment(self.FEATURES, self.TARGETS, [4, 1], config=config)
        second = run_regression_experiment(self.FEATURES, self.TARGETS, [4, 1], config=config)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.records, second.records)
        self.assertEqual(first.final_predictions, second.final_predictions)
        self.assertEqual(first.final_parameters, second.final_parameters)

    def test_training_reduces_loss(self) -> None:
        config = ExperimentConfig(epochs=120, learning_rate=0.03, optimizer="adam", seed=5)
        result = run_regression_experiment(self.FEATURES, self.TARGETS, [5, 1], config=config)
        self.assertLess(result.final_loss, result.initial_loss * 0.2)
        self.assertEqual(len(result.records), 120)
        self.assertEqual(result.records[0].epoch, 1)
        self.assertEqual(result.records[-1].epoch, 120)

    def test_schedule_is_recorded(self) -> None:
        config = ExperimentConfig(epochs=5, learning_rate=0.1, optimizer="sgd", seed=2)

        def schedule(step: int) -> float:
            return cosine_decay(0.1, step, total_steps=4, min_lr=0.01)

        result = run_regression_experiment(self.FEATURES, self.TARGETS, [3, 1], config=config, schedule=schedule)
        self.assertAlmostEqual(result.records[0].learning_rate, 0.1)
        self.assertAlmostEqual(result.records[-1].learning_rate, 0.01)

    def test_gradient_clipping_is_recorded(self) -> None:
        config = ExperimentConfig(epochs=3, learning_rate=0.01, optimizer="sgd", seed=3, clip_max_norm=0.05)
        result = run_regression_experiment(self.FEATURES, self.TARGETS, [3, 1], config=config)
        self.assertTrue(all(record.clipped_from_l2 is not None for record in result.records))

    def test_json_record_is_deterministic_and_parseable(self) -> None:
        config = ExperimentConfig(epochs=2, learning_rate=0.01, optimizer="sgd", seed=1)
        result = run_regression_experiment(self.FEATURES, self.TARGETS, [2, 1], config=config)
        with tempfile.TemporaryDirectory() as directory:
            path = write_experiment_json(Path(directory) / "run.json", result)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["fingerprint"], config.fingerprint())
        self.assertEqual(payload["config"]["epochs"], 2)
        self.assertEqual(len(payload["records"]), 2)
        self.assertEqual(len(payload["final_predictions"]), len(self.FEATURES))

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError): ExperimentConfig(epochs=0)
        with self.assertRaises(ValueError): ExperimentConfig(learning_rate=0.0)
        with self.assertRaises(ValueError): ExperimentConfig(optimizer="unknown")  # type: ignore[arg-type]
        with self.assertRaises(TypeError): ExperimentConfig(seed=True)
        with self.assertRaises(ValueError): ExperimentConfig(clip_max_norm=0.0)
        with self.assertRaises(ValueError): run_regression_experiment([], [], [1])
        with self.assertRaises(ValueError): run_regression_experiment([[1.0], [1.0, 2.0]], [1.0, 2.0], [1])
        with self.assertRaises(ValueError): run_regression_experiment([[1.0]], [1.0], [2])


if __name__ == "__main__":
    unittest.main()
