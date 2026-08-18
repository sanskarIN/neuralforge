from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralforge.training import (  # noqa: E402
    ExperimentConfig,
    config_fingerprint,
    data_fingerprint,
    run_regression_experiment,
)


class TrainingRunnerTests(unittest.TestCase):
    FEATURES = [[-1.0], [0.0], [1.0], [2.0]]
    TARGETS = [-1.0, 1.0, 3.0, 5.0]

    def test_config_fingerprint_is_stable_and_sensitive(self) -> None:
        first = ExperimentConfig(input_size=1, layer_sizes=(1,), epochs=10, seed=7)
        second = ExperimentConfig(input_size=1, layer_sizes=(1,), epochs=10, seed=7)
        changed = ExperimentConfig(input_size=1, layer_sizes=(1,), epochs=11, seed=7)
        self.assertEqual(config_fingerprint(first), config_fingerprint(second))
        self.assertNotEqual(config_fingerprint(first), config_fingerprint(changed))
        self.assertEqual(len(config_fingerprint(first)), 64)

    def test_data_fingerprint_is_stable_and_sensitive(self) -> None:
        first = data_fingerprint(self.FEATURES, self.TARGETS)
        second = data_fingerprint(self.FEATURES, self.TARGETS)
        changed = data_fingerprint(self.FEATURES, [-1.0, 1.0, 3.0, 5.1])
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_linear_regression_experiment_learns(self) -> None:
        config = ExperimentConfig(
            input_size=1,
            layer_sizes=(1,),
            epochs=50,
            learning_rate=0.1,
            optimizer="sgd",
            schedule="constant",
            seed=9,
        )
        result = run_regression_experiment(self.FEATURES, self.TARGETS, config=config)
        self.assertEqual(result.epochs_completed, 50)
        self.assertLess(result.final_train_loss, 1e-4)
        self.assertFalse(result.stopped_early)
        self.assertTrue(all(record.gradient_status != "nonfinite" for record in result.history))

    def test_same_config_and_data_reproduce_history_and_parameters(self) -> None:
        config = ExperimentConfig(
            input_size=1,
            layer_sizes=(3, 1),
            epochs=12,
            learning_rate=0.02,
            optimizer="adam",
            schedule="cosine",
            min_learning_rate=0.005,
            seed=12,
            gradient_clip_norm=5.0,
        )
        first = run_regression_experiment(self.FEATURES, self.TARGETS, config=config)
        second = run_regression_experiment(self.FEATURES, self.TARGETS, config=config)
        self.assertEqual(first.config_fingerprint, second.config_fingerprint)
        self.assertEqual(first.data_fingerprint, second.data_fingerprint)
        self.assertEqual(first.run_fingerprint, second.run_fingerprint)
        self.assertEqual(first.history, second.history)
        self.assertEqual(
            [parameter.data for parameter in first.model.parameters()],
            [parameter.data for parameter in second.model.parameters()],
        )

    def test_cosine_schedule_is_recorded(self) -> None:
        config = ExperimentConfig(
            input_size=1,
            layer_sizes=(1,),
            epochs=5,
            learning_rate=0.1,
            optimizer="sgd",
            schedule="cosine",
            min_learning_rate=0.01,
            seed=3,
        )
        result = run_regression_experiment(self.FEATURES, self.TARGETS, config=config)
        self.assertAlmostEqual(result.history[0].learning_rate, 0.1)
        self.assertAlmostEqual(result.history[-1].learning_rate, 0.01)

    def test_validation_and_early_stopping(self) -> None:
        config = ExperimentConfig(
            input_size=1,
            layer_sizes=(1,),
            epochs=20,
            learning_rate=0.001,
            optimizer="sgd",
            seed=4,
            early_stopping_patience=2,
            early_stopping_min_delta=100.0,
        )
        result = run_regression_experiment(
            self.FEATURES,
            self.TARGETS,
            config=config,
            validation_features=self.FEATURES,
            validation_targets=self.TARGETS,
        )
        self.assertTrue(result.stopped_early)
        self.assertEqual(result.epochs_completed, 3)
        self.assertIsNotNone(result.best_validation_loss)
        self.assertTrue(all(record.validation_loss is not None for record in result.history))

    def test_early_stopping_requires_validation_data(self) -> None:
        config = ExperimentConfig(
            input_size=1,
            layer_sizes=(1,),
            epochs=5,
            early_stopping_patience=2,
        )
        with self.assertRaises(ValueError):
            run_regression_experiment(self.FEATURES, self.TARGETS, config=config)

    def test_invalid_config_and_data_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ExperimentConfig(input_size=0, layer_sizes=(1,))
        with self.assertRaises(ValueError):
            ExperimentConfig(input_size=1, layer_sizes=(2,))
        with self.assertRaises(ValueError):
            ExperimentConfig(
                input_size=1,
                layer_sizes=(1,),
                epochs=4,
                schedule="warmup_cosine",
                warmup_epochs=4,
            )
        config = ExperimentConfig(input_size=1, layer_sizes=(1,), epochs=2)
        with self.assertRaises(ValueError):
            run_regression_experiment([[1.0, 2.0]], [1.0], config=config)
        with self.assertRaises(ValueError):
            run_regression_experiment([[1.0]], [1.0], config=config, validation_features=[[1.0]])


if __name__ == "__main__":
    unittest.main()
