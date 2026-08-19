"""Run and export a reproducible training experiment for NeuralForge Part 020."""

from __future__ import annotations

from neuralforge.schedules import cosine_decay
from neuralforge.training import ExperimentConfig, run_regression_experiment, write_experiment_json


def main() -> None:
    features = [[-1.0], [-0.5], [0.0], [0.5], [1.0]]
    targets = [-1.5, -0.5, 0.5, 1.5, 2.5]
    config = ExperimentConfig(
        epochs=120,
        learning_rate=0.03,
        optimizer="adam",
        seed=20,
        clip_max_norm=5.0,
    )

    def schedule(step: int) -> float:
        return cosine_decay(0.03, step, total_steps=config.epochs - 1, min_lr=0.003)

    result = run_regression_experiment(
        features,
        targets,
        [5, 1],
        config=config,
        schedule=schedule,
    )
    destination = write_experiment_json("artifacts/part-020-experiment.json", result)

    print("NeuralForge Part 020 — reproducible experiment runner")
    print(f"fingerprint: {result.fingerprint}")
    print(f"initial loss: {result.initial_loss:.6f}")
    print(f"final loss: {result.final_loss:.6f}")
    print("final predictions:", [round(value, 4) for value in result.final_predictions])
    print(f"JSON record: {destination}")


if __name__ == "__main__":
    main()
