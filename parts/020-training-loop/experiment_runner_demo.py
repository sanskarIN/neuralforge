"""Run a reproducible tiny experiment for NeuralForge Part 020."""

from __future__ import annotations

from neuralforge.training import ExperimentConfig, run_regression_experiment


def main() -> None:
    features = [[-1.0], [0.0], [1.0], [2.0]]
    targets = [-1.0, 1.0, 3.0, 5.0]
    config = ExperimentConfig(
        input_size=1,
        layer_sizes=(4, 1),
        epochs=40,
        learning_rate=0.03,
        optimizer="adam",
        schedule="cosine",
        min_learning_rate=0.005,
        gradient_clip_norm=5.0,
        seed=2026,
    )

    result = run_regression_experiment(features, targets, config=config)

    print("NeuralForge Part 020 — reproducible experiment runner")
    print(f"config fingerprint: {result.config_fingerprint}")
    print(f"data fingerprint:   {result.data_fingerprint}")
    print(f"run fingerprint:    {result.run_fingerprint}")
    print(f"epochs completed:   {result.epochs_completed}")
    print(f"final train loss:   {result.final_train_loss:.8f}")
    print("last five epochs:")
    for record in result.history[-5:]:
        print(
            f"  epoch={record.epoch:02d} lr={record.learning_rate:.6f} "
            f"loss={record.train_loss:.8f} grad_l2={record.gradient_norm_before_clip:.6f} "
            f"status={record.gradient_status}"
        )


if __name__ == "__main__":
    main()
