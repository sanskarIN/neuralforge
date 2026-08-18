"""Compare perceptron and logistic regression on a tiny binary dataset."""

from __future__ import annotations

from neuralforge.logistic_regression import train_logistic_regression
from neuralforge.perceptron import train_perceptron


def main() -> None:
    features = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    labels = [0, 1, 1, 1]

    perceptron_result = train_perceptron(features, labels, seed=42)
    logistic_result = train_logistic_regression(
        features,
        labels,
        learning_rate=0.5,
        epochs=2_000,
    )

    print("NeuralForge Part 010 — perceptron vs logistic regression")
    print(f"perceptron mistakes/epoch: {perceptron_result.mistakes_per_epoch}")
    print(
        "logistic loss: "
        f"{logistic_result.losses[0]:.6f} -> {logistic_result.losses[-1]:.6f}"
    )
    print()

    for row, target in zip(features, labels):
        perceptron_prediction = perceptron_result.model.predict(row)
        probability = logistic_result.model.predict_probability(row)
        logistic_prediction = logistic_result.model.predict(row)
        print(
            f"input={row} target={target} "
            f"perceptron={perceptron_prediction} "
            f"logistic_probability={probability:.4f} "
            f"logistic_prediction={logistic_prediction}"
        )


if __name__ == "__main__":
    main()
