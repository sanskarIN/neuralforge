"""Train a single dependency-free logistic neuron on the OR truth table."""

from __future__ import annotations

from neuralforge import set_global_seed
from neuralforge.foundations import train_logistic_neuron


def main() -> None:
    set_global_seed(42)

    features = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    targets = [0, 1, 1, 1]

    result = train_logistic_neuron(
        features,
        targets,
        learning_rate=0.5,
        epochs=2_000,
    )

    print("NeuralForge Part 001 — OR-gate logistic neuron")
    print(f"weights: {result.neuron.weights}")
    print(f"bias: {result.neuron.bias:.6f}")
    print(f"loss: {result.losses[0]:.6f} -> {result.losses[-1]:.6f}")
    print()

    for row, target in zip(features, targets):
        probability = result.neuron.predict_proba(row)
        prediction = result.neuron.predict(row)
        print(
            f"input={row} target={target} "
            f"probability={probability:.4f} prediction={prediction}"
        )


if __name__ == "__main__":
    main()
