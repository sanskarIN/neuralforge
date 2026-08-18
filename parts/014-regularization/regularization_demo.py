"""Inspect L2 regularization, dropout, and early stopping for Part 014."""

from __future__ import annotations

from neuralforge.autograd import Value
from neuralforge.regularization import (
    EarlyStopping,
    generalization_gap,
    inverted_dropout,
    l2_penalty,
    parameter_l2_norm,
)


def main() -> None:
    parameters = [Value(1.5), Value(-0.75), Value(0.25)]
    activations = [Value(float(index + 1)) for index in range(8)]

    penalty = l2_penalty(parameters, strength=0.1)
    dropout = inverted_dropout(
        activations,
        drop_probability=0.25,
        training=True,
        seed=42,
    )

    print("NeuralForge Part 014 — regularization and generalization")
    print(f"parameter L2 norm: {parameter_l2_norm(parameters):.6f}")
    print(f"L2 penalty: {penalty.data:.6f}")
    print(f"dropout kept mask: {dropout.kept}")
    print(f"dropout scale: {dropout.scale:.6f}")
    print(f"dropout outputs: {[value.data for value in dropout.outputs]}")
    print(f"example generalization gap: {generalization_gap(0.18, 0.27):.6f}")

    validation_losses = [0.62, 0.55, 0.51, 0.505, 0.504, 0.506]
    stopper = EarlyStopping(patience=2, min_delta=0.01)
    print("validation monitoring:")
    for epoch, loss in enumerate(validation_losses, start=1):
        should_stop = stopper.update(loss)
        print(
            f"  epoch={epoch} validation_loss={loss:.3f} "
            f"best={stopper.best:.3f} bad_epochs={stopper.bad_epochs} "
            f"stop={should_stop}"
        )
        if should_stop:
            break


if __name__ == "__main__":
    main()
