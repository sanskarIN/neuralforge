"""Compare output/loss design choices for NeuralForge Part 016."""

from __future__ import annotations

from neuralforge.autograd import Value
from neuralforge.losses import (
    binary_cross_entropy_with_logits,
    huber_loss,
    mean_squared_error,
    multiclass_cross_entropy,
    recommended_output_loss,
)


def main() -> None:
    regression_predictions = [Value(2.2), Value(4.8), Value(20.0)]
    regression_targets = [2.0, 5.0, 6.0]
    mse = mean_squared_error(regression_predictions, regression_targets)
    huber = huber_loss(regression_predictions, regression_targets, delta=1.0)

    binary_logits = [Value(2.0), Value(-1.5), Value(0.25)]
    binary_targets = [1, 0, 1]
    binary_loss = binary_cross_entropy_with_logits(binary_logits, binary_targets)

    class_logits = [
        [Value(1.0), Value(3.0), Value(-1.0)],
        [Value(2.0), Value(0.5), Value(-2.0)],
    ]
    class_loss = multiclass_cross_entropy(class_logits, [1, 0])

    print("NeuralForge Part 016 — loss functions and output-layer design")
    print(f"regression MSE:   {mse.data:.6f}")
    print(f"regression Huber: {huber.data:.6f}")
    print(f"binary BCE logits:{binary_loss.data:.6f}")
    print(f"multiclass CE:    {class_loss.data:.6f}")
    print()
    for task in ("regression", "binary_classification", "multiclass_classification"):
        pairing = recommended_output_loss(task)
        print(f"{task}: output={pairing.output_activation}; loss={pairing.loss}")


if __name__ == "__main__":
    main()
