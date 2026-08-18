"""Demonstrate stable loss/output-layer pairings for NeuralForge Part 016."""

from __future__ import annotations

from neuralforge.autograd import Value
from neuralforge.losses import (
    binary_cross_entropy_with_logits,
    categorical_cross_entropy_with_logits,
    recommend_output_design,
)


def main() -> None:
    binary_logit = Value(1.25, label="binary_logit")
    binary_loss = binary_cross_entropy_with_logits([binary_logit], [1])
    binary_loss.backward()

    class_logits = [Value(2.0, label="cat"), Value(0.5, label="dog"), Value(-1.0, label="bird")]
    multiclass_loss = categorical_cross_entropy_with_logits(class_logits, 0)
    multiclass_loss.backward()

    print("NeuralForge Part 016 — loss functions and output-layer design")
    print("binary design:", recommend_output_design("binary classification"))
    print(f"binary BCE-with-logits loss: {binary_loss.data:.6f}")
    print(f"d(loss)/d(binary_logit): {binary_logit.grad:.6f}")
    print()
    print("multiclass design:", recommend_output_design("multiclass", classes=3))
    print(f"categorical CE-with-logits loss: {multiclass_loss.data:.6f}")
    print("class-logit gradients:", [round(value.grad, 6) for value in class_logits])


if __name__ == "__main__":
    main()
