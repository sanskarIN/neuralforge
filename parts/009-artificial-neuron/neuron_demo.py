"""Inspect one artificial neuron's forward pass for Part 009."""

from __future__ import annotations

from neuralforge.neuron import ArtificialNeuron, activation_derivative


def main() -> None:
    neuron = ArtificialNeuron(
        weights=(0.8, -0.4, 1.2),
        bias=-0.15,
        activation="relu",
    )
    inputs = [1.5, 2.0, 0.75]
    trace = neuron.trace(inputs)

    print("NeuralForge Part 009 — artificial neuron")
    print(f"inputs: {inputs}")
    print(f"weights: {neuron.weights}")
    print(f"bias: {neuron.bias}")
    print(f"contributions: {trace.contributions}")
    print(f"z (weighted sum): {trace.weighted_sum:.6f}")
    print(f"activation: {neuron.activation}")
    print(f"output: {trace.output:.6f}")
    print(
        "local activation derivative: "
        f"{activation_derivative(neuron.activation, trace.weighted_sum):.6f}"
    )


if __name__ == "__main__":
    main()
