"""Trace a tiny computational graph and differentiate it for Part 011."""

from __future__ import annotations

from neuralforge.autograd import Value, graph_summary


def main() -> None:
    x1 = Value(2.0, label="x1")
    x2 = Value(-3.0, label="x2")
    w1 = Value(-1.0, label="w1")
    w2 = Value(3.0, label="w2")
    bias = Value(6.881373587, label="b")

    logit = x1 * w1 + x2 * w2 + bias
    output = logit.tanh()
    output.label = "output"
    output.backward()

    print("NeuralForge Part 011 — scalar automatic differentiation")
    print(f"graph summary: {graph_summary(output)}")
    print(f"logit: {logit.data:.6f}")
    print(f"output: {output.data:.6f}")
    print("gradients:")
    for value in (x1, x2, w1, w2, bias):
        print(f"  d(output)/d({value.label}) = {value.grad:.6f}")


if __name__ == "__main__":
    main()
