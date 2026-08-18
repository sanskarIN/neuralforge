"""Inspect batch/layer normalization and stable softmax for Part 015."""

from __future__ import annotations

from neuralforge.autograd import Value
from neuralforge.normalization import (
    RunningMoments,
    batch_normalize,
    layer_normalize,
    normalization_parameters,
    stable_softmax,
)


def main() -> None:
    batch = [
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
    ]
    gamma, beta = normalization_parameters(2)
    normalized = batch_normalize(batch, gamma=gamma, beta=beta)

    print("NeuralForge Part 015 — normalization and stability")
    print(f"batch means: {[value.data for value in normalized.means]}")
    print(f"batch variances: {[value.data for value in normalized.variances]}")
    print("batch-normalized rows:")
    for row in normalized.outputs:
        print(f"  {[round(value.data, 6) for value in row]}")

    layer = layer_normalize([2.0, 4.0, 8.0])
    print(f"layer-normalized: {[round(value.data, 6) for value in layer.outputs]}")

    logits = [Value(1000.0), Value(1001.0), Value(1002.0)]
    probabilities = stable_softmax(logits)
    print(f"stable softmax: {[round(value.data, 6) for value in probabilities]}")
    print(f"probability sum: {sum(value.data for value in probabilities):.12f}")

    running = RunningMoments(2, momentum=0.1)
    running.update(batch)
    evaluation = running.normalize([2.5, 25.0], gamma=gamma, beta=beta)
    print(f"running-stat evaluation: {[round(value.data, 6) for value in evaluation]}")


if __name__ == "__main__":
    main()
