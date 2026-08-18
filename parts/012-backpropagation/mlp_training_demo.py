"""Train a tiny multilayer perceptron with scalar backpropagation."""

from __future__ import annotations

from neuralforge.autograd import Value
from neuralforge.nn import MLP, mean_squared_error, sgd_step


def main() -> None:
    features = [[-1.0], [0.0], [1.0]]
    targets = [-1.5, 0.5, 2.5]
    model = MLP(
        1,
        [3, 1],
        hidden_activation="tanh",
        output_activation="linear",
        seed=11,
    )

    print("NeuralForge Part 012 — multilayer backpropagation")
    print(f"trainable parameters: {len(model.parameters())}")

    for step in range(401):
        predictions: list[Value] = []
        for row in features:
            prediction = model(row)
            if not isinstance(prediction, Value):
                raise RuntimeError("expected a scalar model output")
            predictions.append(prediction)

        loss = mean_squared_error(predictions, targets)
        if step in {0, 100, 200, 300, 400}:
            print(f"step={step:3d} loss={loss.data:.8f}")
        if step == 400:
            break

        loss.backward()
        sgd_step(model.parameters(), learning_rate=0.05)

    print("predictions:")
    for row, target in zip(features, targets):
        prediction = model(row)
        if not isinstance(prediction, Value):
            raise RuntimeError("expected a scalar model output")
        print(f"  x={row[0]: .1f} target={target: .3f} prediction={prediction.data: .3f}")


if __name__ == "__main__":
    main()
