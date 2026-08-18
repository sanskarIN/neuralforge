"""Train the same tiny MLP with Adam and gradient clipping."""

from __future__ import annotations

from neuralforge.autograd import Value
from neuralforge.nn import MLP, mean_squared_error
from neuralforge.optim import Adam, clip_grad_norm


def main() -> None:
    features = [[-1.0], [0.0], [1.0]]
    targets = [-1.5, 0.5, 2.5]
    model = MLP(1, [3, 1], seed=11)
    optimizer = Adam(model.parameters(), learning_rate=0.03)

    print("NeuralForge Part 013 — Adam + gradient clipping")

    for step in range(301):
        predictions: list[Value] = []
        for row in features:
            prediction = model(row)
            if not isinstance(prediction, Value):
                raise RuntimeError("expected a scalar model output")
            predictions.append(prediction)

        loss = mean_squared_error(predictions, targets)
        if step in {0, 75, 150, 225, 300}:
            print(f"step={step:3d} loss={loss.data:.8f}")
        if step == 300:
            break

        loss.backward()
        unclipped_norm = clip_grad_norm(model.parameters(), max_norm=5.0)
        optimizer.step()
        optimizer.zero_grad()

        if step == 0:
            print(f"initial gradient norm before clipping: {unclipped_norm:.6f}")


if __name__ == "__main__":
    main()
