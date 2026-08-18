"""Inspect layerwise gradients for NeuralForge Part 019."""

from __future__ import annotations

from neuralforge.autograd import Value
from neuralforge.gradient_flow import gradient_flow_report, mlp_parameter_groups, relative_update_ratio
from neuralforge.nn import MLP


def main() -> None:
    model = MLP(2, [4, 4, 1], hidden_activation="tanh", seed=31)
    prediction = model([1.0, -0.5])
    if not isinstance(prediction, Value):
        raise RuntimeError("expected a scalar model output")

    loss = (prediction - 0.8) ** 2
    loss.backward()
    groups = mlp_parameter_groups(model)
    report = gradient_flow_report(
        groups,
        vanishing_threshold=1e-10,
        exploding_threshold=10.0,
    )

    print("NeuralForge Part 019 — gradient flow diagnostics")
    print(f"loss={loss.data:.6f}; overall={report.overall_status}")
    for layer in report.layers:
        ratio = relative_update_ratio(groups[layer.name], learning_rate=0.05)
        print(
            f"{layer.name}: status={layer.status:9s} "
            f"l2={layer.stats.l2_norm:.6f} max={layer.stats.max_abs:.6f} "
            f"zero_fraction={layer.stats.zero_fraction:.3f} update_ratio={ratio:.6f}"
        )


if __name__ == "__main__":
    main()
