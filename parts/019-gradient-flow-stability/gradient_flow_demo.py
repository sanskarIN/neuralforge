"""Inspect a deep MLP gradient snapshot for NeuralForge Part 019."""

from __future__ import annotations

from neuralforge.gradient_flow import (
    assess_gradient_health,
    gradient_to_parameter_ratio,
    group_gradient_stats,
)
from neuralforge.nn import MLP


def main() -> None:
    model = MLP(3, [8, 8, 8, 1], hidden_activation="tanh", output_activation="linear", seed=19)
    prediction = model([0.5, -1.0, 2.0])
    if not hasattr(prediction, "backward"):
        raise RuntimeError("expected scalar Value output")

    loss = (prediction - 0.75) ** 2
    loss.backward()

    parameters = model.parameters()
    health = assess_gradient_health(parameters, vanishing_l2=1e-12, exploding_max_abs=100.0)

    print("NeuralForge Part 019 — gradient-flow diagnostics")
    print(f"loss: {loss.data:.6f}")
    print(f"status: {health.status}")
    print(f"message: {health.message}")
    print(f"global L2 gradient norm: {health.stats.l2_norm:.6f}")
    print(f"gradient/parameter ratio: {gradient_to_parameter_ratio(parameters):.6f}")
    print("per-layer gradient norms:")
    for layer, stats in sorted(group_gradient_stats(parameters).items()):
        print(
            f"  {layer}: l2={stats.l2_norm:.6f} max_abs={stats.max_abs:.6f} "
            f"zeros={stats.zero_fraction:.1%}"
        )


if __name__ == "__main__":
    main()
