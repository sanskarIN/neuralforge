"""Compare initialization signal propagation for NeuralForge Part 018."""

from __future__ import annotations

import random

from neuralforge.initialization import signal_propagation_profile


def main() -> None:
    rng = random.Random(23)
    batch = [[rng.gauss(0.0, 1.0) for _ in range(32)] for _ in range(96)]

    print("NeuralForge Part 018 — initialization and signal propagation")
    for scheme in ("zeros", "xavier_normal", "he_normal", "lecun_normal"):
        profile = signal_propagation_profile(
            batch,
            [32, 32, 32, 32],
            scheme=scheme,
            activation="relu",
            seed=101,
        )
        variances = ", ".join(f"{value:.5f}" for value in profile.variances)
        print(f"{scheme:15s} variances=[{variances}] ratio={profile.variance_ratio:.5f}")


if __name__ == "__main__":
    main()
