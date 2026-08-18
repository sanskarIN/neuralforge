"""Explore descriptive statistics and bootstrap uncertainty for Part 006."""

from __future__ import annotations

from neuralforge.statistics import (
    bootstrap_mean_interval,
    correlation,
    mean,
    standard_deviation,
)


def main() -> None:
    validation_accuracy = [0.81, 0.83, 0.82, 0.85, 0.84, 0.80, 0.86, 0.83]
    training_minutes = [18, 19, 18, 21, 20, 17, 22, 19]

    interval = bootstrap_mean_interval(
        validation_accuracy,
        confidence=0.95,
        resamples=2_000,
        seed=42,
    )

    print("NeuralForge Part 006 — probability and statistics")
    print(f"mean accuracy: {mean(validation_accuracy):.4f}")
    print(f"accuracy std: {standard_deviation(validation_accuracy):.4f}")
    print(
        f"bootstrap 95% interval: [{interval.lower:.4f}, {interval.upper:.4f}]"
    )
    print(
        "accuracy/training-time correlation: "
        f"{correlation(validation_accuracy, training_minutes):.4f}"
    )


if __name__ == "__main__":
    main()
