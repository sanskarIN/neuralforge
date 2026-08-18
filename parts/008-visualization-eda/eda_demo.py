"""Generate a compact EDA report and SVG scatter plot for Part 008."""

from __future__ import annotations

from pathlib import Path

from neuralforge.eda import describe, histogram, iqr_outlier_mask, write_scatter_svg


def main() -> None:
    epochs = list(range(1, 11))
    validation_loss = [0.91, 0.74, 0.63, 0.55, 0.49, 0.46, 0.44, 0.43, 0.44, 0.47]

    summary = describe(validation_loss)
    loss_histogram = histogram(validation_loss, bins=4)
    outliers = iqr_outlier_mask(validation_loss)

    output = write_scatter_svg(
        Path("artifacts") / "part-008-training-curve.svg",
        epochs,
        validation_loss,
        title="Validation loss by epoch",
    )

    print("NeuralForge Part 008 — exploratory data analysis")
    print(f"summary: {summary}")
    print(f"histogram edges: {loss_histogram.edges}")
    print(f"histogram counts: {loss_histogram.counts}")
    print(f"IQR outlier flags: {outliers}")
    print(f"wrote SVG: {output}")


if __name__ == "__main__":
    main()
