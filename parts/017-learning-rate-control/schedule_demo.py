"""Inspect learning-rate schedules for NeuralForge Part 017."""

from __future__ import annotations

from neuralforge.schedules import ReduceLROnPlateau, warmup_cosine_decay


def main() -> None:
    print("NeuralForge Part 017 — learning-rate schedules and control")
    print("warmup + cosine schedule:")
    for step in range(0, 13, 2):
        rate = warmup_cosine_decay(
            0.1,
            step,
            warmup_steps=4,
            total_steps=12,
            start_lr=0.01,
            min_lr=0.005,
        )
        print(f"  step {step:2d}: lr={rate:.6f}")

    print("\nreduce-on-plateau controller:")
    controller = ReduceLROnPlateau(factor=0.5, patience=2, min_lr=0.005, min_delta=0.001)
    rate = 0.05
    for epoch, validation_loss in enumerate([0.80, 0.70, 0.6995, 0.6992, 0.60], start=1):
        rate = controller.update(validation_loss, rate)
        print(f"  epoch {epoch}: val_loss={validation_loss:.4f}, lr={rate:.5f}")


if __name__ == "__main__":
    main()
