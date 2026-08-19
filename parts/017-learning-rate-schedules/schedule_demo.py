"""Demonstrate learning-rate schedules for NeuralForge Part 017."""

from __future__ import annotations

from neuralforge.schedules import ReduceLROnPlateau, warmup_cosine_decay


def main() -> None:
    print("NeuralForge Part 017 — learning-rate schedules")
    print("warmup + cosine schedule:")
    for step in range(12):
        lr = warmup_cosine_decay(
            0.1,
            step,
            warmup_steps=3,
            total_steps=11,
            min_lr=0.01,
        )
        print(f"  step={step:02d} lr={lr:.6f}")

    scheduler = ReduceLROnPlateau(0.1, factor=0.5, patience=2, min_lr=0.0125)
    validation_losses = [1.00, 0.82, 0.83, 0.84, 0.72, 0.73, 0.74]
    print("\nplateau controller:")
    for epoch, loss in enumerate(validation_losses, start=1):
        update = scheduler.update(loss)
        print(
            f"  epoch={epoch:02d} val_loss={loss:.3f} "
            f"lr={update.learning_rate:.5f} reduced={update.reduced}"
        )


if __name__ == "__main__":
    main()
