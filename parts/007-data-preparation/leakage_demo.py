"""Demonstrate train-only preprocessing state for Part 007."""

from __future__ import annotations

from neuralforge.data_preparation import Standardizer, select_rows, stratified_split_indices


def main() -> None:
    features = [
        [18.0, 120.0],
        [19.0, 125.0],
        [20.0, 130.0],
        [21.0, 135.0],
        [30.0, 180.0],
        [31.0, 185.0],
        [32.0, 190.0],
        [33.0, 195.0],
        [45.0, 260.0],
        [46.0, 270.0],
        [47.0, 280.0],
        [48.0, 290.0],
    ]
    labels = [0] * 6 + [1] * 6

    split = stratified_split_indices(labels, seed=42)
    train_features = select_rows(features, split.train)
    validation_features = select_rows(features, split.validation)
    test_features = select_rows(features, split.test)

    scaler = Standardizer.fit(train_features)

    print("NeuralForge Part 007 — leakage-resistant preparation")
    print(f"train indices: {split.train}")
    print(f"validation indices: {split.validation}")
    print(f"test indices: {split.test}")
    print(f"training-only mean: {scaler.mean}")
    print(f"training-only scale: {scaler.scale}")
    print(f"first transformed train row: {scaler.transform(train_features)[0]}")
    print(f"validation rows transformed with training state: {scaler.transform(validation_features)}")
    print(f"test rows transformed with training state: {scaler.transform(test_features)}")


if __name__ == "__main__":
    main()
