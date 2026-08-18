"""Reproducible training-loop engineering for NeuralForge Part 020."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from .autograd import Value
from .gradient_flow import classify_gradient_health, gradient_statistics
from .losses import mean_squared_error
from .nn import MLP
from .optim import Adam, Momentum, Optimizer, RMSProp, SGD, clip_grad_norm
from .regularization import EarlyStopping
from .reproducibility import set_global_seed
from .schedules import apply_learning_rate, cosine_decay, warmup_cosine_decay

OptimizerName = Literal["sgd", "momentum", "rmsprop", "adam"]
ScheduleName = Literal["constant", "cosine", "warmup_cosine"]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    input_size: int
    layer_sizes: tuple[int, ...]
    epochs: int = 100
    learning_rate: float = 0.01
    optimizer: OptimizerName = "adam"
    schedule: ScheduleName = "constant"
    min_learning_rate: float = 1e-5
    warmup_epochs: int = 5
    hidden_activation: Literal["linear", "tanh", "relu", "sigmoid"] = "tanh"
    seed: int = 42
    gradient_clip_norm: float | None = None
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.input_size, bool) or not isinstance(self.input_size, int) or self.input_size <= 0:
            raise ValueError("input_size must be a positive integer")
        if not self.layer_sizes or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in self.layer_sizes
        ):
            raise ValueError("layer_sizes must contain positive integers")
        if self.layer_sizes[-1] != 1:
            raise ValueError("the Part 020 regression runner requires a scalar output layer")
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if not math.isfinite(float(self.learning_rate)) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and greater than zero")
        if self.optimizer not in ("sgd", "momentum", "rmsprop", "adam"):
            raise ValueError("unsupported optimizer")
        if self.schedule not in ("constant", "cosine", "warmup_cosine"):
            raise ValueError("unsupported schedule")
        if not math.isfinite(float(self.min_learning_rate)) or self.min_learning_rate < 0.0:
            raise ValueError("min_learning_rate must be finite and non-negative")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        if isinstance(self.warmup_epochs, bool) or not isinstance(self.warmup_epochs, int) or self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be a non-negative integer")
        if self.schedule == "warmup_cosine" and not 0 < self.warmup_epochs < self.epochs - 1:
            raise ValueError(
                "warmup_cosine requires 0 < warmup_epochs < epochs - 1 so at least one decay step remains"
            )
        if self.hidden_activation not in ("linear", "tanh", "relu", "sigmoid"):
            raise ValueError("unsupported hidden_activation")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be between 0 and 2**32 - 1")
        if self.gradient_clip_norm is not None:
            value = float(self.gradient_clip_norm)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("gradient_clip_norm must be finite and greater than zero")
        if self.early_stopping_patience is not None:
            if (
                isinstance(self.early_stopping_patience, bool)
                or not isinstance(self.early_stopping_patience, int)
                or self.early_stopping_patience <= 0
            ):
                raise ValueError("early_stopping_patience must be a positive integer")
        if not math.isfinite(float(self.early_stopping_min_delta)) or self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class EpochRecord:
    epoch: int
    learning_rate: float
    train_loss: float
    validation_loss: float | None
    gradient_norm_before_clip: float
    gradient_max_abs: float
    gradient_status: str


@dataclass(slots=True)
class ExperimentResult:
    config: ExperimentConfig
    config_fingerprint: str
    data_fingerprint: str
    run_fingerprint: str
    history: tuple[EpochRecord, ...]
    model: MLP
    stopped_early: bool

    @property
    def epochs_completed(self) -> int:
        return len(self.history)

    @property
    def final_train_loss(self) -> float:
        if not self.history:
            raise RuntimeError("experiment history is empty")
        return self.history[-1].train_loss

    @property
    def best_validation_loss(self) -> float | None:
        values = [record.validation_loss for record in self.history if record.validation_loss is not None]
        return min(values) if values else None


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def config_fingerprint(config: ExperimentConfig) -> str:
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    return _canonical_hash(asdict(config))


def _validate_dataset(
    features: Sequence[Sequence[float | int]],
    targets: Sequence[float | int],
    *,
    input_size: int,
    name: str,
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    rows = tuple(tuple(float(value) for value in row) for row in features)
    target_values = tuple(float(target) for target in targets)
    if not rows:
        raise ValueError(f"{name} features must contain at least one row")
    if len(rows) != len(target_values):
        raise ValueError(f"{name} features and targets must have the same length")
    if any(len(row) != input_size for row in rows):
        raise ValueError(f"every {name} feature row must have width {input_size}")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError(f"{name} features must be finite")
    if any(not math.isfinite(target) for target in target_values):
        raise ValueError(f"{name} targets must be finite")
    return rows, target_values


def data_fingerprint(
    features: Sequence[Sequence[float | int]],
    targets: Sequence[float | int],
) -> str:
    rows = tuple(tuple(float(value) for value in row) for row in features)
    values = tuple(float(target) for target in targets)
    if not rows or len(rows) != len(values):
        raise ValueError("features and targets must be non-empty and have equal length")
    if any(not math.isfinite(value) for row in rows for value in row) or any(
        not math.isfinite(value) for value in values
    ):
        raise ValueError("fingerprinted data must be finite")
    return _canonical_hash({"features": rows, "targets": values})


def _build_optimizer(config: ExperimentConfig, model: MLP) -> Optimizer:
    parameters = model.parameters()
    if config.optimizer == "sgd":
        return SGD(parameters, learning_rate=config.learning_rate)
    if config.optimizer == "momentum":
        return Momentum(parameters, learning_rate=config.learning_rate)
    if config.optimizer == "rmsprop":
        return RMSProp(parameters, learning_rate=config.learning_rate)
    return Adam(parameters, learning_rate=config.learning_rate)


def _learning_rate(config: ExperimentConfig, epoch_index: int) -> float:
    if config.schedule == "constant":
        return config.learning_rate
    total_steps = max(1, config.epochs - 1)
    if config.schedule == "cosine":
        return cosine_decay(
            config.learning_rate,
            epoch_index,
            total_steps=total_steps,
            min_lr=config.min_learning_rate,
        )
    return warmup_cosine_decay(
        config.learning_rate,
        epoch_index,
        warmup_steps=config.warmup_epochs,
        total_steps=total_steps,
        start_lr=config.min_learning_rate,
        min_lr=config.min_learning_rate,
    )


def _predictions(model: MLP, features: tuple[tuple[float, ...], ...]) -> tuple[Value, ...]:
    outputs: list[Value] = []
    for row in features:
        prediction = model(row)
        if not isinstance(prediction, Value):
            raise RuntimeError("regression runner expected one scalar model output")
        outputs.append(prediction)
    return tuple(outputs)


def _evaluate_mse(
    model: MLP,
    features: tuple[tuple[float, ...], ...],
    targets: tuple[float, ...],
) -> float:
    return mean_squared_error(_predictions(model, features), targets).data


def run_regression_experiment(
    train_features: Sequence[Sequence[float | int]],
    train_targets: Sequence[float | int],
    *,
    config: ExperimentConfig,
    validation_features: Sequence[Sequence[float | int]] | None = None,
    validation_targets: Sequence[float | int] | None = None,
) -> ExperimentResult:
    """Run a deterministic full-batch MLP regression experiment."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    train_x, train_y = _validate_dataset(
        train_features,
        train_targets,
        input_size=config.input_size,
        name="training",
    )

    if (validation_features is None) != (validation_targets is None):
        raise ValueError("validation_features and validation_targets must be supplied together")
    validation: tuple[tuple[tuple[float, ...], ...], tuple[float, ...]] | None = None
    if validation_features is not None and validation_targets is not None:
        validation = _validate_dataset(
            validation_features,
            validation_targets,
            input_size=config.input_size,
            name="validation",
        )

    set_global_seed(config.seed)
    model = MLP(
        config.input_size,
        config.layer_sizes,
        hidden_activation=config.hidden_activation,
        output_activation="linear",
        seed=config.seed,
    )
    optimizer = _build_optimizer(config, model)
    early_stopping = (
        EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
        )
        if config.early_stopping_patience is not None
        else None
    )
    if early_stopping is not None and validation is None:
        raise ValueError("early stopping requires validation data")

    records: list[EpochRecord] = []
    stopped_early = False

    for epoch_index in range(config.epochs):
        rate = _learning_rate(config, epoch_index)
        apply_learning_rate(optimizer, rate)
        optimizer.zero_grad()

        loss = mean_squared_error(_predictions(model, train_x), train_y)
        loss.backward()
        stats = gradient_statistics(model.parameters())
        status = classify_gradient_health(stats)
        norm_before_clip = stats.l2_norm

        if config.gradient_clip_norm is not None:
            clip_grad_norm(model.parameters(), config.gradient_clip_norm)
        optimizer.step()

        train_loss = _evaluate_mse(model, train_x, train_y)
        validation_loss = (
            _evaluate_mse(model, validation[0], validation[1])
            if validation is not None
            else None
        )
        records.append(
            EpochRecord(
                epoch=epoch_index + 1,
                learning_rate=rate,
                train_loss=train_loss,
                validation_loss=validation_loss,
                gradient_norm_before_clip=norm_before_clip,
                gradient_max_abs=stats.max_abs,
                gradient_status=status,
            )
        )

        if early_stopping is not None and validation_loss is not None:
            if early_stopping.update(validation_loss):
                stopped_early = True
                break

    configuration_hash = config_fingerprint(config)
    training_data_hash = data_fingerprint(train_x, train_y)
    run_hash = _canonical_hash(
        {
            "config": configuration_hash,
            "training_data": training_data_hash,
            "validation_data": (
                data_fingerprint(validation[0], validation[1]) if validation is not None else None
            ),
        }
    )

    return ExperimentResult(
        config=config,
        config_fingerprint=configuration_hash,
        data_fingerprint=training_data_hash,
        run_fingerprint=run_hash,
        history=tuple(records),
        model=model,
        stopped_early=stopped_early,
    )
