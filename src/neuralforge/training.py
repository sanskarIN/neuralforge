"""Reproducible training-loop engineering for NeuralForge Part 020."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

from .autograd import Value
from .gradient_flow import gradient_stats, gradient_to_parameter_ratio, require_finite_gradients
from .losses import mean_squared_error
from .nn import MLP, Module
from .optim import Adam, Momentum, Optimizer, RMSProp, SGD, clip_grad_norm

OptimizerName = Literal["sgd", "momentum", "rmsprop", "adam"]
Schedule = Callable[[int], float]
LossBuilder = Callable[[], Value]


@dataclass(frozen=True)
class ExperimentConfig:
    epochs: int = 100
    learning_rate: float = 0.01
    optimizer: OptimizerName = "adam"
    seed: int = 42
    clip_max_norm: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.epochs, int) or isinstance(self.epochs, bool) or self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if not math.isfinite(float(self.learning_rate)) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        if self.optimizer not in {"sgd", "momentum", "rmsprop", "adam"}:
            raise ValueError(f"unsupported optimizer: {self.optimizer!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        if self.clip_max_norm is not None:
            value = float(self.clip_max_norm)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("clip_max_norm must be positive and finite")

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    learning_rate: float
    loss: float
    gradient_l2: float
    gradient_max_abs: float
    gradient_parameter_ratio: float
    clipped_from_l2: float | None


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    fingerprint: str
    records: tuple[EpochRecord, ...]
    final_predictions: tuple[float, ...]
    final_parameters: tuple[float, ...]

    @property
    def initial_loss(self) -> float:
        return self.records[0].loss

    @property
    def final_loss(self) -> float:
        return self.records[-1].loss

    def to_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "fingerprint": self.fingerprint,
            "records": [asdict(record) for record in self.records],
            "final_predictions": list(self.final_predictions),
            "final_parameters": list(self.final_parameters),
        }


def build_optimizer(parameters: Sequence[Value], *, name: OptimizerName, learning_rate: float) -> Optimizer:
    if name == "sgd": return SGD(parameters, learning_rate=learning_rate)
    if name == "momentum": return Momentum(parameters, learning_rate=learning_rate)
    if name == "rmsprop": return RMSProp(parameters, learning_rate=learning_rate)
    if name == "adam": return Adam(parameters, learning_rate=learning_rate)
    raise ValueError(f"unsupported optimizer: {name!r}")


def _set_learning_rate(optimizer: Optimizer, learning_rate: float) -> None:
    rate = float(learning_rate)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("schedule produced an invalid learning rate")
    if not hasattr(optimizer, "learning_rate"):
        raise TypeError("optimizer does not expose a learning_rate attribute")
    optimizer.learning_rate = rate  # type: ignore[attr-defined]


def run_training_loop(
    model: Module,
    optimizer: Optimizer,
    loss_builder: LossBuilder,
    *,
    epochs: int,
    schedule: Schedule | None = None,
    clip_max_norm: float | None = None,
) -> tuple[EpochRecord, ...]:
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    clip_limit = None if clip_max_norm is None else float(clip_max_norm)
    if clip_limit is not None and (not math.isfinite(clip_limit) or clip_limit <= 0.0):
        raise ValueError("clip_max_norm must be positive and finite")

    records: list[EpochRecord] = []
    for epoch in range(epochs):
        if schedule is not None:
            _set_learning_rate(optimizer, schedule(epoch))
        rate = float(getattr(optimizer, "learning_rate"))
        optimizer.zero_grad()
        loss = loss_builder()
        if not math.isfinite(loss.data):
            raise ValueError("training loss became non-finite")
        loss.backward()

        parameters = model.parameters()
        require_finite_gradients(parameters)
        before = gradient_stats(parameters)
        ratio = gradient_to_parameter_ratio(parameters)
        clipped_from: float | None = None
        if clip_limit is not None:
            clipped_from = clip_grad_norm(parameters, clip_limit)
        optimizer.step()
        records.append(
            EpochRecord(
                epoch=epoch + 1,
                learning_rate=rate,
                loss=loss.data,
                gradient_l2=before.l2_norm,
                gradient_max_abs=before.max_abs,
                gradient_parameter_ratio=ratio,
                clipped_from_l2=clipped_from,
            )
        )
    return tuple(records)


def run_regression_experiment(
    features: Sequence[Sequence[float]],
    targets: Sequence[float],
    layer_sizes: Sequence[int],
    *,
    config: ExperimentConfig = ExperimentConfig(),
    schedule: Schedule | None = None,
) -> ExperimentResult:
    if not features or not features[0]:
        raise ValueError("features must be a non-empty rectangular matrix")
    width = len(features[0])
    if any(len(row) != width for row in features):
        raise ValueError("features must be rectangular")
    if len(features) != len(targets) or not targets:
        raise ValueError("features and targets must contain the same non-zero number of rows")
    if not layer_sizes or layer_sizes[-1] != 1:
        raise ValueError("regression layer_sizes must end with one output unit")

    normalized_features = tuple(tuple(float(value) for value in row) for row in features)
    normalized_targets = tuple(float(target) for target in targets)
    if any(not math.isfinite(value) for row in normalized_features for value in row):
        raise ValueError("features must be finite")
    if any(not math.isfinite(target) for target in normalized_targets):
        raise ValueError("targets must be finite")

    model = MLP(width, layer_sizes, hidden_activation="tanh", output_activation="linear", seed=config.seed)
    optimizer = build_optimizer(model.parameters(), name=config.optimizer, learning_rate=config.learning_rate)

    def loss_builder() -> Value:
        predictions: list[Value] = []
        for row in normalized_features:
            prediction = model(row)
            if not isinstance(prediction, Value):
                raise RuntimeError("regression experiment expected scalar model output")
            predictions.append(prediction)
        return mean_squared_error(predictions, normalized_targets)

    records = run_training_loop(
        model,
        optimizer,
        loss_builder,
        epochs=config.epochs,
        schedule=schedule,
        clip_max_norm=config.clip_max_norm,
    )

    final_predictions: list[float] = []
    for row in normalized_features:
        prediction = model(row)
        if not isinstance(prediction, Value):
            raise RuntimeError("regression experiment expected scalar model output")
        final_predictions.append(prediction.data)

    return ExperimentResult(
        config=config,
        fingerprint=config.fingerprint(),
        records=records,
        final_predictions=tuple(final_predictions),
        final_parameters=tuple(parameter.data for parameter in model.parameters()),
    )


def write_experiment_json(path: str | Path, result: ExperimentResult) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
