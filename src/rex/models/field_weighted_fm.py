"""Deterministic field-weighted factorization machine.

FwFM keeps one embedding per categorical value, as a normal FM does, while
learning one scalar for every pair of fields.  Freezing all pair scalars to one
is an explicit equivalence mode used for matched scientific controls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rex.data.views import FeatureView, TargetView
from rex.models.context_fm import (
    FEATURE_AUDIT_FILENAME,
    ContextCategoricalEncoder,
    _atomic_json,
    _atomic_npz,
    categorical_fields_from_config,
    persist_member_predictions,
)
from rex.models.official_fm import sigmoid


TRAINING_FILENAME = "training.json"
ENCODER_FILENAME = "encoder.json"
INTERACTION_REPORT_FILENAME = "field_interactions.json"
MODEL_SCHEMA_VERSION = "1.0"


def _positive_float(config: dict[str, Any], name: str, default: float) -> float:
    value = float(config.get(name, default))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _nonnegative_float(config: dict[str, Any], name: str, default: float) -> float:
    value = float(config.get(name, default))
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


class FieldWeightedFM:
    """NumPy FwFM with deterministic Adam updates and bounded pair weights."""

    def __init__(
        self,
        dimension: int,
        field_count: int,
        k: int,
        lr: float,
        l2_embeddings: float,
        l2_linear: float,
        l2_pairs: float,
        seed: int,
        *,
        learn_field_weights: bool = True,
        pair_lr: float | None = None,
        pair_weight_clip: float = 4.0,
    ) -> None:
        if dimension < 1 or field_count < 2 or k < 1:
            raise ValueError("dimension, k, and at least two fields are required")
        for name, value in (
            ("lr", lr),
            ("pair_lr", lr if pair_lr is None else pair_lr),
            ("pair_weight_clip", pair_weight_clip),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("l2_embeddings", l2_embeddings),
            ("l2_linear", l2_linear),
            ("l2_pairs", l2_pairs),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dimension, k)).astype(np.float32)
        self.W = np.zeros(dimension, dtype=np.float32)
        self.b = np.float32(0.0)
        self.field_weights = np.ones((field_count, field_count), dtype=np.float32)
        np.fill_diagonal(self.field_weights, 0.0)
        self.lr = float(lr)
        self.pair_lr = float(lr if pair_lr is None else pair_lr)
        self.l2_embeddings = float(l2_embeddings)
        self.l2_linear = float(l2_linear)
        self.l2_pairs = float(l2_pairs)
        self.learn_field_weights = bool(learn_field_weights)
        self.pair_weight_clip = float(pair_weight_clip)
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mR = np.zeros_like(self.field_weights)
        self.vR = np.zeros_like(self.field_weights)
        self.t = 0

    @property
    def field_count(self) -> int:
        return int(self.field_weights.shape[0])

    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features)
        if values.ndim != 2 or values.shape[1] != self.field_count:
            raise ValueError(
                f"FwFM expected {self.field_count} encoded fields; observed shape {values.shape}"
            )
        if values.dtype.kind not in "iu":
            raise ValueError("FwFM encoded features must be integers")
        if values.size and (values.min() < 0 or values.max() >= len(self.W)):
            raise ValueError("FwFM encoded feature index is out of bounds")
        return values

    def logits(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = self._validate_features(features)
        embeddings = self.V[values]
        scores = self.b + self.W[values].sum(axis=1)
        for left in range(self.field_count - 1):
            for right in range(left + 1, self.field_count):
                dot = np.sum(embeddings[:, left] * embeddings[:, right], axis=1)
                scores = scores + self.field_weights[left, right] * dot
        return np.asarray(scores, dtype=np.float32), embeddings

    def step(self, features: np.ndarray, targets: np.ndarray) -> float:
        values = self._validate_features(features)
        labels = np.asarray(targets, dtype=np.float32)
        if labels.shape != (len(values),) or not np.isin(labels, (0.0, 1.0)).all():
            raise ValueError("FwFM targets must be one aligned binary vector")
        if not len(labels):
            raise ValueError("FwFM cannot train on an empty batch")
        logits, embeddings = self.logits(values)
        probabilities = sigmoid(logits)
        score_gradient = ((probabilities - labels) / len(labels)).astype(np.float32)
        self.apply_score_gradients(values, score_gradient, embeddings=embeddings)
        return float(
            -np.mean(
                labels * np.log(probabilities + 1e-9)
                + (1.0 - labels) * np.log(1.0 - probabilities + 1e-9)
            )
        )

    def apply_score_gradients(
        self,
        features: np.ndarray,
        score_gradients: np.ndarray,
        *,
        embeddings: np.ndarray | None = None,
    ) -> None:
        values = self._validate_features(features)
        gradient = np.asarray(score_gradients, dtype=np.float32)
        if gradient.shape != (len(values),) or not np.isfinite(gradient).all():
            raise ValueError("FwFM score gradients must be finite and aligned")
        if embeddings is None:
            _, embeddings = self.logits(values)

        gradient_v = np.zeros_like(self.V)
        gradient_w = np.zeros_like(self.W)
        gradient_r = np.zeros_like(self.field_weights)
        np.add.at(gradient_w, values, gradient[:, None])
        for left in range(self.field_count - 1):
            left_values = values[:, left]
            left_embeddings = embeddings[:, left]
            for right in range(left + 1, self.field_count):
                right_values = values[:, right]
                right_embeddings = embeddings[:, right]
                weight = self.field_weights[left, right]
                np.add.at(
                    gradient_v,
                    left_values,
                    gradient[:, None] * weight * right_embeddings,
                )
                np.add.at(
                    gradient_v,
                    right_values,
                    gradient[:, None] * weight * left_embeddings,
                )
                if self.learn_field_weights:
                    dot = np.sum(left_embeddings * right_embeddings, axis=1)
                    gradient_r[left, right] = np.sum(gradient * dot)

        gradient_v += self.l2_embeddings * self.V
        gradient_w += self.l2_linear * self.W
        if self.learn_field_weights:
            upper = np.triu(np.ones_like(self.field_weights, dtype=bool), k=1)
            gradient_r[upper] += self.l2_pairs * self.field_weights[upper]

        self.t += 1
        self._adam(self.V, gradient_v, self.mV, self.vV, self.lr)
        self._adam(self.W, gradient_w, self.mW, self.vW, self.lr)
        if self.learn_field_weights:
            self._adam(
                self.field_weights,
                gradient_r,
                self.mR,
                self.vR,
                self.pair_lr,
            )
            upper_values = np.triu(self.field_weights, k=1)
            upper_values = np.clip(
                upper_values,
                -self.pair_weight_clip,
                self.pair_weight_clip,
            )
            self.field_weights = upper_values + upper_values.T
            self.mR = np.triu(self.mR, k=1)
            self.vR = np.triu(self.vR, k=1)
        self.b -= self.lr * gradient.sum()

    def _adam(
        self,
        parameter: np.ndarray,
        gradient: np.ndarray,
        momentum: np.ndarray,
        variance: np.ndarray,
        learning_rate: float,
    ) -> None:
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        momentum *= beta1
        momentum += (1.0 - beta1) * gradient
        variance *= beta2
        variance += (1.0 - beta2) * (gradient * gradient)
        parameter -= learning_rate * (momentum / (1.0 - beta1**self.t)) / (
            np.sqrt(variance / (1.0 - beta2**self.t)) + epsilon
        )

    def predict(self, features: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
        values = self._validate_features(features)
        if batch_size <= 0:
            raise ValueError("FwFM prediction batch_size must be positive")
        if not len(values):
            return np.empty(0, dtype=np.float32)
        scores = np.concatenate(
            [
                self.logits(values[offset : offset + batch_size])[0]
                for offset in range(0, len(values), batch_size)
            ]
        )
        if not np.isfinite(scores).all():
            raise FloatingPointError("FwFM produced non-finite predictions")
        return scores


class FieldWeightedFMPlugin:
    """Bundle-compatible 1-3 member FwFM plugin for discovery and finalists."""

    @staticmethod
    def _member_count(config: dict[str, Any]) -> int:
        count = int(config.get("ensemble_members", 1))
        if not 1 <= count <= 3:
            raise ValueError("FwFM ensemble_members must be between 1 and 3")
        return count

    @staticmethod
    def _aggregation(config: dict[str, Any]) -> str:
        value = str(config.get("aggregation", "mean"))
        if value not in {"mean", "median"}:
            raise ValueError("FwFM aggregation must be mean or median")
        return value

    @staticmethod
    def _equivalence_mode(config: dict[str, Any]) -> bool:
        value = config.get("fm_equivalence_mode", False)
        if not isinstance(value, bool):
            raise ValueError("fm_equivalence_mode must be boolean")
        return value

    @classmethod
    def _new_model(
        cls,
        encoder: ContextCategoricalEncoder,
        config: dict[str, Any],
        seed: int,
    ) -> FieldWeightedFM:
        lr = _positive_float(config, "lr", 0.001)
        base_l2 = _nonnegative_float(config, "l2", 1e-6)
        return FieldWeightedFM(
            dimension=encoder.dimension,
            field_count=len(encoder.fields),
            k=int(config.get("k", 16)),
            lr=lr,
            l2_embeddings=_nonnegative_float(config, "l2_embeddings", base_l2),
            l2_linear=_nonnegative_float(config, "l2_linear", base_l2),
            l2_pairs=_nonnegative_float(config, "l2_pairs", 1e-5),
            seed=seed,
            learn_field_weights=not cls._equivalence_mode(config),
            pair_lr=_positive_float(config, "pair_lr", lr),
            pair_weight_clip=_positive_float(config, "pair_weight_clip", 4.0),
        )

    @classmethod
    def _load_model(
        cls,
        path: Path,
        encoder: ContextCategoricalEncoder,
        config: dict[str, Any],
    ) -> FieldWeightedFM:
        with np.load(path, allow_pickle=False) as saved:
            required = {"V", "W", "b", "t", "field_weights"}
            if set(saved.files) != required:
                raise ValueError(f"FwFM checkpoint schema is invalid: {sorted(saved.files)}")
            model = cls._new_model(encoder, config, 0)
            if saved["V"].ndim != 2 or saved["V"].shape[0] != encoder.dimension:
                raise ValueError("FwFM checkpoint dimension differs from its encoder")
            if saved["W"].shape != (encoder.dimension,):
                raise ValueError("FwFM checkpoint linear weights differ from its encoder")
            if saved["b"].shape != (1,) or saved["t"].shape != (1,):
                raise ValueError("FwFM checkpoint scalar state is invalid")
            if saved["field_weights"].shape != (len(encoder.fields), len(encoder.fields)):
                raise ValueError("FwFM checkpoint field matrix differs from its encoder")
            arrays = (saved["V"], saved["W"], saved["b"], saved["field_weights"])
            if any(not np.isfinite(value).all() for value in arrays):
                raise FloatingPointError("FwFM checkpoint contains NaN or Inf")
            field_weights = np.asarray(saved["field_weights"], dtype=np.float32)
            if not np.array_equal(field_weights, field_weights.T) or not np.array_equal(
                np.diag(field_weights), np.zeros(len(encoder.fields), dtype=np.float32)
            ):
                raise ValueError("FwFM checkpoint field weights are not symmetric with zero diagonal")
            model.V = np.asarray(saved["V"], dtype=np.float32)
            model.W = np.asarray(saved["W"], dtype=np.float32)
            model.b = np.float32(saved["b"][0])
            model.t = int(saved["t"][0])
            model.field_weights = field_weights
        return model

    @staticmethod
    def _interaction_report(
        fields: tuple[str, ...],
        member_weights: list[np.ndarray],
        *,
        equivalence_mode: bool,
    ) -> dict[str, Any]:
        mean_weights = np.mean(np.stack(member_weights), axis=0)
        interactions = [
            {
                "left": fields[left],
                "right": fields[right],
                "mean_weight": float(mean_weights[left, right]),
                "member_weights": [float(item[left, right]) for item in member_weights],
                "absolute_deviation_from_fm": float(abs(mean_weights[left, right] - 1.0)),
            }
            for left in range(len(fields) - 1)
            for right in range(left + 1, len(fields))
        ]
        interactions.sort(
            key=lambda item: (-float(item["absolute_deviation_from_fm"]), item["left"], item["right"])
        )
        return {
            "schema_version": "1.0",
            "equivalence_mode": equivalence_mode,
            "fields": list(fields),
            "interactions": interactions,
        }

    def fit(
        self,
        train_features: FeatureView,
        train_targets: TargetView,
        config: dict[str, Any],
        seed: int,
        output_dir: Path,
    ) -> Path:
        if train_features.rows != len(train_targets.labels):
            raise ValueError("FwFM train features and targets are misaligned")
        if train_features.rows < 1:
            raise ValueError("FwFM cannot fit an empty feature view")
        fields = categorical_fields_from_config(config)
        if len(fields) < 2:
            raise ValueError("FwFM requires at least two categorical fields")
        encoder = ContextCategoricalEncoder.fit(train_features, fields)
        encoded = encoder.transform(train_features)
        members = self._member_count(config)
        aggregation = self._aggregation(config)
        epochs = int(config.get("epochs", 7))
        batch_size = int(config.get("batch_size", 8192))
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("FwFM epochs and batch_size must be positive")
        output_dir.mkdir(parents=True, exist_ok=True)

        histories: list[dict[str, Any]] = []
        member_names: list[str] = []
        member_weights: list[np.ndarray] = []
        for member in range(members):
            member_seed = seed + member
            model = self._new_model(encoder, config, member_seed)
            rng = np.random.default_rng(member_seed)
            losses: list[float] = []
            for _epoch in range(epochs):
                order = rng.permutation(train_features.rows)
                batch_losses: list[float] = []
                for offset in range(0, len(order), batch_size):
                    batch = order[offset : offset + batch_size]
                    batch_losses.append(model.step(encoded[batch], train_targets.labels[batch]))
                mean_loss = float(np.mean(batch_losses))
                if not np.isfinite(mean_loss):
                    raise FloatingPointError("FwFM training produced non-finite loss")
                losses.append(mean_loss)
            name = f"model-{member:03d}.npz"
            _atomic_npz(
                output_dir / name,
                V=model.V,
                W=model.W,
                b=np.asarray([model.b]),
                t=np.asarray([model.t]),
                field_weights=model.field_weights,
            )
            member_names.append(name)
            member_weights.append(model.field_weights.copy())
            histories.append({"member": member, "seed": member_seed, "loss": losses})

        _atomic_json(output_dir / ENCODER_FILENAME, encoder.to_json())
        _atomic_json(output_dir / FEATURE_AUDIT_FILENAME, encoder.audit(train_features))
        _atomic_json(
            output_dir / INTERACTION_REPORT_FILENAME,
            self._interaction_report(
                fields,
                member_weights,
                equivalence_mode=self._equivalence_mode(config),
            ),
        )
        _atomic_json(
            output_dir / TRAINING_FILENAME,
            {
                "schema_version": MODEL_SCHEMA_VERSION,
                "members": member_names,
                "histories": histories,
                "aggregation": aggregation,
                "categorical_fields": list(fields),
                "equivalence_mode": self._equivalence_mode(config),
                "feature_audit": FEATURE_AUDIT_FILENAME,
                "field_interactions": INTERACTION_REPORT_FILENAME,
                "config": config,
            },
        )
        return output_dir / member_names[0]

    def predict(
        self,
        model_artifact: Path,
        features: FeatureView,
        config: dict[str, Any],
        output_dir: Path,
    ) -> np.ndarray:
        root = model_artifact.parent
        encoder = ContextCategoricalEncoder.from_json(
            json.loads((root / ENCODER_FILENAME).read_text(encoding="utf-8"))
        )
        training = json.loads((root / TRAINING_FILENAME).read_text(encoding="utf-8"))
        if training.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise ValueError("unsupported FwFM training schema")
        members = [str(item) for item in training.get("members", [])]
        if len(members) != self._member_count(config):
            raise ValueError("FwFM member count differs from its immutable bundle")
        aggregation = self._aggregation(config)
        if training.get("aggregation") != aggregation:
            raise ValueError("FwFM aggregation differs from its immutable bundle")
        if tuple(training.get("categorical_fields", ())) != encoder.fields:
            raise ValueError("FwFM training and encoder field contracts differ")
        if encoder.fields != categorical_fields_from_config(config):
            raise ValueError("FwFM configured fields differ from its immutable bundle")
        if bool(training.get("equivalence_mode")) != self._equivalence_mode(config):
            raise ValueError("FwFM equivalence mode differs from its immutable bundle")
        encoded = encoder.transform(features)
        scores = np.vstack(
            [self._load_model(root / name, encoder, config).predict(encoded) for name in members]
        )
        combined = persist_member_predictions(
            output_dir,
            scores,
            aggregation=aggregation,
            member_names=members,
            feature_audit=encoder.audit(features),
            plugin=f"{type(self).__module__}:{type(self).__name__}",
        )
        if combined.shape != (features.rows,) or not np.isfinite(combined).all():
            raise FloatingPointError("FwFM produced non-finite or misaligned predictions")
        return combined
