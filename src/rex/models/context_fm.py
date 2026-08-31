"""Context-aware multi-seed FM for robust within-user ranking.

The organizer FM deliberately uses a very small feature set.  This plugin keeps
the same optimizer and loss, adds two inference-safe log fields (hour and
random-exposure state), and averages independently initialized members.  Every
member is saved in the immutable model bundle so prediction is reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rex.data.views import FeatureView, TargetView
from rex.models.official_fm import FM


CONTEXT_FIELDS = (
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "dur_bucket",
    "hour",
    "is_rand",
)


@dataclass
class ContextCategoricalEncoder:
    edges: np.ndarray
    vocabs: list[dict[str, int]]
    unknown: np.ndarray
    offsets: np.ndarray
    dimension: int

    @staticmethod
    def _context(view: FeatureView, name: str) -> np.ndarray:
        key = f"fx__{name}"
        if key not in view.arrays:
            raise ValueError(
                f"context FM requires {key}; rebuild sanitized views with the current bootstrap"
            )
        values = np.asarray(view.arrays[key])
        if values.ndim != 1 or len(values) != view.rows or not np.isfinite(values).all():
            raise ValueError(f"context feature {key} is non-finite or misaligned")
        return values.astype(np.int16, copy=False).astype(str)

    @classmethod
    def fit(cls, view: FeatureView) -> "ContextCategoricalEncoder":
        hour = np.asarray(view.arrays.get("fx__hour", []), dtype=np.int16)
        random_exposure = np.asarray(view.arrays.get("fx__is_rand", []), dtype=np.int16)
        if hour.shape != (view.rows,) or np.any((hour < 0) | (hour > 23)):
            raise ValueError("fx__hour must contain one integer in [0, 23] per row")
        if random_exposure.shape != (view.rows,) or not np.isin(
            random_exposure, (0, 1)
        ).all():
            raise ValueError("fx__is_rand must contain one binary value per row")
        edges = np.quantile(view.arrays["duration_ms"], np.linspace(0, 1, 11)[1:-1])
        raw = cls._raw_fields(view, edges)
        vocabs: list[dict[str, int]] = []
        for column in raw:
            vocab: dict[str, int] = {}
            for value in column:
                key = str(value)
                if key not in vocab:
                    vocab[key] = len(vocab)
            vocabs.append(vocab)
        unknown = np.asarray([len(vocab) for vocab in vocabs], dtype=np.int32)
        dimensions = np.asarray([len(vocab) + 1 for vocab in vocabs], dtype=np.int32)
        offsets = np.cumsum(np.concatenate(([0], dimensions[:-1]))).astype(np.int32)
        return cls(edges, vocabs, unknown, offsets, int(dimensions.sum()))

    @classmethod
    def _raw_fields(cls, view: FeatureView, edges: np.ndarray) -> list[np.ndarray]:
        duration_bucket = np.searchsorted(edges, view.arrays["duration_ms"]).astype(str)
        return [
            view.arrays["user_id"],
            view.arrays["video_id"],
            view.arrays["author_id"],
            view.arrays["tab"],
            duration_bucket,
            cls._context(view, "hour"),
            cls._context(view, "is_rand"),
        ]

    def transform(self, view: FeatureView) -> np.ndarray:
        raw = self._raw_fields(view, self.edges)
        encoded = np.empty((view.rows, len(CONTEXT_FIELDS)), dtype=np.int32)
        for field_index, column in enumerate(raw):
            vocab = self.vocabs[field_index]
            encoded[:, field_index] = np.fromiter(
                (
                    vocab.get(str(value), int(self.unknown[field_index]))
                    for value in column
                ),
                dtype=np.int32,
                count=view.rows,
            ) + self.offsets[field_index]
        return encoded

    def to_json(self) -> dict[str, Any]:
        return {
            "edges": self.edges.tolist(),
            "vocabs": self.vocabs,
            "unknown": self.unknown.tolist(),
            "offsets": self.offsets.tolist(),
            "dimension": self.dimension,
            "fields": list(CONTEXT_FIELDS),
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ContextCategoricalEncoder":
        if tuple(value.get("fields", ())) != CONTEXT_FIELDS:
            raise ValueError("context FM encoder field contract drifted")
        return cls(
            edges=np.asarray(value["edges"], dtype=np.float64),
            vocabs=[
                {str(key): int(index) for key, index in item.items()}
                for item in value["vocabs"]
            ],
            unknown=np.asarray(value["unknown"], dtype=np.int32),
            offsets=np.asarray(value["offsets"], dtype=np.int32),
            dimension=int(value["dimension"]),
        )


class ContextEnsembleFMPlugin:
    """Train 1-7 deterministic FM members and aggregate their raw scores."""

    @staticmethod
    def _member_count(config: dict[str, Any]) -> int:
        count = int(config.get("ensemble_members", 1))
        if not 1 <= count <= 7:
            raise ValueError("ensemble_members must be between 1 and 7")
        return count

    @staticmethod
    def _aggregation(config: dict[str, Any]) -> str:
        aggregation = str(config.get("aggregation", "mean"))
        if aggregation not in {"mean", "median"}:
            raise ValueError("aggregation must be mean or median")
        return aggregation

    @staticmethod
    def _load_model(path: Path, encoder: ContextCategoricalEncoder) -> FM:
        with np.load(path, allow_pickle=False) as saved:
            model = FM(encoder.dimension, saved["V"].shape[1], 0.001, 1e-6, 0)
            model.V = saved["V"]
            model.W = saved["W"]
            model.b = np.float32(saved["b"][0])
            model.t = int(saved["t"][0])
        return model

    def fit(
        self,
        train_features: FeatureView,
        train_targets: TargetView,
        config: dict[str, Any],
        seed: int,
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        encoder = ContextCategoricalEncoder.fit(train_features)
        encoded = encoder.transform(train_features)
        members = self._member_count(config)
        self._aggregation(config)
        epochs = int(config.get("epochs", 7))
        batch_size = int(config.get("batch_size", 8192))
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        histories: list[dict[str, Any]] = []
        member_names: list[str] = []
        for member in range(members):
            member_seed = seed + member
            model = FM(
                encoder.dimension,
                k=int(config.get("k", 16)),
                lr=float(config.get("lr", 0.001)),
                l2=float(config.get("l2", 1e-6)),
                seed=member_seed,
            )
            rng = np.random.default_rng(member_seed)
            losses: list[float] = []
            for _epoch in range(epochs):
                order = rng.permutation(train_features.rows)
                epoch_losses: list[float] = []
                for offset in range(0, len(order), batch_size):
                    batch = order[offset : offset + batch_size]
                    epoch_losses.append(
                        model.step(encoded[batch], train_targets.labels[batch])
                    )
                mean_loss = float(np.mean(epoch_losses))
                if not np.isfinite(mean_loss):
                    raise FloatingPointError("context FM training produced non-finite loss")
                losses.append(mean_loss)
            name = f"model-{member:03d}.npz"
            np.savez_compressed(
                output_dir / name,
                V=model.V,
                W=model.W,
                b=np.asarray([model.b]),
                t=np.asarray([model.t]),
            )
            member_names.append(name)
            histories.append({"member": member, "seed": member_seed, "loss": losses})
        (output_dir / "encoder.json").write_text(
            json.dumps(encoder.to_json(), sort_keys=True), encoding="utf-8"
        )
        (output_dir / "training.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "members": member_names,
                    "histories": histories,
                    "aggregation": self._aggregation(config),
                    "config": config,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return output_dir / member_names[0]

    def predict(
        self,
        model_artifact: Path,
        features: FeatureView,
        config: dict[str, Any],
        output_dir: Path,
    ) -> np.ndarray:
        del output_dir
        root = model_artifact.parent
        encoder = ContextCategoricalEncoder.from_json(
            json.loads((root / "encoder.json").read_text(encoding="utf-8"))
        )
        training = json.loads((root / "training.json").read_text(encoding="utf-8"))
        members = list(training.get("members", []))
        if len(members) != self._member_count(config):
            raise ValueError("context FM model member count differs from its config")
        if training.get("aggregation") != self._aggregation(config):
            raise ValueError("context FM aggregation differs from its immutable bundle")
        encoded = encoder.transform(features)
        scores = np.vstack(
            [self._load_model(root / str(name), encoder).predict(encoded) for name in members]
        )
        combined = (
            np.mean(scores, axis=0)
            if training["aggregation"] == "mean"
            else np.median(scores, axis=0)
        )
        if combined.shape != (features.rows,) or not np.isfinite(combined).all():
            raise FloatingPointError("context FM produced non-finite or misaligned predictions")
        return np.asarray(combined, dtype=np.float64)
