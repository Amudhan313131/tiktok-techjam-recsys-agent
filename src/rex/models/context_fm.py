"""Context-aware multi-seed FM for robust within-user ranking.

The organizer FM deliberately uses a very small feature set.  This plugin keeps
the same optimizer and loss, adds two inference-safe log fields (hour and
random-exposure state), and averages independently initialized members.  Every
member is saved in the immutable model bundle so prediction is reproducible.
"""

from __future__ import annotations

import json
import os
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

MEMBER_PREDICTIONS_FILENAME = "member_predictions.npz"
PREDICTION_EVIDENCE_FILENAME = "prediction_evidence.json"
FEATURE_AUDIT_FILENAME = "feature_audit.json"
MAX_CATEGORICAL_FIELDS = 32


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def categorical_fields_from_config(config: dict[str, Any]) -> tuple[str, ...]:
    """Return the configured field contract while preserving the E15 default."""

    configured = config.get("categorical_fields")
    if configured is None:
        return CONTEXT_FIELDS
    if not isinstance(configured, (list, tuple)) or not configured:
        raise ValueError("categorical_fields must be a non-empty list of field names")
    fields = tuple(str(item).strip() for item in configured)
    if any(not item for item in fields):
        raise ValueError("categorical_fields cannot contain an empty field name")
    if len(set(fields)) != len(fields):
        raise ValueError("categorical_fields must not contain duplicates")
    if len(fields) > MAX_CATEGORICAL_FIELDS:
        raise ValueError(
            f"categorical_fields cannot contain more than {MAX_CATEGORICAL_FIELDS} fields"
        )
    return fields


def reconstruct_member_predictions(
    members: np.ndarray,
    aggregation: str,
) -> np.ndarray:
    """Reconstruct an ensemble score from a persisted member-by-row matrix."""

    values = np.asarray(members)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("member predictions must have shape (members, rows)")
    if values.dtype.kind not in "biuf":
        raise ValueError("member predictions must be numeric")
    if not np.isfinite(values).all():
        raise FloatingPointError("member predictions contain NaN or Inf")
    if aggregation == "mean":
        result = np.mean(values, axis=0)
    elif aggregation == "median":
        result = np.median(values, axis=0)
    else:
        raise ValueError("aggregation must be mean or median")
    return np.asarray(result, dtype=np.float64)


def persist_member_predictions(
    output_dir: Path,
    members: np.ndarray,
    *,
    aggregation: str,
    member_names: list[str],
    feature_audit: dict[str, Any],
    plugin: str,
) -> np.ndarray:
    """Persist member scores plus enough evidence to reconstruct the aggregate."""

    matrix = np.asarray(members)
    if matrix.ndim != 2 or matrix.shape[0] != len(member_names):
        raise ValueError("member prediction names and matrix disagree")
    aggregate = reconstruct_member_predictions(matrix, aggregation)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        output_dir / MEMBER_PREDICTIONS_FILENAME,
        members=matrix,
        aggregate=aggregate,
    )
    _atomic_json(
        output_dir / PREDICTION_EVIDENCE_FILENAME,
        {
            "schema_version": "1.0",
            "plugin": plugin,
            "aggregation": aggregation,
            "member_names": member_names,
            "member_count": int(matrix.shape[0]),
            "rows": int(matrix.shape[1]),
            "member_predictions": MEMBER_PREDICTIONS_FILENAME,
            "feature_audit": feature_audit,
        },
    )
    return aggregate


def load_member_predictions(output_dir: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and verify persisted prediction evidence from a model prediction call."""

    root = Path(output_dir)
    manifest = json.loads((root / PREDICTION_EVIDENCE_FILENAME).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0":
        raise ValueError("unsupported member-prediction evidence schema")
    archive = (root / str(manifest["member_predictions"])).resolve()
    try:
        archive.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("member-prediction archive escapes its evidence directory") from error
    with np.load(archive, allow_pickle=False) as payload:
        if set(payload.files) != {"members", "aggregate"}:
            raise ValueError("member-prediction archive has an invalid schema")
        members = np.asarray(payload["members"])
        aggregate = np.asarray(payload["aggregate"], dtype=np.float64)
    expected = reconstruct_member_predictions(members, str(manifest["aggregation"]))
    if aggregate.shape != expected.shape or not np.array_equal(aggregate, expected):
        raise ValueError("persisted aggregate does not reconstruct from member predictions")
    if members.shape != (int(manifest["member_count"]), int(manifest["rows"])):
        raise ValueError("member-prediction archive disagrees with its evidence manifest")
    if len(manifest.get("member_names", [])) != members.shape[0]:
        raise ValueError("member names disagree with persisted member predictions")
    return members, aggregate, manifest


@dataclass
class ContextCategoricalEncoder:
    edges: np.ndarray
    vocabs: list[dict[str, int]]
    unknown: np.ndarray
    offsets: np.ndarray
    dimension: int
    fields: tuple[str, ...] = CONTEXT_FIELDS

    def __post_init__(self) -> None:
        count = len(self.fields)
        if not self.fields or len(set(self.fields)) != count:
            raise ValueError("categorical encoder fields must be non-empty and unique")
        if len(self.vocabs) != count or self.unknown.shape != (count,):
            raise ValueError("categorical encoder vocabulary contract is inconsistent")
        if self.offsets.shape != (count,):
            raise ValueError("categorical encoder offset contract is inconsistent")
        dimensions: list[int] = []
        for index, vocab in enumerate(self.vocabs):
            if sorted(vocab.values()) != list(range(len(vocab))):
                raise ValueError("categorical encoder vocabulary indices are not canonical")
            if int(self.unknown[index]) != len(vocab):
                raise ValueError("categorical encoder unknown index is invalid")
            dimensions.append(len(vocab) + 1)
        expected_offsets = np.cumsum(
            np.concatenate(([0], np.asarray(dimensions[:-1], dtype=np.int32)))
        ).astype(np.int32)
        if not np.array_equal(self.offsets, expected_offsets):
            raise ValueError("categorical encoder offsets are invalid")
        if self.dimension != sum(dimensions):
            raise ValueError("categorical encoder dimension is invalid")
        if self.edges.ndim != 1 or not np.isfinite(self.edges).all():
            raise ValueError("categorical encoder duration edges are invalid")

    @staticmethod
    def _column(view: FeatureView, name: str) -> np.ndarray:
        key = name if name in view.arrays else f"fx__{name}"
        if key not in view.arrays:
            raise ValueError(
                f"categorical model requires {key}; rebuild the sanitized feature view"
            )
        values = np.asarray(view.arrays[key])
        finite = values.dtype.kind not in "biufc" or np.isfinite(values).all()
        if values.ndim != 1 or len(values) != view.rows or not finite:
            raise ValueError(f"categorical feature {key} is non-finite or misaligned")
        return values

    @classmethod
    def fit(
        cls,
        view: FeatureView,
        fields: tuple[str, ...] = CONTEXT_FIELDS,
    ) -> "ContextCategoricalEncoder":
        fields = tuple(fields)
        if not fields or len(set(fields)) != len(fields):
            raise ValueError("categorical encoder fields must be non-empty and unique")
        if view.rows < 1:
            raise ValueError("categorical encoder cannot fit an empty feature view")
        if "hour" in fields:
            hour = np.asarray(cls._column(view, "hour"), dtype=np.int16)
            if hour.shape != (view.rows,) or np.any((hour < 0) | (hour > 23)):
                raise ValueError("fx__hour must contain one integer in [0, 23] per row")
        if "is_rand" in fields:
            random_exposure = np.asarray(cls._column(view, "is_rand"), dtype=np.int16)
            if random_exposure.shape != (view.rows,) or not np.isin(
                random_exposure, (0, 1)
            ).all():
                raise ValueError("fx__is_rand must contain one binary value per row")
        edges = np.quantile(view.arrays["duration_ms"], np.linspace(0, 1, 11)[1:-1])
        raw = cls._raw_fields(view, edges, fields)
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
        return cls(edges, vocabs, unknown, offsets, int(dimensions.sum()), fields)

    @classmethod
    def _raw_fields(
        cls,
        view: FeatureView,
        edges: np.ndarray,
        fields: tuple[str, ...],
    ) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        for field in fields:
            if field == "dur_bucket":
                values = np.searchsorted(edges, view.arrays["duration_ms"])
            else:
                values = cls._column(view, field)
            result.append(np.asarray(values).astype(str))
        return result

    def transform(self, view: FeatureView) -> np.ndarray:
        raw = self._raw_fields(view, self.edges, self.fields)
        encoded = np.empty((view.rows, len(self.fields)), dtype=np.int32)
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

    def audit(self, view: FeatureView) -> dict[str, Any]:
        """Describe constants and unknown coverage without reading any labels."""

        raw = self._raw_fields(view, self.edges, self.fields)
        items: list[dict[str, Any]] = []
        for index, (name, values) in enumerate(zip(self.fields, raw, strict=True)):
            unique = np.unique(values)
            unknown_count = int(
                sum(str(value) not in self.vocabs[index] for value in values)
            )
            item: dict[str, Any] = {
                "field": name,
                "unique_count": int(len(unique)),
                "constant": bool(len(unique) <= 1),
                "unknown_count": unknown_count,
                "unknown_rate": float(unknown_count / view.rows) if view.rows else 0.0,
            }
            if len(unique) == 1:
                item["constant_value"] = str(unique[0])
            items.append(item)
        return {"schema_version": "1.0", "rows": view.rows, "fields": items}

    def to_json(self) -> dict[str, Any]:
        return {
            "edges": self.edges.tolist(),
            "vocabs": self.vocabs,
            "unknown": self.unknown.tolist(),
            "offsets": self.offsets.tolist(),
            "dimension": self.dimension,
            "fields": list(self.fields),
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ContextCategoricalEncoder":
        fields = tuple(str(item) for item in value.get("fields", CONTEXT_FIELDS))
        if not fields or len(set(fields)) != len(fields):
            raise ValueError("context FM encoder field contract drifted")
        vocabs = [
            {str(key): int(index) for key, index in item.items()}
            for item in value["vocabs"]
        ]
        if len(vocabs) != len(fields):
            raise ValueError("context FM encoder vocabulary and field counts differ")
        return cls(
            edges=np.asarray(value["edges"], dtype=np.float64),
            vocabs=vocabs,
            unknown=np.asarray(value["unknown"], dtype=np.int32),
            offsets=np.asarray(value["offsets"], dtype=np.int32),
            dimension=int(value["dimension"]),
            fields=fields,
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
        encoder = ContextCategoricalEncoder.fit(
            train_features,
            categorical_fields_from_config(config),
        )
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
        _atomic_json(output_dir / "encoder.json", encoder.to_json())
        _atomic_json(output_dir / FEATURE_AUDIT_FILENAME, encoder.audit(train_features))
        _atomic_json(
            output_dir / "training.json",
            {
                "schema_version": "1.1",
                "members": member_names,
                "histories": histories,
                "aggregation": self._aggregation(config),
                "config": config,
                "feature_audit": FEATURE_AUDIT_FILENAME,
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
            json.loads((root / "encoder.json").read_text(encoding="utf-8"))
        )
        training = json.loads((root / "training.json").read_text(encoding="utf-8"))
        members = list(training.get("members", []))
        if len(members) != self._member_count(config):
            raise ValueError("context FM model member count differs from its config")
        if training.get("aggregation") != self._aggregation(config):
            raise ValueError("context FM aggregation differs from its immutable bundle")
        if encoder.fields != categorical_fields_from_config(config):
            raise ValueError("context FM categorical fields differ from its immutable bundle")
        encoded = encoder.transform(features)
        scores = np.vstack(
            [self._load_model(root / str(name), encoder).predict(encoded) for name in members]
        )
        combined = persist_member_predictions(
            output_dir,
            scores,
            aggregation=str(training["aggregation"]),
            member_names=[str(name) for name in members],
            feature_audit=encoder.audit(features),
            plugin=f"{type(self).__module__}:{type(self).__name__}",
        )
        if combined.shape != (features.rows,) or not np.isfinite(combined).all():
            raise FloatingPointError("context FM produced non-finite or misaligned predictions")
        return np.asarray(combined, dtype=np.float64)
