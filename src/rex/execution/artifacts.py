"""Immutable artifact creation, hashing, and prediction validation."""

from __future__ import annotations

import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from rex.contracts import ArtifactRef
from rex.data.manifest import sha256_file
from rex.data.views import FeatureView, load_feature_view


class ArtifactError(RuntimeError):
    pass


def atomic_write_json(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


def artifact_ref(path: str | Path, kind: str, artifact_id: str | None = None) -> ArtifactRef:
    candidate = Path(path).resolve()
    if not candidate.is_file():
        raise ArtifactError(f"artifact missing: {candidate}")
    digest = sha256_file(candidate)
    return ArtifactRef(
        artifact_id=artifact_id or f"{kind}-{digest[:16]}",
        kind=kind,
        path=str(candidate),
        sha256=digest,
        size_bytes=candidate.stat().st_size,
    )


def write_prediction_artifact(
    path: str | Path,
    features: FeatureView | str | Path,
    scores: np.ndarray,
) -> Path:
    view = features if isinstance(features, FeatureView) else load_feature_view(features)
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) != view.rows:
        raise ArtifactError(f"prediction length mismatch: expected {view.rows}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ArtifactError("predictions contain NaN or Inf")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            row_id=view.arrays["row_id"],
            user_id=view.arrays["user_id"],
            video_id=view.arrays["video_id"],
            score=values,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


def load_prediction_artifact(
    path: str | Path,
    expected_features: FeatureView | str | Path | None = None,
) -> dict[str, np.ndarray]:
    candidate = Path(path)
    with np.load(candidate, allow_pickle=False) as payload:
        if set(payload.files) != {"row_id", "user_id", "video_id", "score"}:
            raise ArtifactError(f"invalid prediction columns: {payload.files}")
        arrays = {name: payload[name] for name in payload.files}
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ArtifactError("prediction arrays have inconsistent lengths")
    scores = np.asarray(arrays["score"], dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ArtifactError("prediction scores must be a finite one-dimensional array")
    expected_ids = np.arange(len(scores), dtype=np.int64)
    if not np.array_equal(arrays["row_id"], expected_ids):
        raise ArtifactError("prediction row_id must be contiguous and zero-based")
    if expected_features is not None:
        view = (
            expected_features
            if isinstance(expected_features, FeatureView)
            else load_feature_view(expected_features)
        )
        for name in ("row_id", "user_id", "video_id"):
            if not np.array_equal(arrays[name], view.arrays[name]):
                raise ArtifactError(f"prediction alignment mismatch for {name}")
    return arrays


def validate_metric_mapping(metrics: dict[str, Any]) -> None:
    for name in ("GAUC", "nDCG@5", "primary"):
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ArtifactError(f"metric {name} is missing or non-finite")
