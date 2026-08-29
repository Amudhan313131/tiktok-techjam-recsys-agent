"""Typed access to sanitized feature and target views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rex.data.manifest import sha256_file


FEATURE_COLUMNS = (
    "row_id",
    "date",
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_ms",
)
ENGINEERED_PREFIX = "fx__"


@dataclass(frozen=True)
class FeatureView:
    path: Path
    arrays: dict[str, np.ndarray]
    sha256: str

    @property
    def rows(self) -> int:
        return len(self.arrays["row_id"])


@dataclass(frozen=True)
class TargetView:
    path: Path
    labels: np.ndarray
    sha256: str


def load_feature_view(path: str | Path) -> FeatureView:
    candidate = Path(path)
    with np.load(candidate, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    missing = set(FEATURE_COLUMNS) - arrays.keys()
    extra = arrays.keys() - set(FEATURE_COLUMNS)
    invalid_extra = {name for name in extra if not name.startswith(ENGINEERED_PREFIX)}
    if missing or invalid_extra:
        raise ValueError(
            f"invalid feature view columns: missing={sorted(missing)}, "
            f"invalid_extra={sorted(invalid_extra)}"
        )
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("feature view arrays have inconsistent lengths")
    expected = np.arange(len(arrays["row_id"]), dtype=np.int64)
    if not np.array_equal(arrays["row_id"], expected):
        raise ValueError("row_id must be contiguous and zero-based")
    for name in extra:
        value = arrays[name]
        if value.ndim != 1 or value.dtype.kind not in "biuf":
            raise ValueError(f"engineered feature {name} must be one-dimensional numeric data")
    return FeatureView(candidate, arrays, sha256_file(candidate))


def load_target_view(path: str | Path) -> TargetView:
    candidate = Path(path)
    with np.load(candidate, allow_pickle=False) as payload:
        if set(payload.files) != {"long_view"}:
            raise ValueError(f"target view must contain only long_view: {payload.files}")
        labels = payload["long_view"]
    if labels.ndim != 1 or not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("long_view target must be a one-dimensional binary array")
    return TargetView(candidate, labels.astype(np.float32, copy=False), sha256_file(candidate))
