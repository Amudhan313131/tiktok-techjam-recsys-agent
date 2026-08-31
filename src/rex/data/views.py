"""Typed access to sanitized feature and target views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rex.data.manifest import load_benchmark_manifest, sha256_file
from rex.features.static_metadata import (
    CATEGORICAL_ARRAYS,
    FORBIDDEN_STATISTIC_FIELDS,
    NUMERIC_ARRAYS,
    STATIC_IDENTITY_ARRAYS,
)


FEATURE_COLUMNS = (
    "row_id",
    "date",
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_ms",
)
TEMPORAL_ORDER_COLUMNS = ("time_ms", "source_row_key")
STATIC_METADATA_COLUMNS = CATEGORICAL_ARRAYS + NUMERIC_ARRAYS + STATIC_IDENTITY_ARRAYS
ENGINEERED_PREFIX = "fx__"
AUXILIARY_TARGET_COLUMNS = ("is_click", "is_like", "is_follow", "is_hate", "long_view")


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


@dataclass(frozen=True)
class FeedbackTargetView:
    path: Path
    arrays: dict[str, np.ndarray]
    sha256: str

    @property
    def rows(self) -> int:
        return len(self.arrays["long_view"])


def load_feature_view(path: str | Path) -> FeatureView:
    candidate = Path(path)
    with np.load(candidate, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    missing = set(FEATURE_COLUMNS) - arrays.keys()
    extra = arrays.keys() - set(FEATURE_COLUMNS)
    allowed_extra = set(TEMPORAL_ORDER_COLUMNS) | set(STATIC_METADATA_COLUMNS)
    invalid_extra = {
        name
        for name in extra
        if name not in allowed_extra and not name.startswith(ENGINEERED_PREFIX)
    }
    normalized = {
        name.removeprefix(ENGINEERED_PREFIX)
        .removeprefix("meta__")
        .removeprefix("meta_num__")
        .removeprefix("identity__")
        for name in arrays
    }
    forbidden_outcomes = set(load_benchmark_manifest()["forbidden_inference_columns"])
    forbidden_present = forbidden_outcomes.intersection(normalized)
    forbidden_statistics = FORBIDDEN_STATISTIC_FIELDS.intersection(normalized)
    if forbidden_present or forbidden_statistics:
        raise ValueError(
            "feature view contains forbidden inference fields: "
            f"outcomes={sorted(forbidden_present)}, "
            f"statistics={sorted(forbidden_statistics)}"
        )
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
    if set(TEMPORAL_ORDER_COLUMNS).intersection(arrays) not in (
        set(),
        set(TEMPORAL_ORDER_COLUMNS),
    ):
        raise ValueError("time_ms and source_row_key must be present together")
    if set(TEMPORAL_ORDER_COLUMNS) <= arrays.keys():
        time_ms = arrays["time_ms"]
        source_row_key = arrays["source_row_key"]
        if time_ms.ndim != 1 or time_ms.dtype.kind not in "iu" or np.any(time_ms < 0):
            raise ValueError("time_ms must be one-dimensional non-negative integer data")
        if (
            source_row_key.ndim != 1
            or source_row_key.dtype.kind not in "iu"
            or len(np.unique(source_row_key)) != len(source_row_key)
        ):
            raise ValueError("source_row_key must be one-dimensional unique integer data")
    for name in extra:
        value = arrays[name]
        if name in CATEGORICAL_ARRAYS:
            if value.ndim != 1 or value.dtype.kind not in "SUiu" or (
                value.dtype.kind in "iu" and np.any(value < 0)
            ):
                raise ValueError(
                    f"static categorical feature {name} must be one-dimensional "
                    "string or non-negative integer data"
                )
        elif name not in TEMPORAL_ORDER_COLUMNS and (
            value.ndim != 1 or value.dtype.kind not in "biuf" or not np.isfinite(value).all()
        ):
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


def load_feedback_target_view(path: str | Path) -> FeedbackTargetView:
    """Load the capability-separated train/valid prior-feedback vault."""

    candidate = Path(path)
    with np.load(candidate, allow_pickle=False) as payload:
        if set(payload.files) != set(AUXILIARY_TARGET_COLUMNS):
            raise ValueError(
                "feedback target view must contain exactly "
                f"{list(AUXILIARY_TARGET_COLUMNS)}: {payload.files}"
            )
        arrays = {
            name: payload[name].astype(np.float32, copy=False)
            for name in AUXILIARY_TARGET_COLUMNS
        }
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("feedback target arrays have inconsistent lengths")
    for name, values in arrays.items():
        if values.ndim != 1 or not np.isin(values, [0.0, 1.0]).all():
            raise ValueError(f"{name} feedback target must be one-dimensional binary data")
    return FeedbackTargetView(candidate, arrays, sha256_file(candidate))
