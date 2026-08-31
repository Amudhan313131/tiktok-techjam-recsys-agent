"""Deterministic, inference-safe KuaiRand static metadata joins.

Only immutable basic user/video attributes are exposed here.  Outcome-derived
monthly video statistics are deliberately not read by this module.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rex.data.manifest import canonical_json_bytes, sha256_bytes, sha256_file


STATIC_METADATA_SCHEMA_VERSION = "1.0"
UNKNOWN_TOKEN = "UNK"

USER_SOURCE = "user_features_pure.csv"
VIDEO_SOURCE = "video_features_basic_pure.csv"
FORBIDDEN_STATIC_SOURCE = "video_features_statistic_pure.csv"

SELECTED_USER_COLUMNS = (
    "user_active_degree",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
)
SELECTED_USER_NUMERIC_COLUMNS = (
    "follow_user_num",
    "fans_user_num",
    "friend_user_num",
    "register_days",
)
SELECTED_VIDEO_COLUMNS = (
    "video_type",
    "upload_type",
    "tag",
    "music_type",
)
DERIVED_VIDEO_COLUMNS = (
    "aspect_bucket",
    "upload_age_bucket",
    "duration_consistency_bucket",
)

CATEGORICAL_ARRAYS = tuple(
    [f"meta__{name}" for name in SELECTED_VIDEO_COLUMNS]
    + [f"meta__{name}" for name in DERIVED_VIDEO_COLUMNS]
    + [f"meta__{name}" for name in SELECTED_USER_COLUMNS]
)
NUMERIC_ARRAYS = (
    "meta_num__video_metadata_covered",
    "meta_num__user_metadata_covered",
    "meta_num__follow_user_num_log1p",
    "meta_num__fans_user_num_log1p",
    "meta_num__friend_user_num_log1p",
    "meta_num__register_days_log1p",
)
STATIC_IDENTITY_ARRAYS = ("identity__static_metadata_source",)

# These names come from the supplied month-level aggregate table.  Blocking
# their engineered aliases prevents a caller from smuggling post-period outcome
# aggregates into an otherwise sanitized view.
FORBIDDEN_STATISTIC_FIELDS = frozenset(
    {
        "counts",
        "show_cnt",
        "show_user_num",
        "play_cnt",
        "play_user_num",
        "play_duration",
        "complete_play_cnt",
        "complete_play_user_num",
        "valid_play_cnt",
        "valid_play_user_num",
        "long_time_play_cnt",
        "long_time_play_user_num",
        "short_time_play_cnt",
        "short_time_play_user_num",
        "play_progress",
        "comment_stay_duration",
        "like_cnt",
        "like_user_num",
        "click_like_cnt",
        "double_click_cnt",
        "cancel_like_cnt",
        "cancel_like_user_num",
        "comment_cnt",
        "comment_user_num",
        "direct_comment_cnt",
        "reply_comment_cnt",
        "delete_comment_cnt",
        "delete_comment_user_num",
        "comment_like_cnt",
        "comment_like_user_num",
        "follow_cnt",
        "follow_user_num",
        "cancel_follow_cnt",
        "cancel_follow_user_num",
        "share_cnt",
        "share_user_num",
        "download_cnt",
        "download_user_num",
        "report_cnt",
        "report_user_num",
        "reduce_similar_cnt",
        "reduce_similar_user_num",
        "collect_cnt",
        "collect_user_num",
        "cancel_collect_cnt",
        "cancel_collect_user_num",
        "direct_comment_user_num",
        "reply_comment_user_num",
        "share_all_cnt",
        "share_all_user_num",
        "outsite_share_all_cnt",
    }
)


class StaticMetadataError(ValueError):
    """Raised when an immutable side table violates its join contract."""


@dataclass(frozen=True)
class StaticSourceIdentity:
    name: str
    sha256: str | None
    header: tuple[str, ...]
    rows: int


@dataclass(frozen=True)
class StaticMetadataRow:
    video_values: tuple[str, ...]
    user_values: tuple[str, ...]
    user_numeric: tuple[float | None, ...]
    upload_ordinal: int | None
    aspect_bucket: str
    basic_duration_ms: float | None
    video_covered: bool
    user_covered: bool


@dataclass(frozen=True)
class StaticMetadataTables:
    users: Mapping[str, Mapping[str, str]]
    videos: Mapping[str, Mapping[str, str]]
    user_source: StaticSourceIdentity
    video_source: StaticSourceIdentity

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": STATIC_METADATA_SCHEMA_VERSION,
            "selected_user_columns": list(SELECTED_USER_COLUMNS),
            "selected_user_numeric_columns": list(SELECTED_USER_NUMERIC_COLUMNS),
            "selected_video_columns": list(SELECTED_VIDEO_COLUMNS),
            "derived_video_columns": list(DERIVED_VIDEO_COLUMNS),
            "forbidden_source": FORBIDDEN_STATIC_SOURCE,
            "sources": {
                "user": self.user_source.__dict__,
                "video": self.video_source.__dict__,
            },
        }


@dataclass(frozen=True)
class StaticMetadataTransform:
    upload_age_edges: tuple[float, ...]
    categorical_vocabularies: Mapping[str, frozenset[str]]
    identity_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": STATIC_METADATA_SCHEMA_VERSION,
            "unknown_token": UNKNOWN_TOKEN,
            "upload_age_edges_days": list(self.upload_age_edges),
            "categorical_vocabularies": {
                name: _ordered_vocabulary(values)
                for name, values in sorted(self.categorical_vocabularies.items())
            },
            "categorical_encoding": "sha256_token_prefix_63bit_with_UNK_at_zero",
            "identity_sha256": self.identity_sha256,
        }


def _load_table(path: Path, key: str) -> tuple[dict[str, dict[str, str]], StaticSourceIdentity]:
    if not path.is_file():
        return {}, StaticSourceIdentity(path.name, None, (), 0)
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if key not in header:
            raise StaticMetadataError(f"{path.name} is missing key column {key!r}")
        for source_row, raw in enumerate(reader, start=1):
            identity = str(raw.get(key, "")).strip()
            if not identity:
                raise StaticMetadataError(f"{path.name} row {source_row} has an empty {key}")
            if identity in rows:
                raise StaticMetadataError(f"{path.name} contains duplicate {key}={identity!r}")
            rows[identity] = {name: str(value or "").strip() for name, value in raw.items()}
    return rows, StaticSourceIdentity(path.name, sha256_file(path), header, len(rows))


def load_static_metadata(data_dir: str | Path) -> StaticMetadataTables:
    """Load only the two allow-listed basic side tables."""

    root = Path(data_dir)
    users, user_source = _load_table(root / USER_SOURCE, "user_id")
    videos, video_source = _load_table(root / VIDEO_SOURCE, "video_id")
    return StaticMetadataTables(users, videos, user_source, video_source)


def _token(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null", "-124"}:
        return UNKNOWN_TOKEN
    return text


def _optional_float(value: object) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _upload_ordinal(value: object) -> int | None:
    text = str(value or "").strip()
    try:
        return date.fromisoformat(text).toordinal()
    except ValueError:
        return None


def interaction_ordinal(value: int) -> int:
    text = str(int(value))
    if len(text) != 8:
        raise StaticMetadataError(f"interaction date must be YYYYMMDD, observed {value!r}")
    try:
        return date(int(text[:4]), int(text[4:6]), int(text[6:])).toordinal()
    except ValueError as error:
        raise StaticMetadataError(f"invalid interaction date {value!r}") from error


def _aspect_bucket(width: object, height: object) -> str:
    parsed_width = _optional_float(width)
    parsed_height = _optional_float(height)
    if parsed_width is None or parsed_height in {None, 0.0}:
        return UNKNOWN_TOKEN
    ratio = parsed_width / parsed_height
    if ratio < 0.8:
        return "portrait"
    if ratio <= 1.25:
        return "square"
    return "landscape"


def join_static_metadata(
    tables: StaticMetadataTables,
    *,
    user_id: str,
    video_id: str,
) -> StaticMetadataRow:
    user = tables.users.get(str(user_id))
    video = tables.videos.get(str(video_id))
    return StaticMetadataRow(
        video_values=tuple(_token((video or {}).get(name)) for name in SELECTED_VIDEO_COLUMNS),
        user_values=tuple(_token((user or {}).get(name)) for name in SELECTED_USER_COLUMNS),
        user_numeric=tuple(
            _optional_float((user or {}).get(name)) for name in SELECTED_USER_NUMERIC_COLUMNS
        ),
        upload_ordinal=_upload_ordinal((video or {}).get("upload_dt")),
        aspect_bucket=_aspect_bucket(
            (video or {}).get("server_width"), (video or {}).get("server_height")
        ),
        basic_duration_ms=_optional_float((video or {}).get("video_duration")),
        video_covered=video is not None,
        user_covered=user is not None,
    )


def _raw_categories(
    metadata: StaticMetadataRow,
    *,
    interaction_date: int,
    duration_ms: float,
    upload_age_edges: Sequence[float],
) -> dict[str, str]:
    values: dict[str, str] = {}
    values.update(
        zip((f"meta__{name}" for name in SELECTED_VIDEO_COLUMNS), metadata.video_values)
    )
    values.update(zip((f"meta__{name}" for name in SELECTED_USER_COLUMNS), metadata.user_values))
    values["meta__aspect_bucket"] = metadata.aspect_bucket
    if metadata.upload_ordinal is None:
        values["meta__upload_age_bucket"] = UNKNOWN_TOKEN
    else:
        age = interaction_ordinal(interaction_date) - metadata.upload_ordinal
        values["meta__upload_age_bucket"] = f"q{int(np.searchsorted(upload_age_edges, age))}"
    if metadata.basic_duration_ms in {None, 0.0} or not math.isfinite(float(duration_ms)):
        values["meta__duration_consistency_bucket"] = UNKNOWN_TOKEN
    else:
        ratio = float(duration_ms) / float(metadata.basic_duration_ms)
        if ratio < 0.8:
            values["meta__duration_consistency_bucket"] = "shorter"
        elif ratio <= 1.25:
            values["meta__duration_consistency_bucket"] = "consistent"
        else:
            values["meta__duration_consistency_bucket"] = "longer"
    return values


def fit_static_metadata_transform(
    metadata_rows: Sequence[StaticMetadataRow],
    interaction_dates: Sequence[int],
    durations_ms: Sequence[float],
) -> StaticMetadataTransform:
    if not (len(metadata_rows) == len(interaction_dates) == len(durations_ms)):
        raise StaticMetadataError("static metadata fit inputs have inconsistent lengths")
    ages = [
        interaction_ordinal(int(day)) - row.upload_ordinal
        for row, day in zip(metadata_rows, interaction_dates)
        if row.upload_ordinal is not None
    ]
    if ages:
        edges = tuple(
            map(
                float,
                np.unique(np.quantile(np.asarray(ages, dtype=np.float64), [0.2, 0.4, 0.6, 0.8])),
            )
        )
    else:
        edges = ()
    vocabularies: dict[str, set[str]] = {name: {UNKNOWN_TOKEN} for name in CATEGORICAL_ARRAYS}
    for metadata, day, duration in zip(metadata_rows, interaction_dates, durations_ms):
        for name, value in _raw_categories(
            metadata,
            interaction_date=int(day),
            duration_ms=float(duration),
            upload_age_edges=edges,
        ).items():
            vocabularies[name].add(value)
    frozen = {name: frozenset(values) for name, values in vocabularies.items()}
    payload = {
        "schema_version": STATIC_METADATA_SCHEMA_VERSION,
        "unknown_token": UNKNOWN_TOKEN,
        "upload_age_edges_days": list(edges),
        "categorical_vocabularies": {
            name: _ordered_vocabulary(values) for name, values in sorted(frozen.items())
        },
        "categorical_encoding": "sha256_token_prefix_63bit_with_UNK_at_zero",
    }
    return StaticMetadataTransform(
        edges,
        frozen,
        sha256_bytes(canonical_json_bytes(payload)),
    )


def fit_static_metadata_transform_for_rows(
    tables: StaticMetadataTables,
    user_ids: Sequence[str],
    video_ids: Sequence[str],
    interaction_dates: Sequence[int],
    durations_ms: Sequence[float],
) -> StaticMetadataTransform:
    """Fit without retaining one joined metadata object per interaction."""

    rows = len(user_ids)
    if not (rows == len(video_ids) == len(interaction_dates) == len(durations_ms)):
        raise StaticMetadataError("static metadata fit inputs have inconsistent lengths")
    ages: list[int] = []
    for user_id, video_id, day in zip(user_ids, video_ids, interaction_dates):
        metadata = join_static_metadata(tables, user_id=str(user_id), video_id=str(video_id))
        if metadata.upload_ordinal is not None:
            ages.append(interaction_ordinal(int(day)) - metadata.upload_ordinal)
    edges = (
        tuple(
            map(
                float,
                np.unique(
                    np.quantile(np.asarray(ages, dtype=np.float64), [0.2, 0.4, 0.6, 0.8])
                ),
            )
        )
        if ages
        else ()
    )
    vocabularies: dict[str, set[str]] = {name: {UNKNOWN_TOKEN} for name in CATEGORICAL_ARRAYS}
    for user_id, video_id, day, duration in zip(
        user_ids, video_ids, interaction_dates, durations_ms
    ):
        metadata = join_static_metadata(tables, user_id=str(user_id), video_id=str(video_id))
        for name, value in _raw_categories(
            metadata,
            interaction_date=int(day),
            duration_ms=float(duration),
            upload_age_edges=edges,
        ).items():
            vocabularies[name].add(value)
    frozen = {name: frozenset(values) for name, values in vocabularies.items()}
    payload = {
        "schema_version": STATIC_METADATA_SCHEMA_VERSION,
        "unknown_token": UNKNOWN_TOKEN,
        "upload_age_edges_days": list(edges),
        "categorical_vocabularies": {
            name: _ordered_vocabulary(values) for name, values in sorted(frozen.items())
        },
        "categorical_encoding": "sha256_token_prefix_63bit_with_UNK_at_zero",
    }
    return StaticMetadataTransform(edges, frozen, sha256_bytes(canonical_json_bytes(payload)))


def materialize_static_metadata(
    metadata_rows: Iterable[StaticMetadataRow],
    interaction_dates: Sequence[int],
    durations_ms: Sequence[float],
    transform: StaticMetadataTransform,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rows = len(interaction_dates)
    if rows != len(durations_ms):
        raise StaticMetadataError("static metadata materialization inputs have inconsistent lengths")
    category_codes = {
        name: _categorical_codebook(vocabulary)
        for name, vocabulary in transform.categorical_vocabularies.items()
    }
    categories = {
        name: np.zeros(rows, dtype=np.int64) for name in CATEGORICAL_ARRAYS
    }
    numeric = {name: np.zeros(rows, dtype=np.float32) for name in NUMERIC_ARRAYS}
    unknown_counts = {name: 0 for name in CATEGORICAL_ARRAYS}
    video_rows_covered = 0
    user_rows_covered = 0
    processed = 0
    for index, (metadata, day, duration) in enumerate(
        zip(metadata_rows, interaction_dates, durations_ms)
    ):
        processed = index + 1
        raw = _raw_categories(
            metadata,
            interaction_date=int(day),
            duration_ms=float(duration),
            upload_age_edges=transform.upload_age_edges,
        )
        for name in CATEGORICAL_ARRAYS:
            value = raw[name]
            if value not in transform.categorical_vocabularies[name]:
                value = UNKNOWN_TOKEN
            categories[name][index] = category_codes[name][value]
            unknown_counts[name] += int(value == UNKNOWN_TOKEN)
        video_rows_covered += int(metadata.video_covered)
        user_rows_covered += int(metadata.user_covered)
        numeric["meta_num__video_metadata_covered"][index] = float(metadata.video_covered)
        numeric["meta_num__user_metadata_covered"][index] = float(metadata.user_covered)
        for name, value in zip(SELECTED_USER_NUMERIC_COLUMNS, metadata.user_numeric):
            numeric[f"meta_num__{name}_log1p"][index] = (
                math.log1p(value) if value is not None else 0.0
            )
    if processed != rows:
        raise StaticMetadataError("static metadata iterator ended before aligned input arrays")
    arrays: dict[str, np.ndarray] = dict(categories)
    arrays.update(numeric)
    coverage = {
        "rows": rows,
        "video_rows_covered": video_rows_covered,
        "user_rows_covered": user_rows_covered,
        "video_coverage_rate": (
            float(video_rows_covered / rows) if rows else 0.0
        ),
        "user_coverage_rate": (
            float(user_rows_covered / rows) if rows else 0.0
        ),
        "unknown_counts": unknown_counts,
    }
    return arrays, coverage


def materialize_static_metadata_for_rows(
    tables: StaticMetadataTables,
    user_ids: Sequence[str],
    video_ids: Sequence[str],
    interaction_dates: Sequence[int],
    durations_ms: Sequence[float],
    transform: StaticMetadataTransform,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rows = len(user_ids)
    if not (rows == len(video_ids) == len(interaction_dates) == len(durations_ms)):
        raise StaticMetadataError("static metadata materialization inputs have inconsistent lengths")
    metadata = (
        join_static_metadata(tables, user_id=str(user_id), video_id=str(video_id))
        for user_id, video_id in zip(user_ids, video_ids)
    )
    arrays, coverage = materialize_static_metadata(
        metadata, interaction_dates, durations_ms, transform
    )
    source_identity = sha256_bytes(canonical_json_bytes(tables.identity_payload()))
    arrays[STATIC_IDENTITY_ARRAYS[0]] = np.full(
        rows,
        int(source_identity[:15], 16),
        dtype=np.int64,
    )
    return arrays, coverage


def static_metadata_manifest(
    tables: StaticMetadataTables,
    transform: StaticMetadataTransform,
    coverage_by_split: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        **tables.identity_payload(),
        "base_view_policy": "stored_but_model_opt_in",
        "model_opt_in_prefixes": ["meta__", "meta_num__"],
        "transform": transform.payload(),
        "coverage_by_split": {
            name: dict(value) for name, value in sorted(coverage_by_split.items())
        },
    }
    payload["identity_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def _ordered_vocabulary(values: Iterable[str]) -> list[str]:
    return [UNKNOWN_TOKEN, *sorted(value for value in values if value != UNKNOWN_TOKEN)]


def _categorical_codebook(values: Iterable[str]) -> dict[str, int]:
    codebook = {
        value: (
            0
            if value == UNKNOWN_TOKEN
            else int(sha256_bytes(value.encode("utf-8"))[:15], 16)
        )
        for value in values
    }
    if len(set(codebook.values())) != len(codebook):
        raise StaticMetadataError("static categorical hash codes collided")
    return codebook
