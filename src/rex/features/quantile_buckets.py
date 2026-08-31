"""Label-free train-fitted buckets for numeric point-in-time features.

The context FM consumes categorical fields.  This adapter converts numeric
history features into compact categorical IDs while preserving the temporal
contract of the source bundle: bin boundaries are fitted exactly once on the
training bundle, serialized, and then applied unchanged to later partitions.
No target or label is accepted by any public function in this module.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from rex.features.base import FeatureBundle


STATE_SCHEMA_VERSION = "1.0"
MISSING_BUCKET = 0
FIRST_VALUE_BUCKET = 1
QUANTILE_STRATEGY = "quantile"
CAPPED_COUNT_STRATEGY = "capped_count"
SUPPORTED_STRATEGIES = frozenset({QUANTILE_STRATEGY, CAPPED_COUNT_STRATEGY})

# A conservative default subset from candidate_recency and
# multifeedback_history.  Callers can explicitly request any numeric fields.
DEFAULT_HIGH_SIGNAL_FIELDS = (
    "pt_user_author_rate",
    "pt_user_author_count",
    "pt_user_author_days_since",
    "pt_user_video_count",
    "pt_user_video_positive",
    "pt_user_video_days_since",
    "pt_user_rate_h1p0",
    "pt_user_count_h1p0",
    "pt_user_rate_h7p0",
    "pt_user_count_h7p0",
    "pt_feedback_user_count",
    "pt_feedback_user_click_rate",
    "pt_feedback_user_long_view_rate",
    "pt_feedback_user_author_count",
    "pt_feedback_user_author_click_rate",
    "pt_feedback_user_author_long_view_rate",
)


def _default_strategy(name: str) -> str:
    if name.endswith("_count") or name.endswith("_positive"):
        return CAPPED_COUNT_STRATEGY
    return QUANTILE_STRATEGY


def _default_missing_values(name: str) -> tuple[float, ...]:
    if name.endswith("_days_since") or name.endswith("_last_outcome"):
        return (-1.0,)
    return ()


def _validated_bundle_rows(bundle: FeatureBundle, *, fitting: bool) -> int:
    if not bundle.arrays:
        raise ValueError("numeric bucket input bundle cannot be empty")
    first = np.asarray(next(iter(bundle.arrays.values())))
    if first.ndim != 1:
        raise ValueError("numeric bucket features must be one-dimensional")
    rows = len(first)
    if fitting and rows < 1:
        raise ValueError("numeric buckets cannot be fitted on an empty bundle")
    bundle.validate(rows)
    for name, raw in bundle.arrays.items():
        values = np.asarray(raw)
        if values.ndim != 1 or len(values) != rows:
            raise ValueError(f"numeric bucket feature {name} is misaligned")
        if values.dtype.kind not in "biuf":
            raise ValueError(f"numeric bucket feature {name} must be numeric")
        if not np.isfinite(values).all():
            raise ValueError(f"numeric bucket feature {name} contains NaN or Inf")
    return rows


def _canonical_missing(values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(sorted({float(value) for value in values}))
    if not np.isfinite(np.asarray(result, dtype=np.float64)).all():
        raise ValueError("numeric bucket missing sentinels must be finite")
    return result


def _quantile_edges(values: np.ndarray, bins: int) -> tuple[float, ...]:
    if len(values) < 2 or float(np.min(values)) == float(np.max(values)):
        return ()
    raw = np.quantile(values, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    return tuple(
        float(value)
        for value in np.unique(raw)
        if minimum < float(value) < maximum
    )


def _capped_count_edges(values: np.ndarray, cap: int) -> tuple[float, ...]:
    if np.any(values < 0):
        raise ValueError("capped-count bucket features cannot contain negative values")
    if not len(values):
        return ()
    observed_maximum = min(float(np.max(values)), float(cap))
    boundaries: list[float] = []
    upper = 1
    while upper - 0.5 < observed_maximum:
        boundaries.append(float(upper) - 0.5)
        upper *= 2
    return tuple(boundaries)


@dataclass(frozen=True)
class FittedNumericBucket:
    """Frozen categorical mapping for one numeric source feature."""

    source: str
    output: str
    strategy: str
    edges: tuple[float, ...]
    missing_values: tuple[float, ...]
    training_rows: int
    non_missing_rows: int
    training_min: float | None
    training_max: float | None
    count_cap: int | None
    source_provenance: dict[str, object]

    def __post_init__(self) -> None:
        if not self.source or not self.output:
            raise ValueError("numeric bucket source and output names cannot be empty")
        if self.strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"unsupported numeric bucket strategy: {self.strategy}")
        edges = np.asarray(self.edges, dtype=np.float64)
        if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
            raise ValueError("numeric bucket edges must be finite and strictly increasing")
        _canonical_missing(self.missing_values)
        if self.training_rows < 1 or not 0 <= self.non_missing_rows <= self.training_rows:
            raise ValueError("numeric bucket training row counts are invalid")
        if self.non_missing_rows:
            if self.training_min is None or self.training_max is None:
                raise ValueError("numeric bucket training range is missing")
            if not np.isfinite([self.training_min, self.training_max]).all():
                raise ValueError("numeric bucket training range must be finite")
            if self.training_min > self.training_max:
                raise ValueError("numeric bucket training range is reversed")
        elif self.training_min is not None or self.training_max is not None:
            raise ValueError("all-missing numeric buckets cannot have a training range")
        if self.strategy == CAPPED_COUNT_STRATEGY:
            if self.count_cap is None or self.count_cap < 1:
                raise ValueError("capped-count buckets require a positive count cap")
        elif self.count_cap is not None:
            raise ValueError("quantile buckets cannot define a count cap")
        try:
            json.dumps(self.source_provenance, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("numeric bucket source provenance must be JSON serializable") from error

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "FittedNumericBucket":
        return cls(
            source=str(value["source"]),
            output=str(value["output"]),
            strategy=str(value["strategy"]),
            edges=tuple(float(item) for item in value["edges"]),
            missing_values=tuple(float(item) for item in value["missing_values"]),
            training_rows=int(value["training_rows"]),
            non_missing_rows=int(value["non_missing_rows"]),
            training_min=(
                None if value["training_min"] is None else float(value["training_min"])
            ),
            training_max=(
                None if value["training_max"] is None else float(value["training_max"])
            ),
            count_cap=None if value["count_cap"] is None else int(value["count_cap"]),
            source_provenance=dict(value["source_provenance"]),
        )


@dataclass(frozen=True)
class QuantileBucketState:
    """Serializable collection of frozen numeric bucket mappings."""

    buckets: tuple[FittedNumericBucket, ...]

    def __post_init__(self) -> None:
        if not self.buckets:
            raise ValueError("at least one numeric bucket is required")
        sources = [item.source for item in self.buckets]
        outputs = [item.output for item in self.buckets]
        if len(set(sources)) != len(sources):
            raise ValueError("numeric bucket source names must be unique")
        if len(set(outputs)) != len(outputs):
            raise ValueError("numeric bucket output names must be unique")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "missing_bucket": MISSING_BUCKET,
            "first_value_bucket": FIRST_VALUE_BUCKET,
            "buckets": [item.to_json() for item in self.buckets],
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "QuantileBucketState":
        if value.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ValueError("unsupported numeric-bucket state schema")
        if value.get("missing_bucket") != MISSING_BUCKET:
            raise ValueError("numeric-bucket missing index drifted")
        if value.get("first_value_bucket") != FIRST_VALUE_BUCKET:
            raise ValueError("numeric-bucket value offset drifted")
        return cls(
            tuple(FittedNumericBucket.from_json(item) for item in value["buckets"])
        )


def fit_quantile_buckets(
    train: FeatureBundle,
    fields: Iterable[str] | None = None,
    *,
    strategies: Mapping[str, str] | None = None,
    missing_values: Mapping[str, Iterable[float]] | None = None,
    quantile_bins: int = 8,
    count_cap: int = 64,
    output_prefix: str = "bucket__",
) -> QuantileBucketState:
    """Fit bucket edges solely from numeric training features.

    With ``fields=None``, the available members of
    :data:`DEFAULT_HIGH_SIGNAL_FIELDS` are selected in a stable order.  Count
    fields default to capped powers-of-two bands; other fields use quantiles.
    """

    rows = _validated_bundle_rows(train, fitting=True)
    if quantile_bins < 2:
        raise ValueError("quantile_bins must be at least two")
    if count_cap < 1:
        raise ValueError("count_cap must be positive")
    if not output_prefix:
        raise ValueError("numeric bucket output prefix cannot be empty")

    selected = (
        tuple(name for name in DEFAULT_HIGH_SIGNAL_FIELDS if name in train.arrays)
        if fields is None
        else tuple(str(name) for name in fields)
    )
    if not selected:
        raise ValueError("no numeric bucket fields were selected")
    if len(set(selected)) != len(selected):
        raise ValueError("numeric bucket fields must be unique")
    missing_fields = set(selected).difference(train.arrays)
    if missing_fields:
        raise ValueError(f"numeric bucket fields are missing: {sorted(missing_fields)}")

    strategy_map = dict(strategies or {})
    missing_map = dict(missing_values or {})
    unknown_overrides = (set(strategy_map) | set(missing_map)).difference(selected)
    if unknown_overrides:
        raise ValueError(
            f"numeric bucket overrides reference unselected fields: {sorted(unknown_overrides)}"
        )

    fitted: list[FittedNumericBucket] = []
    for name in selected:
        strategy = strategy_map.get(name, _default_strategy(name))
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"unsupported numeric bucket strategy: {strategy}")
        sentinels = _canonical_missing(
            missing_map.get(name, _default_missing_values(name))
        )
        values = np.asarray(train.arrays[name], dtype=np.float64)
        mask = ~np.isin(values, np.asarray(sentinels, dtype=np.float64))
        observed = values[mask]
        if strategy == QUANTILE_STRATEGY:
            edges = _quantile_edges(observed, quantile_bins)
            cap: int | None = None
        else:
            edges = _capped_count_edges(observed, count_cap)
            cap = count_cap
        fitted.append(
            FittedNumericBucket(
                source=name,
                output=f"{output_prefix}{name}",
                strategy=strategy,
                edges=edges,
                missing_values=sentinels,
                training_rows=rows,
                non_missing_rows=int(len(observed)),
                training_min=float(np.min(observed)) if len(observed) else None,
                training_max=float(np.max(observed)) if len(observed) else None,
                count_cap=cap,
                source_provenance=dict(train.provenance[name]),
            )
        )
    return QuantileBucketState(tuple(fitted))


def apply_quantile_buckets(
    bundle: FeatureBundle,
    state: QuantileBucketState,
) -> FeatureBundle:
    """Apply frozen bins without consulting labels or the apply distribution."""

    rows = _validated_bundle_rows(bundle, fitting=False)
    arrays: dict[str, np.ndarray] = {}
    provenance: dict[str, dict[str, object]] = {}
    for fitted in state.buckets:
        if fitted.source not in bundle.arrays:
            raise ValueError(f"numeric bucket source feature is missing: {fitted.source}")
        values = np.asarray(bundle.arrays[fitted.source], dtype=np.float64)
        if values.ndim != 1 or len(values) != rows:
            raise ValueError(f"numeric bucket feature {fitted.source} is misaligned")
        missing = np.isin(values, np.asarray(fitted.missing_values, dtype=np.float64))
        transformed = values.copy()
        if fitted.strategy == CAPPED_COUNT_STRATEGY:
            if np.any(transformed[~missing] < 0):
                raise ValueError(
                    f"capped-count bucket feature {fitted.source} contains negative values"
                )
            transformed = np.minimum(transformed, float(fitted.count_cap))
        encoded = np.full(rows, MISSING_BUCKET, dtype=np.int32)
        encoded[~missing] = (
            np.searchsorted(
                np.asarray(fitted.edges, dtype=np.float64),
                transformed[~missing],
                side="right",
            ).astype(np.int32)
            + FIRST_VALUE_BUCKET
        )
        arrays[fitted.output] = encoded
        provenance[fitted.output] = {
            "cutoff": "train-fitted numeric distribution; no targets",
            "source_feature": fitted.source,
            "strategy": fitted.strategy,
            "edges": list(fitted.edges),
            "missing_values": list(fitted.missing_values),
            "missing_bucket": MISSING_BUCKET,
            "first_value_bucket": FIRST_VALUE_BUCKET,
            "training_rows": fitted.training_rows,
            "non_missing_training_rows": fitted.non_missing_rows,
            "training_min": fitted.training_min,
            "training_max": fitted.training_max,
            "count_cap": fitted.count_cap,
            "source_provenance": fitted.source_provenance,
        }
    result = FeatureBundle(arrays, provenance)
    result.validate(rows)
    return result


def fit_transform_quantile_buckets(
    train: FeatureBundle,
    fields: Iterable[str] | None = None,
    **kwargs: Any,
) -> tuple[QuantileBucketState, FeatureBundle]:
    """Fit on and transform a training bundle with the same frozen state."""

    state = fit_quantile_buckets(train, fields, **kwargs)
    return state, apply_quantile_buckets(train, state)
