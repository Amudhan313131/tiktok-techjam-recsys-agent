"""Deterministic temporal ordering helpers for point-in-time features."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


MILLISECONDS_PER_DAY = 86_400_000.0


def validate_temporal_keys(
    time_ms: np.ndarray,
    source_row_keys: np.ndarray,
    *,
    expected_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate immutable event times and tie-break keys.

    The source row key is used only to make ordering deterministic. Rows with
    equal timestamps are still emitted as one group, so their outcomes cannot
    influence one another.
    """

    times = np.asarray(time_ms)
    keys = np.asarray(source_row_keys)
    rows = len(times)
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"temporal key rows differ: {rows} != {expected_rows}")
    if times.ndim != 1 or keys.ndim != 1 or len(keys) != rows:
        raise ValueError("time_ms and source_row_keys must be aligned one-dimensional arrays")
    if not np.issubdtype(times.dtype, np.number) or not np.isfinite(times).all():
        raise ValueError("time_ms must be finite numeric values")
    normalized_times = times.astype(np.int64, copy=False)
    if np.any(normalized_times < 0):
        raise ValueError("time_ms cannot be negative")
    normalized_keys = keys.astype(str)
    if len(np.unique(normalized_keys)) != rows:
        raise ValueError("source_row_keys must be globally unique")
    return normalized_times, normalized_keys


def strict_timestamp_groups(
    time_ms: np.ndarray,
    source_row_keys: np.ndarray,
) -> Iterator[np.ndarray]:
    """Yield deterministic timestamp groups in chronological order."""

    times, keys = validate_temporal_keys(time_ms, source_row_keys)
    order = np.lexsort((keys, times))
    if len(order) == 0:
        return
    ordered_times = times[order]
    boundaries = np.flatnonzero(ordered_times[1:] != ordered_times[:-1]) + 1
    for group in np.split(order, boundaries):
        yield group


def elapsed_days(later_ms: int | float, earlier_ms: int | float) -> float:
    """Return a non-negative elapsed-day interval."""

    return max(0.0, (float(later_ms) - float(earlier_ms)) / MILLISECONDS_PER_DAY)


def exponential_decay(
    later_ms: int | float,
    earlier_ms: int | float,
    half_life_days: float,
) -> float:
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    return float(0.5 ** (elapsed_days(later_ms, earlier_ms) / half_life_days))
