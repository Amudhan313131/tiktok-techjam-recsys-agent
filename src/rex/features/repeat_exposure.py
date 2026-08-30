"""Point-in-time repeated user-video exposure features."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from functools import lru_cache

import numpy as np

from rex.data.temporal import date_batches
from rex.features.base import FeatureBundle


@lru_cache(maxsize=128)
def _calendar_date(value: int) -> date:
    text = str(int(value))
    if len(text) != 8:
        raise ValueError(f"date must be YYYYMMDD, got {value}")
    return date(int(text[:4]), int(text[4:6]), int(text[6:8]))


def calendar_days_between(later: int, earlier: int) -> int:
    return (_calendar_date(later) - _calendar_date(earlier)).days


def repeat_exposure_features(
    user_ids: np.ndarray,
    video_ids: np.ndarray,
    dates: np.ndarray,
    row_ids: np.ndarray,
    labels: np.ndarray,
) -> FeatureBundle:
    rows = len(user_ids)
    if not all(len(value) == rows for value in (video_ids, dates, row_ids, labels)):
        raise ValueError("repeat-exposure inputs have different lengths")
    counts: dict[tuple[str, str], int] = defaultdict(int)
    positives: dict[tuple[str, str], int] = defaultdict(int)
    last_outcome: dict[tuple[str, str], float] = {}
    last_date: dict[tuple[str, str], int] = {}
    prior_count = np.zeros(rows, dtype=np.int32)
    prior_positive = np.zeros(rows, dtype=np.int32)
    prior_last = np.full(rows, -1.0, dtype=np.float32)
    days_since = np.full(rows, -1, dtype=np.int32)
    for batch in date_batches(dates, row_ids):
        for index in batch:
            key = (str(user_ids[index]), str(video_ids[index]))
            prior_count[index] = counts[key]
            prior_positive[index] = positives[key]
            prior_last[index] = last_outcome.get(key, -1.0)
            days_since[index] = (
                calendar_days_between(int(dates[index]), last_date[key]) if key in last_date else -1
            )
        for index in batch:
            key = (str(user_ids[index]), str(video_ids[index]))
            counts[key] += 1
            positives[key] += int(labels[index])
            last_outcome[key] = float(labels[index])
            last_date[key] = int(dates[index])
    cutoff = "strictly earlier date for same user-video pair"
    names = {
        "repeat_prior_count": prior_count,
        "repeat_prior_positive": prior_positive,
        "repeat_last_outcome": prior_last,
        "repeat_days_since": days_since,
    }
    return FeatureBundle(names, {name: {"cutoff": cutoff} for name in names})


def fit_repeat_exposure_state(
    user_ids: np.ndarray,
    video_ids: np.ndarray,
    dates: np.ndarray,
    labels: np.ndarray,
) -> dict[str, object]:
    """Fit user-video state from a completed historical partition."""

    counts: dict[tuple[str, str], int] = defaultdict(int)
    positives: dict[tuple[str, str], int] = defaultdict(int)
    last_outcome: dict[tuple[str, str], float] = {}
    last_date: dict[tuple[str, str], int] = {}
    order = np.argsort(dates, kind="stable")
    for index in order:
        key = (str(user_ids[index]), str(video_ids[index]))
        counts[key] += 1
        positives[key] += int(labels[index])
        last_outcome[key] = float(labels[index])
        last_date[key] = int(dates[index])
    return {
        "counts": dict(counts),
        "positives": dict(positives),
        "last_outcome": last_outcome,
        "last_date": last_date,
    }


def apply_repeat_exposure_state(
    user_ids: np.ndarray,
    video_ids: np.ndarray,
    dates: np.ndarray,
    state: dict[str, object],
) -> FeatureBundle:
    """Apply frozen training history without consuming evaluation outcomes."""

    counts = state["counts"]
    positives = state["positives"]
    last_outcome = state["last_outcome"]
    last_date = state["last_date"]
    prior_count = np.zeros(len(user_ids), dtype=np.int32)
    prior_positive = np.zeros(len(user_ids), dtype=np.int32)
    prior_last = np.full(len(user_ids), -1.0, dtype=np.float32)
    days_since = np.full(len(user_ids), -1, dtype=np.int32)
    for index, (user, video, current_date) in enumerate(
        zip(user_ids, video_ids, dates, strict=True)
    ):
        key = (str(user), str(video))
        prior_count[index] = int(counts.get(key, 0))
        prior_positive[index] = int(positives.get(key, 0))
        prior_last[index] = float(last_outcome.get(key, -1.0))
        if key in last_date:
            days_since[index] = calendar_days_between(int(current_date), int(last_date[key]))
    arrays = {
        "repeat_prior_count": prior_count,
        "repeat_prior_positive": prior_positive,
        "repeat_last_outcome": prior_last,
        "repeat_days_since": days_since,
    }
    cutoff = "frozen fit split only; no evaluation labels"
    return FeatureBundle(arrays, {name: {"cutoff": cutoff} for name in arrays})
