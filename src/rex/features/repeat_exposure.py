"""Point-in-time repeated user-video exposure features."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from rex.data.temporal import date_batches
from rex.features.base import FeatureBundle


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
            days_since[index] = int(dates[index] - last_date[key]) if key in last_date else -1
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
