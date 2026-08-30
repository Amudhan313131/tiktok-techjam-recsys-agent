"""Leakage-safe exponentially decayed user-history summaries."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from rex.data.temporal import date_batches
from rex.features.base import FeatureBundle
from rex.features.repeat_exposure import calendar_days_between


def _decay(days: int, half_life_days: float) -> float:
    return float(0.5 ** (max(days, 0) / half_life_days))


def recency_history_features(
    user_ids: np.ndarray,
    dates: np.ndarray,
    row_ids: np.ndarray,
    labels: np.ndarray,
    *,
    half_life_days: float = 7.0,
    prior_strength: float = 10.0,
) -> FeatureBundle:
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    positive: dict[str, float] = defaultdict(float)
    count: dict[str, float] = defaultdict(float)
    last_date: dict[str, int] = {}
    global_mean = float(np.mean(labels)) if len(labels) else 0.5
    rates = np.empty(len(labels), dtype=np.float32)
    counts = np.empty(len(labels), dtype=np.float32)
    for batch in date_batches(dates, row_ids):
        current = int(dates[batch[0]])
        for index in batch:
            user = str(user_ids[index])
            factor = (
                _decay(calendar_days_between(current, last_date[user]), half_life_days)
                if user in last_date
                else 1.0
            )
            decayed_positive = positive[user] * factor
            decayed_count = count[user] * factor
            rates[index] = (decayed_positive + prior_strength * global_mean) / (
                decayed_count + prior_strength
            )
            counts[index] = decayed_count
        for index in batch:
            user = str(user_ids[index])
            if last_date.get(user) != current:
                factor = (
                    _decay(calendar_days_between(current, last_date[user]), half_life_days)
                    if user in last_date
                    else 1.0
                )
                positive[user] *= factor
                count[user] *= factor
                last_date[user] = current
            positive[user] += float(labels[index])
            count[user] += 1.0
    arrays = {"recency_user_rate": rates, "recency_user_count": counts}
    provenance = {
        name: {
            "cutoff": "strictly earlier date; same-day rows excluded",
            "half_life_days": half_life_days,
            "prior_strength": prior_strength,
        }
        for name in arrays
    }
    return FeatureBundle(arrays, provenance)


def fit_recency_state(
    user_ids: np.ndarray,
    dates: np.ndarray,
    labels: np.ndarray,
    *,
    half_life_days: float = 7.0,
    prior_strength: float = 10.0,
) -> dict[str, object]:
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    positive: dict[str, float] = defaultdict(float)
    count: dict[str, float] = defaultdict(float)
    last_date: dict[str, int] = {}
    for index in np.argsort(dates, kind="stable"):
        user = str(user_ids[index])
        current = int(dates[index])
        if user in last_date and current != last_date[user]:
            factor = _decay(calendar_days_between(current, last_date[user]), half_life_days)
            positive[user] *= factor
            count[user] *= factor
        positive[user] += float(labels[index])
        count[user] += 1.0
        last_date[user] = current
    return {
        "positive": dict(positive),
        "count": dict(count),
        "last_date": last_date,
        "global_mean": float(np.mean(labels)) if len(labels) else 0.5,
        "half_life_days": half_life_days,
        "prior_strength": prior_strength,
    }


def apply_recency_state(
    user_ids: np.ndarray,
    dates: np.ndarray,
    state: dict[str, object],
) -> FeatureBundle:
    positive = state["positive"]
    count = state["count"]
    last_date = state["last_date"]
    global_mean = float(state["global_mean"])
    half_life_days = float(state["half_life_days"])
    prior_strength = float(state["prior_strength"])
    rates = np.empty(len(user_ids), dtype=np.float32)
    counts = np.empty(len(user_ids), dtype=np.float32)
    for index, (user_value, date_value) in enumerate(zip(user_ids, dates, strict=True)):
        user = str(user_value)
        factor = (
            _decay(calendar_days_between(int(date_value), int(last_date[user])), half_life_days)
            if user in last_date
            else 1.0
        )
        decayed_positive = float(positive.get(user, 0.0)) * factor
        decayed_count = float(count.get(user, 0.0)) * factor
        rates[index] = (decayed_positive + prior_strength * global_mean) / (
            decayed_count + prior_strength
        )
        counts[index] = decayed_count
    arrays = {"recency_user_rate": rates, "recency_user_count": counts}
    provenance = {
        name: {
            "cutoff": "frozen fit split only; no evaluation labels",
            "half_life_days": half_life_days,
            "prior_strength": prior_strength,
        }
        for name in arrays
    }
    return FeatureBundle(arrays, provenance)
