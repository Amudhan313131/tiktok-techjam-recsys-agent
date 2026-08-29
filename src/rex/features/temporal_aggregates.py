"""Leakage-safe expanding and train-to-future aggregate features."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from rex.data.temporal import date_batches
from rex.features.base import FeatureBundle


def expanding_target_rate(
    keys: np.ndarray,
    dates: np.ndarray,
    row_ids: np.ndarray,
    labels: np.ndarray,
    *,
    prior_strength: float = 20.0,
) -> FeatureBundle:
    if not (len(keys) == len(dates) == len(row_ids) == len(labels)):
        raise ValueError("aggregate inputs have different lengths")
    positives: dict[str, float] = defaultdict(float)
    impressions: dict[str, int] = defaultdict(int)
    global_positive = 0.0
    global_count = 0
    rates = np.empty(len(labels), dtype=np.float32)
    counts = np.empty(len(labels), dtype=np.int32)
    for batch in date_batches(dates, row_ids):
        prior_mean = global_positive / global_count if global_count else 0.5
        for index in batch:
            key = str(keys[index])
            rates[index] = (positives[key] + prior_strength * prior_mean) / (
                impressions[key] + prior_strength
            )
            counts[index] = impressions[key]
        for index in batch:
            key = str(keys[index])
            positives[key] += float(labels[index])
            impressions[key] += 1
            global_positive += float(labels[index])
            global_count += 1
    cutoff = "strictly earlier date; same-day rows excluded"
    return FeatureBundle(
        arrays={"target_rate": rates, "prior_impressions": counts},
        provenance={
            "target_rate": {"cutoff": cutoff, "prior_strength": prior_strength},
            "prior_impressions": {"cutoff": cutoff},
        },
    )


def fit_entity_statistics(
    keys: np.ndarray,
    labels: np.ndarray,
    *,
    prior_strength: float = 20.0,
) -> dict[str, object]:
    global_mean = float(np.mean(labels)) if len(labels) else 0.0
    positive: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    for key, label in zip(keys, labels, strict=True):
        normalized = str(key)
        positive[normalized] += float(label)
        count[normalized] += 1
    return {
        "global_mean": global_mean,
        "prior_strength": prior_strength,
        "positive": dict(positive),
        "count": dict(count),
    }


def apply_entity_statistics(keys: np.ndarray, statistics: dict[str, object]) -> FeatureBundle:
    global_mean = float(statistics["global_mean"])
    prior = float(statistics["prior_strength"])
    positive = statistics["positive"]
    count = statistics["count"]
    rates = np.empty(len(keys), dtype=np.float32)
    counts = np.empty(len(keys), dtype=np.int32)
    for index, key in enumerate(keys):
        normalized = str(key)
        n = int(count.get(normalized, 0))
        rates[index] = (float(positive.get(normalized, 0.0)) + prior * global_mean) / (n + prior)
        counts[index] = n
    cutoff = "fit split only; no evaluation labels"
    return FeatureBundle(
        arrays={"target_rate": rates, "prior_impressions": counts},
        provenance={
            "target_rate": {"cutoff": cutoff, "prior_strength": prior},
            "prior_impressions": {"cutoff": cutoff},
        },
    )
