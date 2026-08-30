"""Cheap point-in-time user-author and user-duration history summaries."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from rex.data.temporal import date_batches
from rex.features.base import FeatureBundle


def candidate_history_summaries(
    user_ids: np.ndarray,
    author_ids: np.ndarray,
    durations_ms: np.ndarray,
    dates: np.ndarray,
    row_ids: np.ndarray,
    labels: np.ndarray,
    *,
    prior_strength: float = 10.0,
) -> FeatureBundle:
    rows = len(user_ids)
    global_positive = 0.0
    global_count = 0
    author_pos: dict[tuple[str, str], float] = defaultdict(float)
    author_count: dict[tuple[str, str], int] = defaultdict(int)
    duration_pos: dict[tuple[str, int], float] = defaultdict(float)
    duration_count: dict[tuple[str, int], int] = defaultdict(int)
    author_rate = np.empty(rows, dtype=np.float32)
    duration_rate = np.empty(rows, dtype=np.float32)
    history_length: dict[str, int] = defaultdict(int)
    history = np.empty(rows, dtype=np.int32)
    buckets = np.minimum((np.asarray(durations_ms) // 3000).astype(np.int32), 20)
    for batch in date_batches(dates, row_ids):
        prior_mean = global_positive / global_count if global_count else 0.5
        for index in batch:
            user = str(user_ids[index])
            author_key = (user, str(author_ids[index]))
            duration_key = (user, int(buckets[index]))
            author_rate[index] = (author_pos[author_key] + prior_strength * prior_mean) / (
                author_count[author_key] + prior_strength
            )
            duration_rate[index] = (duration_pos[duration_key] + prior_strength * prior_mean) / (
                duration_count[duration_key] + prior_strength
            )
            history[index] = history_length[user]
        for index in batch:
            user = str(user_ids[index])
            author_key = (user, str(author_ids[index]))
            duration_key = (user, int(buckets[index]))
            author_pos[author_key] += float(labels[index])
            author_count[author_key] += 1
            duration_pos[duration_key] += float(labels[index])
            duration_count[duration_key] += 1
            history_length[user] += 1
            global_positive += float(labels[index])
            global_count += 1
    cutoff = "strictly earlier date in user history"
    arrays = {
        "user_author_rate": author_rate,
        "user_duration_rate": duration_rate,
        "user_history_length": history,
    }
    return FeatureBundle(arrays, {name: {"cutoff": cutoff} for name in arrays})


def fit_candidate_history_state(
    user_ids: np.ndarray,
    author_ids: np.ndarray,
    durations_ms: np.ndarray,
    labels: np.ndarray,
    *,
    prior_strength: float = 10.0,
) -> dict[str, object]:
    """Fit frozen user-candidate summaries from training history only."""

    author_pos: dict[tuple[str, str], float] = defaultdict(float)
    author_count: dict[tuple[str, str], int] = defaultdict(int)
    duration_pos: dict[tuple[str, int], float] = defaultdict(float)
    duration_count: dict[tuple[str, int], int] = defaultdict(int)
    history_length: dict[str, int] = defaultdict(int)
    buckets = np.minimum((np.asarray(durations_ms) // 3000).astype(np.int32), 20)
    for user, author, bucket, label in zip(
        user_ids, author_ids, buckets, labels, strict=True
    ):
        normalized_user = str(user)
        author_key = (normalized_user, str(author))
        duration_key = (normalized_user, int(bucket))
        author_pos[author_key] += float(label)
        author_count[author_key] += 1
        duration_pos[duration_key] += float(label)
        duration_count[duration_key] += 1
        history_length[normalized_user] += 1
    return {
        "global_mean": float(np.mean(labels)) if len(labels) else 0.5,
        "prior_strength": prior_strength,
        "author_pos": dict(author_pos),
        "author_count": dict(author_count),
        "duration_pos": dict(duration_pos),
        "duration_count": dict(duration_count),
        "history_length": dict(history_length),
    }


def apply_candidate_history_state(
    user_ids: np.ndarray,
    author_ids: np.ndarray,
    durations_ms: np.ndarray,
    state: dict[str, object],
) -> FeatureBundle:
    """Apply frozen training summaries without updating from evaluation rows."""

    prior_mean = float(state["global_mean"])
    prior_strength = float(state["prior_strength"])
    author_pos = state["author_pos"]
    author_count = state["author_count"]
    duration_pos = state["duration_pos"]
    duration_count = state["duration_count"]
    history_length = state["history_length"]
    author_rate = np.empty(len(user_ids), dtype=np.float32)
    duration_rate = np.empty(len(user_ids), dtype=np.float32)
    history = np.empty(len(user_ids), dtype=np.int32)
    buckets = np.minimum((np.asarray(durations_ms) // 3000).astype(np.int32), 20)
    for index, (user, author, bucket) in enumerate(
        zip(user_ids, author_ids, buckets, strict=True)
    ):
        normalized_user = str(user)
        author_key = (normalized_user, str(author))
        duration_key = (normalized_user, int(bucket))
        author_rate[index] = (
            float(author_pos.get(author_key, 0.0)) + prior_strength * prior_mean
        ) / (int(author_count.get(author_key, 0)) + prior_strength)
        duration_rate[index] = (
            float(duration_pos.get(duration_key, 0.0)) + prior_strength * prior_mean
        ) / (int(duration_count.get(duration_key, 0)) + prior_strength)
        history[index] = int(history_length.get(normalized_user, 0))
    arrays = {
        "user_author_rate": author_rate,
        "user_duration_rate": duration_rate,
        "user_history_length": history,
    }
    cutoff = "frozen fit split only; no evaluation labels"
    return FeatureBundle(arrays, {name: {"cutoff": cutoff} for name in arrays})
