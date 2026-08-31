"""Strict point-in-time candidate-conditioned recency features."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from rex.features.base import FeatureBundle
from rex.features.temporal_order import elapsed_days, exponential_decay, strict_timestamp_groups


DEFAULT_HALF_LIVES = (1.0, 3.0, 7.0, 14.0)


def _validate_inputs(*values: np.ndarray) -> int:
    rows = len(values[0]) if values else 0
    if any(np.asarray(value).ndim != 1 or len(value) != rows for value in values):
        raise ValueError("candidate-recency inputs must be aligned one-dimensional arrays")
    return rows


def _decayed_value(
    values: dict[Any, float],
    last_time: dict[Any, int],
    key: Any,
    current_time: int,
    half_life_days: float,
) -> float:
    value = float(values.get(key, 0.0))
    if key not in last_time:
        return value
    return value * exponential_decay(current_time, last_time[key], half_life_days)


def candidate_recency_features(
    user_ids: np.ndarray,
    video_ids: np.ndarray,
    author_ids: np.ndarray,
    time_ms: np.ndarray,
    source_row_keys: np.ndarray,
    labels: np.ndarray,
    *,
    half_lives_days: tuple[float, ...] = DEFAULT_HALF_LIVES,
    prior_strength: float = 10.0,
    candidate_half_life_days: float = 7.0,
) -> FeatureBundle:
    """Build training-row features without current/equal-time outcomes."""

    rows = _validate_inputs(
        user_ids, video_ids, author_ids, time_ms, source_row_keys, labels
    )
    if prior_strength <= 0 or any(value <= 0 for value in half_lives_days):
        raise ValueError("prior strength and half lives must be positive")
    labels = np.asarray(labels, dtype=np.float64)
    if not np.isfinite(labels).all():
        raise ValueError("candidate-recency labels must be finite")
    times = np.asarray(time_ms, dtype=np.int64)

    global_positive = 0.0
    global_count = 0
    user_positive = [defaultdict(float) for _ in half_lives_days]
    user_count = [defaultdict(float) for _ in half_lives_days]
    user_last_time = [dict() for _ in half_lives_days]
    author_positive: dict[tuple[str, str], float] = defaultdict(float)
    author_count: dict[tuple[str, str], float] = defaultdict(float)
    author_last_time: dict[tuple[str, str], int] = {}
    video_count: dict[tuple[str, str], int] = defaultdict(int)
    video_positive: dict[tuple[str, str], int] = defaultdict(int)
    video_last_outcome: dict[tuple[str, str], float] = {}
    video_last_time: dict[tuple[str, str], int] = {}

    arrays: dict[str, np.ndarray] = {
        "pt_user_author_rate": np.empty(rows, dtype=np.float32),
        "pt_user_author_count": np.empty(rows, dtype=np.float32),
        "pt_user_author_days_since": np.full(rows, -1.0, dtype=np.float32),
        "pt_user_video_count": np.zeros(rows, dtype=np.int32),
        "pt_user_video_positive": np.zeros(rows, dtype=np.int32),
        "pt_user_video_last_outcome": np.full(rows, -1.0, dtype=np.float32),
        "pt_user_video_days_since": np.full(rows, -1.0, dtype=np.float32),
    }
    for half_life in half_lives_days:
        suffix = str(half_life).replace(".", "p")
        arrays[f"pt_user_rate_h{suffix}"] = np.empty(rows, dtype=np.float32)
        arrays[f"pt_user_count_h{suffix}"] = np.empty(rows, dtype=np.float32)

    for group in strict_timestamp_groups(times, source_row_keys):
        current = int(times[group[0]])
        prior_mean = global_positive / global_count if global_count else 0.5
        for index in group:
            user = str(user_ids[index])
            author_key = (user, str(author_ids[index]))
            video_key = (user, str(video_ids[index]))
            decayed_author_positive = _decayed_value(
                author_positive,
                author_last_time,
                author_key,
                current,
                candidate_half_life_days,
            )
            decayed_author_count = _decayed_value(
                author_count,
                author_last_time,
                author_key,
                current,
                candidate_half_life_days,
            )
            arrays["pt_user_author_rate"][index] = (
                decayed_author_positive + prior_strength * prior_mean
            ) / (decayed_author_count + prior_strength)
            arrays["pt_user_author_count"][index] = decayed_author_count
            if author_key in author_last_time:
                arrays["pt_user_author_days_since"][index] = elapsed_days(
                    current, author_last_time[author_key]
                )
            arrays["pt_user_video_count"][index] = video_count[video_key]
            arrays["pt_user_video_positive"][index] = video_positive[video_key]
            arrays["pt_user_video_last_outcome"][index] = video_last_outcome.get(
                video_key, -1.0
            )
            if video_key in video_last_time:
                arrays["pt_user_video_days_since"][index] = elapsed_days(
                    current, video_last_time[video_key]
                )
            for position, half_life in enumerate(half_lives_days):
                decayed_positive = _decayed_value(
                    user_positive[position],
                    user_last_time[position],
                    user,
                    current,
                    half_life,
                )
                decayed_count = _decayed_value(
                    user_count[position],
                    user_last_time[position],
                    user,
                    current,
                    half_life,
                )
                suffix = str(half_life).replace(".", "p")
                arrays[f"pt_user_rate_h{suffix}"][index] = (
                    decayed_positive + prior_strength * prior_mean
                ) / (decayed_count + prior_strength)
                arrays[f"pt_user_count_h{suffix}"][index] = decayed_count

        for index in group:
            user = str(user_ids[index])
            label = float(labels[index])
            author_key = (user, str(author_ids[index]))
            video_key = (user, str(video_ids[index]))
            if author_key in author_last_time:
                factor = exponential_decay(
                    current, author_last_time[author_key], candidate_half_life_days
                )
                author_positive[author_key] *= factor
                author_count[author_key] *= factor
            author_positive[author_key] += label
            author_count[author_key] += 1.0
            author_last_time[author_key] = current
            video_count[video_key] += 1
            video_positive[video_key] += int(label > 0.5)
            video_last_outcome[video_key] = label
            video_last_time[video_key] = current
            for position, half_life in enumerate(half_lives_days):
                if user in user_last_time[position]:
                    factor = exponential_decay(
                        current, user_last_time[position][user], half_life
                    )
                    user_positive[position][user] *= factor
                    user_count[position][user] *= factor
                user_positive[position][user] += label
                user_count[position][user] += 1.0
                user_last_time[position][user] = current
            global_positive += label
            global_count += 1

    provenance = {
        name: {
            "cutoff": "strictly earlier timestamp group",
            "equal_timestamp_outcomes_excluded": True,
            "prior_strength": prior_strength,
            "half_lives_days": list(half_lives_days),
            "candidate_half_life_days": candidate_half_life_days,
        }
        for name in arrays
    }
    return FeatureBundle(arrays, provenance)


def fit_candidate_recency_state(
    user_ids: np.ndarray,
    video_ids: np.ndarray,
    author_ids: np.ndarray,
    time_ms: np.ndarray,
    source_row_keys: np.ndarray,
    labels: np.ndarray,
    *,
    half_lives_days: tuple[float, ...] = DEFAULT_HALF_LIVES,
    prior_strength: float = 10.0,
    candidate_half_life_days: float = 7.0,
) -> dict[str, object]:
    """Fit a frozen state from the complete training partition."""

    _validate_inputs(user_ids, video_ids, author_ids, time_ms, source_row_keys, labels)
    times = np.asarray(time_ms, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float64)
    user_positive = [defaultdict(float) for _ in half_lives_days]
    user_count = [defaultdict(float) for _ in half_lives_days]
    user_last_time = [dict() for _ in half_lives_days]
    author_positive: dict[tuple[str, str], float] = defaultdict(float)
    author_count: dict[tuple[str, str], float] = defaultdict(float)
    author_last_time: dict[tuple[str, str], int] = {}
    video_count: dict[tuple[str, str], int] = defaultdict(int)
    video_positive: dict[tuple[str, str], int] = defaultdict(int)
    video_last_outcome: dict[tuple[str, str], float] = {}
    video_last_time: dict[tuple[str, str], int] = {}
    for group in strict_timestamp_groups(times, source_row_keys):
        current = int(times[group[0]])
        for index in group:
            user = str(user_ids[index])
            label = float(labels[index])
            author_key = (user, str(author_ids[index]))
            video_key = (user, str(video_ids[index]))
            if author_key in author_last_time:
                factor = exponential_decay(
                    current, author_last_time[author_key], candidate_half_life_days
                )
                author_positive[author_key] *= factor
                author_count[author_key] *= factor
            author_positive[author_key] += label
            author_count[author_key] += 1.0
            author_last_time[author_key] = current
            video_count[video_key] += 1
            video_positive[video_key] += int(label > 0.5)
            video_last_outcome[video_key] = label
            video_last_time[video_key] = current
            for position, half_life in enumerate(half_lives_days):
                if user in user_last_time[position]:
                    factor = exponential_decay(
                        current, user_last_time[position][user], half_life
                    )
                    user_positive[position][user] *= factor
                    user_count[position][user] *= factor
                user_positive[position][user] += label
                user_count[position][user] += 1.0
                user_last_time[position][user] = current
    return {
        "global_mean": float(np.mean(labels)) if len(labels) else 0.5,
        "prior_strength": prior_strength,
        "half_lives_days": tuple(half_lives_days),
        "candidate_half_life_days": candidate_half_life_days,
        "user_positive": [dict(value) for value in user_positive],
        "user_count": [dict(value) for value in user_count],
        "user_last_time": user_last_time,
        "author_positive": dict(author_positive),
        "author_count": dict(author_count),
        "author_last_time": author_last_time,
        "video_count": dict(video_count),
        "video_positive": dict(video_positive),
        "video_last_outcome": video_last_outcome,
        "video_last_time": video_last_time,
    }


def apply_candidate_recency_state(
    user_ids: np.ndarray,
    video_ids: np.ndarray,
    author_ids: np.ndarray,
    time_ms: np.ndarray,
    state: dict[str, object],
) -> FeatureBundle:
    """Apply frozen outcome state without consuming apply-partition labels."""

    rows = _validate_inputs(user_ids, video_ids, author_ids, time_ms)
    times = np.asarray(time_ms, dtype=np.int64)
    prior = float(state["prior_strength"])
    global_mean = float(state["global_mean"])
    half_lives = tuple(float(value) for value in state["half_lives_days"])
    candidate_half_life = float(state["candidate_half_life_days"])
    arrays: dict[str, np.ndarray] = {
        "pt_user_author_rate": np.empty(rows, dtype=np.float32),
        "pt_user_author_count": np.empty(rows, dtype=np.float32),
        "pt_user_author_days_since": np.full(rows, -1.0, dtype=np.float32),
        "pt_user_video_count": np.zeros(rows, dtype=np.int32),
        "pt_user_video_positive": np.zeros(rows, dtype=np.int32),
        "pt_user_video_last_outcome": np.full(rows, -1.0, dtype=np.float32),
        "pt_user_video_days_since": np.full(rows, -1.0, dtype=np.float32),
    }
    for half_life in half_lives:
        suffix = str(half_life).replace(".", "p")
        arrays[f"pt_user_rate_h{suffix}"] = np.empty(rows, dtype=np.float32)
        arrays[f"pt_user_count_h{suffix}"] = np.empty(rows, dtype=np.float32)
    for index in range(rows):
        current = int(times[index])
        user = str(user_ids[index])
        author_key = (user, str(author_ids[index]))
        video_key = (user, str(video_ids[index]))
        author_positive = _decayed_value(
            state["author_positive"],
            state["author_last_time"],
            author_key,
            current,
            candidate_half_life,
        )
        author_count = _decayed_value(
            state["author_count"],
            state["author_last_time"],
            author_key,
            current,
            candidate_half_life,
        )
        arrays["pt_user_author_rate"][index] = (
            author_positive + prior * global_mean
        ) / (author_count + prior)
        arrays["pt_user_author_count"][index] = author_count
        if author_key in state["author_last_time"]:
            arrays["pt_user_author_days_since"][index] = elapsed_days(
                current, state["author_last_time"][author_key]
            )
        arrays["pt_user_video_count"][index] = int(
            state["video_count"].get(video_key, 0)
        )
        arrays["pt_user_video_positive"][index] = int(
            state["video_positive"].get(video_key, 0)
        )
        arrays["pt_user_video_last_outcome"][index] = float(
            state["video_last_outcome"].get(video_key, -1.0)
        )
        if video_key in state["video_last_time"]:
            arrays["pt_user_video_days_since"][index] = elapsed_days(
                current, state["video_last_time"][video_key]
            )
        for position, half_life in enumerate(half_lives):
            positive = _decayed_value(
                state["user_positive"][position],
                state["user_last_time"][position],
                user,
                current,
                half_life,
            )
            count = _decayed_value(
                state["user_count"][position],
                state["user_last_time"][position],
                user,
                current,
                half_life,
            )
            suffix = str(half_life).replace(".", "p")
            arrays[f"pt_user_rate_h{suffix}"][index] = (
                positive + prior * global_mean
            ) / (count + prior)
            arrays[f"pt_user_count_h{suffix}"][index] = count
    provenance = {
        name: {
            "cutoff": "frozen fit split only; no apply outcomes",
            "prior_strength": prior,
            "half_lives_days": list(half_lives),
            "candidate_half_life_days": candidate_half_life,
        }
        for name in arrays
    }
    return FeatureBundle(arrays, provenance)
