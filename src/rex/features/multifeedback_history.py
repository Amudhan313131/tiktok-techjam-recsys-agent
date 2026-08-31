"""Leakage-safe historical aggregates from prior feedback channels."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from rex.features.base import FeatureBundle
from rex.features.temporal_order import strict_timestamp_groups


ENTITY_FEEDBACK: dict[str, tuple[str, ...]] = {
    "user": ("is_click", "is_like", "is_follow", "is_hate", "long_view"),
    "user_author": ("is_click", "is_like", "long_view"),
    "video": ("is_click", "long_view"),
    "author": ("is_click", "long_view"),
}


def _short_name(value: str) -> str:
    return value[3:] if value.startswith("is_") else value


def _entity_key(
    entity: str,
    user: object,
    video: object,
    author: object,
) -> str | tuple[str, str]:
    if entity == "user":
        return str(user)
    if entity == "video":
        return str(video)
    if entity == "author":
        return str(author)
    if entity == "user_author":
        return str(user), str(author)
    raise ValueError(f"unknown feedback entity: {entity}")


def _validate(
    user_ids: np.ndarray,
    video_ids: np.ndarray,
    author_ids: np.ndarray,
    time_ms: np.ndarray,
    source_row_keys: np.ndarray,
    feedback: dict[str, np.ndarray],
) -> int:
    rows = len(user_ids)
    values = (user_ids, video_ids, author_ids, time_ms, source_row_keys)
    if any(np.asarray(value).ndim != 1 or len(value) != rows for value in values):
        raise ValueError("multi-feedback inputs must be aligned one-dimensional arrays")
    required = {name for names in ENTITY_FEEDBACK.values() for name in names}
    missing = required.difference(feedback)
    if missing:
        raise ValueError(f"multi-feedback labels are missing: {sorted(missing)}")
    for name in required:
        labels = np.asarray(feedback[name])
        if labels.ndim != 1 or len(labels) != rows or not np.isfinite(labels).all():
            raise ValueError(f"feedback label {name} is non-finite or misaligned")
        if not np.isin(labels, (0, 1)).all():
            raise ValueError(f"feedback label {name} must be binary")
    return rows


def _empty_arrays(rows: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for entity, feedback_names in ENTITY_FEEDBACK.items():
        arrays[f"pt_feedback_{entity}_count"] = np.zeros(rows, dtype=np.int32)
        for feedback_name in feedback_names:
            arrays[
                f"pt_feedback_{entity}_{_short_name(feedback_name)}_rate"
            ] = np.empty(rows, dtype=np.float32)
    return arrays


def multifeedback_history_features(
    user_ids: np.ndarray,
    video_ids: np.ndarray,
    author_ids: np.ndarray,
    time_ms: np.ndarray,
    source_row_keys: np.ndarray,
    feedback: dict[str, np.ndarray],
    *,
    prior_strength: float = 20.0,
) -> FeatureBundle:
    """Build compact prior-feedback rates for training rows."""

    rows = _validate(
        user_ids, video_ids, author_ids, time_ms, source_row_keys, feedback
    )
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")
    times = np.asarray(time_ms, dtype=np.int64)
    arrays = _empty_arrays(rows)
    counts: dict[str, dict[Any, int]] = {
        entity: defaultdict(int) for entity in ENTITY_FEEDBACK
    }
    positives: dict[str, dict[str, dict[Any, float]]] = {
        entity: {
            name: defaultdict(float) for name in feedback_names
        }
        for entity, feedback_names in ENTITY_FEEDBACK.items()
    }
    global_positive = {name: 0.0 for name in {n for v in ENTITY_FEEDBACK.values() for n in v}}
    global_count = 0

    for group in strict_timestamp_groups(times, source_row_keys):
        priors = {
            name: global_positive[name] / global_count if global_count else 0.5
            for name in global_positive
        }
        for index in group:
            for entity, feedback_names in ENTITY_FEEDBACK.items():
                key = _entity_key(
                    entity, user_ids[index], video_ids[index], author_ids[index]
                )
                count = counts[entity][key]
                arrays[f"pt_feedback_{entity}_count"][index] = count
                for feedback_name in feedback_names:
                    arrays[
                        f"pt_feedback_{entity}_{_short_name(feedback_name)}_rate"
                    ][index] = (
                        positives[entity][feedback_name][key]
                        + prior_strength * priors[feedback_name]
                    ) / (count + prior_strength)
        for index in group:
            for entity, feedback_names in ENTITY_FEEDBACK.items():
                key = _entity_key(
                    entity, user_ids[index], video_ids[index], author_ids[index]
                )
                counts[entity][key] += 1
                for feedback_name in feedback_names:
                    positives[entity][feedback_name][key] += float(
                        feedback[feedback_name][index]
                    )
            for feedback_name in global_positive:
                global_positive[feedback_name] += float(feedback[feedback_name][index])
            global_count += 1

    provenance = {
        name: {
            "cutoff": "strictly earlier timestamp group",
            "equal_timestamp_outcomes_excluded": True,
            "prior_strength": prior_strength,
            "feedback_sources": sorted(global_positive),
        }
        for name in arrays
    }
    return FeatureBundle(arrays, provenance)


def fit_multifeedback_state(
    user_ids: np.ndarray,
    video_ids: np.ndarray,
    author_ids: np.ndarray,
    time_ms: np.ndarray,
    source_row_keys: np.ndarray,
    feedback: dict[str, np.ndarray],
    *,
    prior_strength: float = 20.0,
) -> dict[str, object]:
    """Fit final aggregate state from the authorized training partition."""

    _validate(user_ids, video_ids, author_ids, time_ms, source_row_keys, feedback)
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")
    counts: dict[str, dict[Any, int]] = {
        entity: defaultdict(int) for entity in ENTITY_FEEDBACK
    }
    positives: dict[str, dict[str, dict[Any, float]]] = {
        entity: {
            name: defaultdict(float) for name in feedback_names
        }
        for entity, feedback_names in ENTITY_FEEDBACK.items()
    }
    times = np.asarray(time_ms, dtype=np.int64)
    for group in strict_timestamp_groups(times, source_row_keys):
        for index in group:
            for entity, feedback_names in ENTITY_FEEDBACK.items():
                key = _entity_key(
                    entity, user_ids[index], video_ids[index], author_ids[index]
                )
                counts[entity][key] += 1
                for feedback_name in feedback_names:
                    positives[entity][feedback_name][key] += float(
                        feedback[feedback_name][index]
                    )
    return {
        "prior_strength": prior_strength,
        "counts": {entity: dict(value) for entity, value in counts.items()},
        "positives": {
            entity: {name: dict(value) for name, value in feedback_values.items()}
            for entity, feedback_values in positives.items()
        },
        "global_means": {
            name: float(np.mean(feedback[name])) if len(feedback[name]) else 0.5
            for name in {n for v in ENTITY_FEEDBACK.values() for n in v}
        },
    }


def apply_multifeedback_state(
    user_ids: np.ndarray,
    video_ids: np.ndarray,
    author_ids: np.ndarray,
    state: dict[str, object],
) -> FeatureBundle:
    """Apply frozen feedback state without accepting evaluation outcomes."""

    rows = len(user_ids)
    if any(
        np.asarray(value).ndim != 1 or len(value) != rows
        for value in (video_ids, author_ids)
    ):
        raise ValueError("multi-feedback apply inputs are misaligned")
    prior_strength = float(state["prior_strength"])
    counts = state["counts"]
    positives = state["positives"]
    global_means = state["global_means"]
    arrays = _empty_arrays(rows)
    for index in range(rows):
        for entity, feedback_names in ENTITY_FEEDBACK.items():
            key = _entity_key(entity, user_ids[index], video_ids[index], author_ids[index])
            count = int(counts[entity].get(key, 0))
            arrays[f"pt_feedback_{entity}_count"][index] = count
            for feedback_name in feedback_names:
                arrays[
                    f"pt_feedback_{entity}_{_short_name(feedback_name)}_rate"
                ][index] = (
                    float(positives[entity][feedback_name].get(key, 0.0))
                    + prior_strength * float(global_means[feedback_name])
                ) / (count + prior_strength)
    provenance = {
        name: {
            "cutoff": "frozen fit split only; no apply outcomes",
            "prior_strength": prior_strength,
            "feedback_sources": sorted(global_means),
        }
        for name in arrays
    }
    return FeatureBundle(arrays, provenance)
