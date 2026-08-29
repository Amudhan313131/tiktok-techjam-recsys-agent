"""Within-user rank normalization and non-negative model blending."""

from __future__ import annotations

import numpy as np

from rex.data.groups import user_groups


def per_user_percentile_ranks(user_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    result = np.empty(len(scores), dtype=np.float64)
    for indices in user_groups(user_ids).values():
        values = np.asarray(scores)[indices]
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(indices), dtype=np.float64)
        ranks[order] = np.arange(len(indices), dtype=np.float64)
        result[indices] = ranks / max(1, len(indices) - 1)
    return result


def per_user_standardize(user_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    result = np.empty(len(scores), dtype=np.float64)
    for indices in user_groups(user_ids).values():
        values = np.asarray(scores, dtype=np.float64)[indices]
        std = values.std()
        result[indices] = (values - values.mean()) / std if std > 0 else 0.0
    return result


def blend_scores(
    user_ids: np.ndarray,
    predictions: list[np.ndarray],
    weights: np.ndarray,
    *,
    normalization: str = "percentile",
) -> np.ndarray:
    if not predictions:
        raise ValueError("at least one prediction vector is required")
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) != len(predictions) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("ensemble weights must be non-negative and match predictions")
    weights /= weights.sum()
    normalize = per_user_percentile_ranks if normalization == "percentile" else per_user_standardize
    normalized = np.column_stack([normalize(user_ids, values) for values in predictions])
    return normalized @ weights
