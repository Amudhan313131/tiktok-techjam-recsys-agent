"""Deterministic segment diagnostics and user-bootstrap uncertainty estimates."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def _user_groups(user_ids: np.ndarray) -> dict[str, np.ndarray]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, user_id in enumerate(user_ids):
        buckets[str(user_id)].append(index)
    return {key: np.asarray(value, dtype=np.int64) for key, value in buckets.items()}


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = int(labels.sum())
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def _ndcg5(labels: np.ndarray, scores: np.ndarray) -> float:
    ranked = labels[np.argsort(-scores, kind="mergesort")[:5]]
    discount = 1.0 / np.log2(np.arange(2, len(ranked) + 2))
    dcg = float((ranked * discount).sum())
    ideal = np.sort(labels)[::-1][:5]
    idcg = float((ideal * discount[: len(ideal)]).sum())
    return 0.0 if idcg == 0 else dcg / idcg


def per_user_metrics(
    user_ids: np.ndarray, labels: np.ndarray, scores: np.ndarray
) -> list[dict[str, Any]]:
    """Return sufficient per-user statistics for exact aggregation and bootstrap."""
    result: list[dict[str, Any]] = []
    for user_id, indices in sorted(_user_groups(user_ids).items()):
        user_labels = labels[indices].astype(np.int8, copy=False)
        user_scores = scores[indices].astype(np.float64, copy=False)
        positives = int(user_labels.sum())
        eligible = 0 < positives < len(indices)
        result.append(
            {
                "user_id": user_id,
                "rows": len(indices),
                "positives": positives,
                "gauc_weight": positives if eligible else 0,
                "auc": _auc(user_labels, user_scores),
                "ndcg5": _ndcg5(user_labels, user_scores),
            }
        )
    return result


def aggregate_user_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    items = list(rows)
    weight = sum(int(item["gauc_weight"]) for item in items)
    gauc = (
        sum(float(item["auc"]) * int(item["gauc_weight"]) for item in items) / weight
        if weight
        else 0.5
    )
    ndcg = float(np.mean([float(item["ndcg5"]) for item in items])) if items else 0.0
    return {"GAUC": gauc, "nDCG@5": ndcg, "primary": (gauc + ndcg) / 2.0}


def user_bootstrap_ci(
    user_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    samples: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    per_user = per_user_metrics(user_ids, labels, scores)
    if not per_user:
        return {"low": 0.0, "high": 0.0, "samples": 0}
    rng = np.random.default_rng(seed)
    primaries = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        indices = rng.integers(0, len(per_user), size=len(per_user))
        primaries[sample] = aggregate_user_metrics([per_user[i] for i in indices])["primary"]
    return {
        "low": float(np.quantile(primaries, alpha / 2)),
        "high": float(np.quantile(primaries, 1 - alpha / 2)),
        "samples": samples,
    }


def segment_report(
    user_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    segments: dict[str, np.ndarray],
) -> dict[str, dict[str, float | int]]:
    report: dict[str, dict[str, float | int]] = {}
    for name, mask in sorted(segments.items()):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != labels.shape:
            raise ValueError(f"segment {name!r} has shape {mask.shape}, expected {labels.shape}")
        metrics = aggregate_user_metrics(per_user_metrics(user_ids[mask], labels[mask], scores[mask]))
        report[name] = {**metrics, "rows": int(mask.sum()), "users": len(set(user_ids[mask].tolist()))}
    return report


def prediction_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("prediction arrays must have identical shapes")
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])
