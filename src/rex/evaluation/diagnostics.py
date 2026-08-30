"""Deterministic segment diagnostics and user-bootstrap uncertainty estimates."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from rex.data.views import FeatureView


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


def user_bootstrap_delta_ci(
    user_ids: np.ndarray,
    labels: np.ndarray,
    candidate_scores: np.ndarray,
    reference_scores: np.ndarray,
    *,
    samples: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Paired user-bootstrap interval for candidate minus reference primary."""

    if candidate_scores.shape != reference_scores.shape:
        raise ValueError("candidate/reference prediction shapes differ")
    candidate = per_user_metrics(user_ids, labels, candidate_scores)
    reference = per_user_metrics(user_ids, labels, reference_scores)
    if [row["user_id"] for row in candidate] != [row["user_id"] for row in reference]:
        raise ValueError("candidate/reference user groups differ")
    if not candidate:
        return {"low": 0.0, "high": 0.0, "mean": 0.0, "probability_positive": 0.0, "samples": 0}
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        indices = rng.integers(0, len(candidate), size=len(candidate))
        candidate_primary = aggregate_user_metrics([candidate[i] for i in indices])["primary"]
        reference_primary = aggregate_user_metrics([reference[i] for i in indices])["primary"]
        deltas[sample] = candidate_primary - reference_primary
    return {
        "low": float(np.quantile(deltas, alpha / 2)),
        "high": float(np.quantile(deltas, 1 - alpha / 2)),
        "mean": float(np.mean(deltas)),
        "probability_positive": float(np.mean(deltas > 0)),
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


def _arrays(view: FeatureView | dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return view.arrays if isinstance(view, FeatureView) else view


def standard_segments(
    evaluation: FeatureView | dict[str, np.ndarray],
    *,
    history: FeatureView | dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Build the standard, deterministic diagnostic segmentation matrix."""

    arrays = _arrays(evaluation)
    rows = len(arrays["row_id"])
    train = _arrays(history) if history is not None else None
    segments: dict[str, np.ndarray] = {"all": np.ones(rows, dtype=bool)}

    history_users = set(map(str, train["user_id"])) if train is not None else set()
    history_videos = set(map(str, train["video_id"])) if train is not None else set()
    history_pairs = (
        set(zip(map(str, train["user_id"]), map(str, train["video_id"]), strict=True))
        if train is not None
        else set()
    )
    history_authors = (
        set(zip(map(str, train["user_id"]), map(str, train["author_id"]), strict=True))
        if train is not None
        else set()
    )
    users = np.asarray([str(value) for value in arrays["user_id"]])
    videos = np.asarray([str(value) for value in arrays["video_id"]])
    authors = np.asarray([str(value) for value in arrays["author_id"]])
    if train is not None:
        warm_user = np.fromiter((value in history_users for value in users), bool, count=rows)
        warm_video = np.fromiter((value in history_videos for value in videos), bool, count=rows)
        repeated = np.fromiter(
            ((user, video) in history_pairs for user, video in zip(users, videos, strict=True)),
            bool,
            count=rows,
        )
        author_affinity = np.fromiter(
            ((user, author) in history_authors for user, author in zip(users, authors, strict=True)),
            bool,
            count=rows,
        )
        segments.update(
            {
                "user:cold": ~warm_user,
                "user:warm": warm_user,
                "video:cold": ~warm_video,
                "video:warm": warm_video,
                "repeat:first": ~repeated,
                "repeat:seen": repeated,
                "author_affinity:new": ~author_affinity,
                "author_affinity:seen": author_affinity,
            }
        )

    if "fx__user_history_length" in arrays:
        lengths = np.asarray(arrays["fx__user_history_length"])
    elif train is not None:
        counts: dict[str, int] = defaultdict(int)
        for user in train["user_id"]:
            counts[str(user)] += 1
        lengths = np.fromiter((counts[value] for value in users), np.int64, count=rows)
    else:
        lengths = np.zeros(rows, dtype=np.int64)
    segments.update(
        {
            "history:0": lengths == 0,
            "history:1-4": (lengths >= 1) & (lengths <= 4),
            "history:5-19": (lengths >= 5) & (lengths <= 19),
            "history:20+": lengths >= 20,
        }
    )

    reference_duration = (
        np.asarray(train["duration_ms"], dtype=np.float64)
        if train is not None
        else np.asarray(arrays["duration_ms"], dtype=np.float64)
    )
    duration = np.asarray(arrays["duration_ms"], dtype=np.float64)
    edges = np.quantile(reference_duration, [0.25, 0.5, 0.75]) if len(reference_duration) else [0, 0, 0]
    buckets = np.searchsorted(edges, duration, side="right")
    for index in range(4):
        segments[f"duration:q{index + 1}"] = buckets == index
    for tab in sorted(set(map(str, arrays["tab"]))):
        segments[f"tab:{tab}"] = np.asarray([str(value) == tab for value in arrays["tab"]])
    for value in sorted(set(map(int, arrays["date"]))):
        segments[f"date:{value}"] = np.asarray(arrays["date"]) == value
    return segments


def standard_segment_report(
    evaluation: FeatureView | dict[str, np.ndarray],
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    history: FeatureView | dict[str, np.ndarray] | None = None,
) -> dict[str, dict[str, float | int]]:
    arrays = _arrays(evaluation)
    return segment_report(
        arrays["user_id"],
        labels,
        scores,
        standard_segments(evaluation, history=history),
    )


def compare_diagnostics(
    evaluation: FeatureView | dict[str, np.ndarray],
    labels: np.ndarray,
    candidate_scores: np.ndarray,
    reference_scores: np.ndarray,
    *,
    history: FeatureView | dict[str, np.ndarray] | None = None,
    bootstrap_samples: int = 500,
    seed: int = 0,
    segment_threshold: float = 0.002,
) -> dict[str, Any]:
    """Create one judge-readable comparison with uncertainty and segment changes."""

    arrays = _arrays(evaluation)
    candidate = aggregate_user_metrics(
        per_user_metrics(arrays["user_id"], labels, candidate_scores)
    )
    reference = aggregate_user_metrics(
        per_user_metrics(arrays["user_id"], labels, reference_scores)
    )
    candidate_segments = standard_segment_report(
        evaluation, labels, candidate_scores, history=history
    )
    reference_segments = standard_segment_report(
        evaluation, labels, reference_scores, history=history
    )
    segment_deltas = {
        name: float(candidate_segments[name]["primary"])
        - float(reference_segments[name]["primary"])
        for name in candidate_segments
        if int(candidate_segments[name]["rows"]) > 0
        and int(reference_segments[name]["rows"]) > 0
    }
    return {
        "candidate": candidate,
        "reference": reference,
        "delta": {name: candidate[name] - reference[name] for name in candidate},
        "candidate_primary_ci": user_bootstrap_ci(
            arrays["user_id"], labels, candidate_scores, samples=bootstrap_samples, seed=seed
        ),
        "primary_delta_ci": user_bootstrap_delta_ci(
            arrays["user_id"],
            labels,
            candidate_scores,
            reference_scores,
            samples=bootstrap_samples,
            seed=seed,
        ),
        "prediction_correlation": prediction_correlation(candidate_scores, reference_scores),
        "candidate_segments": candidate_segments,
        "reference_segments": reference_segments,
        "segment_primary_deltas": segment_deltas,
        "segment_wins": sorted(
            name for name, delta in segment_deltas.items() if delta >= segment_threshold
        ),
        "segment_regressions": sorted(
            name for name, delta in segment_deltas.items() if delta <= -segment_threshold
        ),
    }
