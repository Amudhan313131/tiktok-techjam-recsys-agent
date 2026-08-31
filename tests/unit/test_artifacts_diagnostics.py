from __future__ import annotations

import numpy as np
import pytest

from rex.evaluation.diagnostics import (
    aggregate_user_metrics,
    compare_diagnostics,
    per_user_metrics,
    prediction_correlation,
    pool_bootstrap_delta_evidence,
    passes_uncertainty_gate,
    user_bootstrap_ci,
    user_bootstrap_delta_ci,
)
from rex.execution.artifacts import (
    ArtifactError,
    load_prediction_artifact,
    write_prediction_artifact,
)


def test_prediction_roundtrip_and_alignment(feature_target_paths, tmp_path) -> None:
    features, _ = feature_target_paths
    path = write_prediction_artifact(tmp_path / "pred.npz", features, np.arange(8, dtype=float))
    loaded = load_prediction_artifact(path, features)
    assert loaded["score"].tolist() == list(range(8))


def test_nonfinite_prediction_is_rejected(feature_target_paths, tmp_path) -> None:
    features, _ = feature_target_paths
    with pytest.raises(ArtifactError):
        write_prediction_artifact(tmp_path / "pred.npz", features, np.full(8, np.nan))


def test_user_metrics_are_exact_on_perfect_ranking() -> None:
    users = np.asarray(["a", "a", "b", "b"])
    labels = np.asarray([1, 0, 1, 0])
    scores = np.asarray([1.0, 0.0, 2.0, -1.0])
    metrics = aggregate_user_metrics(per_user_metrics(users, labels, scores))
    assert metrics == {"GAUC": 1.0, "nDCG@5": 1.0, "primary": 1.0}


def test_bootstrap_is_deterministic() -> None:
    users = np.asarray(["a", "a", "b", "b"])
    labels = np.asarray([1, 0, 0, 1])
    scores = np.asarray([1.0, 0.0, 1.0, 0.0])
    assert user_bootstrap_ci(users, labels, scores, samples=20, seed=4) == user_bootstrap_ci(
        users, labels, scores, samples=20, seed=4
    )


def test_prediction_correlation_handles_constant_arrays() -> None:
    assert prediction_correlation(np.ones(3), np.ones(3)) == 0.0


def test_paired_bootstrap_and_comparison_detect_improvement(feature_target_paths) -> None:
    features, targets = feature_target_paths
    from rex.data.views import load_feature_view, load_target_view

    view = load_feature_view(features)
    labels = load_target_view(targets).labels
    candidate = labels.copy()
    reference = 1.0 - labels
    interval = user_bootstrap_delta_ci(
        view.arrays["user_id"], labels, candidate, reference, samples=20, seed=3
    )
    assert interval["low"] > 0
    comparison = compare_diagnostics(
        view,
        labels,
        candidate,
        reference,
        history=view,
        bootstrap_samples=20,
        seed=3,
    )
    assert comparison["delta"]["primary"] > 0
    assert comparison["primary_delta_ci"]["probability_positive"] == 1.0
    assert comparison["primary_delta_ci"]["std"] >= 0.0
    assert comparison["segment_support"]["all"]["rows"] == len(labels)


def test_temporal_bootstrap_pooling_is_support_weighted_and_gateable() -> None:
    pooled = pool_bootstrap_delta_evidence(
        [
            {
                "mean": 0.002,
                "low": 0.001,
                "high": 0.003,
                "std": 0.0005,
                "probability_positive": 0.99,
                "samples": 500,
                "users": 100,
            },
            {
                "mean": 0.001,
                "low": 0.0002,
                "high": 0.0018,
                "std": 0.0004,
                "probability_positive": 0.98,
                "samples": 500,
                "users": 200,
            },
        ]
    )
    assert pooled["folds"] == 2
    assert pooled["users"] == 300
    assert 0.001 < pooled["mean"] < 0.002
    assert passes_uncertainty_gate(pooled, minimum_probability_positive=0.90)
    assert not passes_uncertainty_gate({}, minimum_probability_positive=0.90)
