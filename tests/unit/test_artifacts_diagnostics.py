from __future__ import annotations

import numpy as np
import pytest

from rex.evaluation.diagnostics import aggregate_user_metrics, per_user_metrics, prediction_correlation, user_bootstrap_ci
from rex.execution.artifacts import ArtifactError, load_prediction_artifact, write_prediction_artifact


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
