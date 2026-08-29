from __future__ import annotations

import csv

import numpy as np

from rex.data.views import load_target_view
from rex.evaluation.official_adapter import evaluate_predictions
from rex.evaluation.submission import build_submission
from rex.execution.artifacts import write_prediction_artifact


def test_perfect_predictions_score_one_through_frozen_evaluator(feature_target_paths, tmp_path) -> None:
    features, targets = feature_target_paths
    labels = load_target_view(targets).labels
    prediction = write_prediction_artifact(tmp_path / "pred.npz", features, labels)
    metrics = evaluate_predictions(features, targets, prediction, split="valid", seed=0)
    assert metrics.GAUC == 1.0
    assert metrics.ndcg5 == 1.0
    assert metrics.primary == 1.0


def test_submission_builder_preserves_duplicate_rows(feature_target_paths, tmp_path) -> None:
    features, _ = feature_target_paths
    prediction = write_prediction_artifact(tmp_path / "pred.npz", features, np.arange(8))
    csv_path = build_submission(prediction, tmp_path / "submission.csv")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["row_id", "user_id", "video_id", "score"]
    assert len(rows) == 9
    assert rows[1][1:3] == ["u1", "v1"]
