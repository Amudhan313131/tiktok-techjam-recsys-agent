from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import rex.reporting.finalizer as finalizer
from rex.contracts import Metrics
from rex.data.manifest import sha256_file
from rex.data.views import load_feature_view
from rex.execution.artifacts import artifact_ref, write_prediction_artifact
from rex.models.bundle import create_model_bundle, validate_model_bundle
from rex.reporting.finalizer import create_best_valid_bundle


def test_best_valid_bundle_never_creates_a_test_submission(
    feature_target_paths: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_path, _ = feature_target_paths
    features = load_feature_view(feature_path)
    config = tmp_path / "experiment.json"
    config.write_text('{"model":"fixture"}\n', encoding="utf-8")
    config_hash = sha256_file(config)
    bundle_dir = tmp_path / "source-model"
    bundle_dir.mkdir()
    primary = bundle_dir / "model.bin"
    primary.write_bytes(b"trusted-validation-model")
    model_bundle = create_model_bundle(
        bundle_dir,
        primary,
        plugin="tests:FixtureModel",
        seed=7,
        commit_sha="candidate-commit",
        config_sha256=config_hash,
        data_view_sha256=features.sha256,
        features=features,
    )
    predictions = write_prediction_artifact(
        tmp_path / "valid-predictions.npz",
        features,
        np.linspace(-1.0, 1.0, features.rows),
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"split":"valid"}\n', encoding="utf-8")
    metrics = Metrics(
        GAUC=0.62,
        **{"nDCG@5": 0.58},
        primary=0.60,
        users=4,
        rows=8,
        evaluator_sha256="0" * 64,
        split="valid",
        seed=7,
    )

    seal_calls: list[tuple[Path, Path]] = []
    original_seal = finalizer._seal

    def record_seal(stage: Path, output: Path) -> None:
        seal_calls.append((stage, output))
        original_seal(stage, output)

    monkeypatch.setattr(finalizer, "_seal", record_seal)
    arguments = dict(
        output_dir=tmp_path / "best-valid",
        run_id="production-run",
        experiment_id="candidate-001",
        model_bundle=artifact_ref(model_bundle, "model_bundle"),
        valid_predictions=artifact_ref(predictions, "predictions"),
        evidence_index=artifact_ref(evidence, "evidence_index"),
        metrics=metrics,
        commit_sha="candidate-commit",
        config_path=config,
        config_sha256=config_hash,
    )
    result = create_best_valid_bundle(
        **arguments,
    )
    replay = create_best_valid_bundle(**arguments)

    manifest = json.loads(Path(result.path).read_text(encoding="utf-8"))
    assert manifest["kind"] == "best_valid"
    assert replay.sha256 == result.sha256
    assert len(seal_calls) == 1
    assert manifest["test_prediction_created"] is False
    assert not (tmp_path / "best-valid" / "submission.csv").exists()
    copied = validate_model_bundle(
        tmp_path / "best-valid" / "model" / "model_bundle.json",
        expected_commit_sha="candidate-commit",
        expected_config_sha256=config_hash,
    )
    assert copied.primary_path.read_bytes() == b"trusted-validation-model"
