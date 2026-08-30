from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rex.contracts import ArtifactRef, Metrics
from rex.data.manifest import sha256_file
from rex.data.views import load_feature_view
from rex.evaluation.baseline import OFFICIAL_FM_PLUGIN, reproduce_fm_seed
from rex.evaluation.submission import (
    SubmissionError,
    build_submission,
    validate_submission,
    validate_submission_matches_predictions,
)
from rex.execution.artifacts import artifact_ref, write_prediction_artifact
from rex.execution.sandbox import SandboxMode
from rex.models.bundle import create_model_bundle, validate_model_bundle
from rex.reporting.finalizer import create_final_bundle


def _test_features(path: Path) -> Path:
    np.savez_compressed(
        path,
        row_id=np.arange(3, dtype=np.int64),
        date=np.asarray([20220429, 20220429, 20220430]),
        user_id=np.asarray(["u1", "u1", "u2"]),
        video_id=np.asarray(["v1", "v1", "v2"]),
        author_id=np.asarray(["a1", "a1", "a2"]),
        tab=np.asarray(["1", "1", "2"]),
        duration_ms=np.asarray([10.0, 10.0, 20.0], dtype=np.float32),
    )
    return path


def _raw_data(path: Path) -> Path:
    path.mkdir()
    (path / "video_features_basic_pure.csv").write_text(
        "video_id,author_id\nv1,a1\nv2,a2\n", encoding="utf-8"
    )
    header = "date,user_id,video_id,tab,duration_ms,long_view\n"
    (path / "log_standard_4_08_to_4_21_pure.csv").write_text(header, encoding="utf-8")
    (path / "log_standard_4_22_to_5_08_pure.csv").write_text(
        header
        + "20220429,u1,v1,1,10,0\n"
        + "20220429,u1,v1,1,10,1\n"
        + "20220430,u2,v2,2,20,0\n",
        encoding="utf-8",
    )
    return path


def _metrics() -> Metrics:
    return Metrics(
        GAUC=0.62,
        **{"nDCG@5": 0.58},
        primary=0.60,
        users=2,
        rows=3,
        evaluator_sha256="0" * 64,
        split="valid",
        seed=7,
    )


def test_baseline_emits_the_same_strict_model_bundle_as_candidates(
    feature_target_paths: tuple[Path, Path], tmp_path: Path
) -> None:
    features, targets = feature_target_paths
    views = tmp_path / "views"
    (views / "label_vault").mkdir(parents=True)
    (views / "train_features.npz").write_bytes(features.read_bytes())
    (views / "valid_features.npz").write_bytes(features.read_bytes())
    (views / "label_vault/train_targets.npz").write_bytes(targets.read_bytes())
    (views / "label_vault/valid_targets.npz").write_bytes(targets.read_bytes())

    reproduce_fm_seed(
        views,
        seed=0,
        epochs=1,
        patience=1,
        batch_size=4,
        evidence_dir=tmp_path / "baseline",
    )

    loaded = validate_model_bundle(
        tmp_path / "baseline/model_bundle.json",
        expected_plugin=OFFICIAL_FM_PLUGIN,
        expected_config_sha256=sha256_file(tmp_path / "baseline/config.json"),
        expected_data_view_sha256=sha256_file(views / "train_features.npz"),
        expected_features=load_feature_view(views / "train_features.npz"),
    )
    assert loaded.primary_path.name == "model.npz"
    assert {member.name for member in loaded.manifest.members} == {
        "encoder.json",
        "model.npz",
    }


def test_submission_requires_canonical_alignment_and_explicit_small_test_override(
    tmp_path: Path,
) -> None:
    feature_path = _test_features(tmp_path / "test-features.npz")
    scores = np.asarray([0.1, 0.2, 0.3])
    predictions = write_prediction_artifact(tmp_path / "predictions.npz", feature_path, scores)

    with pytest.raises(SubmissionError, match="expected 170588"):
        build_submission(
            predictions,
            tmp_path / "rejected.csv",
            expected_features=feature_path,
        )

    submission = build_submission(
        predictions,
        tmp_path / "submission.csv",
        expected_features=feature_path,
        expected_rows=3,
    )
    lines = submission.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "row_id,user_id,video_id,score"
    assert lines[1].split(",")[1:3] == ["u1", "v1"]
    assert lines[2].split(",")[1:3] == ["u1", "v1"]
    validate_submission_matches_predictions(
        submission,
        predictions,
        expected_features=feature_path,
        expected_rows=3,
    )


def test_submission_rejects_alignment_drift_and_nonfinite_scores(tmp_path: Path) -> None:
    feature_path = _test_features(tmp_path / "test-features.npz")
    with np.load(feature_path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    np.savez_compressed(
        tmp_path / "misaligned.npz",
        row_id=arrays["row_id"],
        user_id=arrays["user_id"],
        video_id=np.asarray(["v2", "v1", "v2"]),
        score=np.asarray([0.1, 0.2, 0.3]),
    )
    with pytest.raises(SubmissionError, match="alignment mismatch"):
        build_submission(
            tmp_path / "misaligned.npz",
            tmp_path / "bad.csv",
            expected_features=feature_path,
            expected_rows=3,
        )

    np.savez_compressed(
        tmp_path / "nan.npz",
        row_id=arrays["row_id"],
        user_id=arrays["user_id"],
        video_id=arrays["video_id"],
        score=np.asarray([0.1, np.nan, 0.3]),
    )
    with pytest.raises(SubmissionError, match="finite"):
        build_submission(
            tmp_path / "nan.npz",
            tmp_path / "nan.csv",
            expected_features=feature_path,
            expected_rows=3,
        )


def test_organizer_checker_can_only_run_check_mode(tmp_path: Path) -> None:
    feature_path = _test_features(tmp_path / "test-features.npz")
    data_dir = _raw_data(tmp_path / "raw")
    predictions = write_prediction_artifact(
        tmp_path / "predictions.npz",
        feature_path,
        np.asarray([0.1, 0.2, 0.3]),
    )
    submission = build_submission(
        predictions,
        tmp_path / "submission.csv",
        expected_features=feature_path,
        expected_rows=3,
    )

    result = validate_submission(
        submission,
        data_dir=data_dir,
        split="test",
        sandbox_mode=SandboxMode.FIXTURE,
    )

    assert result.valid, result.stderr
    assert result.command[-1] == "--check"
    assert "--score" not in result.command
    assert "--make" not in result.command
    assert result.sandbox_evidence["submit_py_sha256"]


def test_final_bundle_is_checked_twice_content_addressed_and_sealed(tmp_path: Path) -> None:
    feature_path = _test_features(tmp_path / "test-features.npz")
    features = load_feature_view(feature_path)
    data_dir = _raw_data(tmp_path / "raw")
    config = tmp_path / "config.json"
    config.write_text('{"model":"fixture"}\n', encoding="utf-8")
    config_hash = sha256_file(config)
    model_dir = tmp_path / "source-model"
    model_dir.mkdir()
    checkpoint = model_dir / "model.bin"
    checkpoint.write_bytes(b"immutable winner")
    sidecar = model_dir / "sidecar.json"
    sidecar.write_text('{"version":1}\n', encoding="utf-8")
    bundle = create_model_bundle(
        model_dir,
        checkpoint,
        plugin="tests:FixtureModel",
        seed=7,
        commit_sha="winner-commit",
        config_sha256=config_hash,
        data_view_sha256=features.sha256,
        features=features,
        member_paths=(checkpoint, sidecar),
    )
    predictions = write_prediction_artifact(
        tmp_path / "test-predictions.npz",
        features,
        np.asarray([0.1, 0.2, 0.3]),
    )
    submission = build_submission(
        predictions,
        tmp_path / "submission.csv",
        expected_features=features,
        expected_rows=3,
    )
    report = tmp_path / "report.json"
    report.write_text('{"test_scored":false}\n', encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"artifacts":[]}\n', encoding="utf-8")

    result = create_final_bundle(
        tmp_path / "sealed-final",
        run_id="run-1",
        experiment_id="winner-1",
        submission=artifact_ref(submission, "submission"),
        model_bundle=artifact_ref(bundle, "model_bundle"),
        predictions=artifact_ref(predictions, "test_predictions"),
        config_path=config,
        report=artifact_ref(report, "report"),
        evidence_index=artifact_ref(evidence, "evidence_index"),
        metrics=_metrics(),
        commit_sha="winner-commit",
        config_sha256=config_hash,
        expected_test_features=features,
        data_dir=data_dir,
        expected_rows=3,
        checker_sandbox_mode=SandboxMode.FIXTURE,
    )

    payload = json.loads(Path(result.path).read_text(encoding="utf-8"))
    assert payload["sealed"] is True
    assert payload["test_rows"] == 3
    assert payload["test_scored"] is False
    assert payload["organizer_checks"] == 2
    assert (tmp_path / "sealed-final/checks/source.json").is_file()
    assert (tmp_path / "sealed-final/checks/copied.json").is_file()
    validate_model_bundle(
        tmp_path / "sealed-final/model/model_bundle.json",
        expected_commit_sha="winner-commit",
        expected_config_sha256=config_hash,
        expected_features=features,
    )
    for raw in payload["artifacts"].values():
        ref = ArtifactRef.model_validate(raw)
        assert sha256_file(ref.path) == ref.sha256

    replay = create_final_bundle(
        tmp_path / "sealed-final",
        run_id="run-1",
        experiment_id="winner-1",
        submission=artifact_ref(submission, "submission"),
        model_bundle=artifact_ref(bundle, "model_bundle"),
        predictions=artifact_ref(predictions, "test_predictions"),
        config_path=config,
        report=artifact_ref(report, "report"),
        evidence_index=artifact_ref(evidence, "evidence_index"),
        metrics=_metrics(),
        commit_sha="winner-commit",
        config_sha256=config_hash,
        expected_test_features=features,
        data_dir=data_dir,
        expected_rows=3,
        checker_sandbox_mode=SandboxMode.FIXTURE,
    )
    assert replay.sha256 == result.sha256
