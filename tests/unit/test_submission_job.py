from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import rex.submission.coordinator as coordinator_module
from rex.contracts import AttemptStatus, Metrics, RunRequest, RunResult
from rex.data.manifest import sha256_file
from rex.data.views import load_feature_view
from rex.execution.artifacts import artifact_ref, write_prediction_artifact
from rex.models.bundle import create_model_bundle
from rex.reporting.finalizer import create_best_valid_bundle
from rex.store.db import Database
from rex.submission.coordinator import (
    CheckResult,
    FinalSubmissionCoordinator,
    SubmissionCoordinatorError,
    SubmissionDependencies,
    SubmissionJobConfig,
    build_aligned_csv,
)
from rex.submission.repository import (
    SubmissionRepository,
    SubmissionRepositoryError,
    SubmissionState,
)


def _git(cwd: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=cwd, text=True, capture_output=True, check=True
    )
    return completed.stdout.strip()


@dataclass
class SourceFixture:
    project: Path
    production_db: Path
    run_id: str
    commit: str
    test_features: Path
    test_hash: str


def _make_view(path: Path, *, prefix: str) -> Path:
    np.savez_compressed(
        path,
        row_id=np.arange(3, dtype=np.int64),
        user_id=np.asarray([f"{prefix}-u1", f"{prefix}-u1", f"{prefix}-u2"]),
        video_id=np.asarray(["v1", "v1", "v2"]),
        author_id=np.asarray(["a1", "a1", "a2"]),
        tab=np.asarray(["1", "1", "1"]),
        date=np.asarray([20220408, 20220409, 20220410], dtype=np.int64),
        duration_ms=np.asarray([10, 20, 30], dtype=np.float32),
    )
    return path


def _source_fixture(tmp_path: Path, *, state: str = "COMPLETE") -> SourceFixture:
    project = tmp_path / "project"
    project.mkdir()
    _git(project, "init")
    _git(project, "config", "user.email", "test@example.com")
    _git(project, "config", "user.name", "Submission Test")
    tracked = project / "tracked.txt"
    tracked.write_text("immutable source\n", encoding="utf-8")
    _git(project, "add", "tracked.txt")
    _git(project, "commit", "-m", "test source")
    commit = _git(project, "rev-parse", "HEAD")

    run_id = "production-run"
    run_root = tmp_path / "runs" / run_id
    production_db = run_root / "state.sqlite3"
    database = Database(production_db)
    database.initialize()
    train_path = _make_view(tmp_path / "train.npz", prefix="train")
    test_path = _make_view(tmp_path / "test.npz", prefix="test")
    train = load_feature_view(train_path)
    test = load_feature_view(test_path)
    config = tmp_path / "winner.yaml"
    config.write_text("model: fixture\n", encoding="utf-8")
    config_hash = sha256_file(config)
    model_dir = tmp_path / "winner-model"
    model_dir.mkdir()
    primary = model_dir / "model.bin"
    primary.write_bytes(b"immutable winner")
    model_manifest = create_model_bundle(
        model_dir,
        primary,
        plugin="tests.fixture:Winner",
        seed=11,
        commit_sha=commit,
        config_sha256=config_hash,
        data_view_sha256=train.sha256,
        features=train,
    )
    valid_predictions = write_prediction_artifact(
        tmp_path / "valid_predictions.npz", train, np.asarray([0.1, 0.2, 0.3])
    )
    report = run_root / "report"
    report.mkdir(parents=True)
    report_files = {
        "events.jsonl": '{"event":"complete"}\n',
        "evidence_index.json": '{"test_scored":false}\n',
        "experiment_graph.json": '{"nodes":[]}\n',
        "experiments.md": "# Experiments\n",
        "interventions.json": "[]\n",
        "manual_interventions.json": json.dumps(
            {"manual_intervention_count": 0, "manual_interventions": []}
        ),
        "manual_interventions.md": "# Manual intervention summary\n\nNone.\n",
        "iteration_logs.json": "[]\n",
        "resources.json": json.dumps(
            {
                "wall_seconds": 1,
                "agent_wall_seconds": 1,
                "llm_input_tokens": 7,
                "llm_output_tokens": 3,
                "llm_total_tokens": 10,
                "iterations_used": 1,
                "iteration_cap": 50,
                "gpu_hours": 0.0,
            }
        ),
        "results.json": json.dumps(
            {
                "dataset": "KuaiRand-Pure",
                "validation_best": {
                    "GAUC": 0.62,
                    "nDCG@5": 0.58,
                    "primary": 0.60,
                    "split": "valid",
                },
                "validation_best_experiment_id": "candidate-001",
                "official_baseline": {
                    "GAUC": 0.6674,
                    "nDCG@5": 0.5357,
                    "primary": 0.6016,
                    "split": "valid",
                },
                "delta_over_official_baseline": {
                    "GAUC": -0.0474,
                    "nDCG@5": 0.0443,
                    "primary": -0.0016,
                },
                "hidden_test_scored_locally": False,
            }
        ),
        "artifact_summary.json": "{}\n",
        "recovery_events.json": "{}\n",
        "environment_identity.json": json.dumps(
            {
                "runtime_kind": "docker",
                "worker_image_digest": "sha256:" + "9" * 64,
                "container_platform": "linux/arm64",
            }
        ),
    }
    for name, content in report_files.items():
        (report / name).write_text(content, encoding="utf-8")
    evidence = report / "evidence_index.json"
    best_valid = create_best_valid_bundle(
        run_root / "best-valid",
        run_id=run_id,
        experiment_id="candidate-001",
        model_bundle=artifact_ref(model_manifest, "model_bundle"),
        valid_predictions=artifact_ref(valid_predictions, "predictions"),
        evidence_index=artifact_ref(evidence, "evidence_index"),
        metrics=Metrics(
            GAUC=0.62,
            **{"nDCG@5": 0.58},
            primary=0.60,
            users=2,
            rows=3,
            evaluator_sha256="0" * 64,
            split="valid",
            seed=11,
        ),
        commit_sha=commit,
        config_path=config,
        config_sha256=config_hash,
    )
    with database.connect() as connection:
        now = "2026-08-30T00:00:00+00:00"
        connection.execute(
            "INSERT INTO runs(run_id,state,created_at,updated_at,deadline_epoch_ms,root_commit,"
            "environment_sha256,data_manifest_sha256,evaluator_sha256,"
            "search_champion_experiment_id) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                state,
                now,
                now,
                1,
                commit,
                "1" * 64,
                "2" * 64,
                "3" * 64,
                "candidate-001",
            ),
        )
        connection.execute(
            "INSERT INTO experiments(experiment_id,run_id,iteration_number,parent_id,operator,"
            "hypothesis,proposal_json,state,commit_sha,config_sha256,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "candidate-001",
                run_id,
                1,
                None,
                "MODEL_BLOCK",
                "fixture",
                "{}",
                "PROMOTED",
                commit,
                config_hash,
                now,
                now,
            ),
        )
        best_ref = artifact_ref(best_valid.path, "best_valid_manifest")
        connection.execute(
            "INSERT INTO artifacts(artifact_id,experiment_id,attempt_id,kind,path,sha256,size_bytes,"
            "schema_version,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                best_ref.artifact_id,
                "candidate-001",
                None,
                best_ref.kind,
                best_ref.path,
                best_ref.sha256,
                best_ref.size_bytes,
                best_ref.schema_version,
                now,
            ),
        )
    return SourceFixture(project, production_db, run_id, commit, test_path, test.sha256)


class FakePredictor:
    def __init__(self) -> None:
        self.requests: list[RunRequest] = []
        self.fail_once = False

    def __call__(self, request: RunRequest, attempt_dir: Path) -> RunResult:
        self.requests.append(request)
        if self.fail_once and len(self.requests) == 1:
            raise RuntimeError("simulated coordinator interruption")
        assert request.operation == "predict"
        assert request.split == "test"
        assert request.rung == "predict"
        assert request.target_view_path is None
        features = load_feature_view(request.feature_view_path)
        predictions = write_prediction_artifact(
            Path(request.output_dir) / "predictions.npz",
            features,
            np.asarray([0.9, 0.4, 0.1]),
        )
        return RunResult(
            run_id=request.run_id,
            experiment_id=request.experiment_id,
            attempt_id=request.attempt_id,
            status=AttemptStatus.SUCCESS,
            exit_code=0,
            command_sha256="4" * 64,
            commit_sha=request.commit_sha,
            config_sha256=request.config_sha256,
            data_view_sha256=request.data_view_sha256,
            environment_sha256=request.environment_sha256,
            artifacts=[artifact_ref(predictions, "predictions")],
            wall_seconds=0.1,
        )


class FakeChecker:
    def __init__(self, *, score_capability: bool = False) -> None:
        self.paths: list[Path] = []
        self.score_capability = score_capability

    def __call__(self, csv_path: Path) -> CheckResult:
        self.paths.append(csv_path.resolve())
        command = (
            "python",
            "submit.py",
            str(csv_path),
            "--split",
            "test",
            "--score" if self.score_capability else "--check",
        )
        return CheckResult(True, command, "OK", "", 0)


def _coordinator(
    tmp_path: Path,
    source: SourceFixture,
    *,
    predictor: FakePredictor | None = None,
    checker: FakeChecker | None = None,
) -> tuple[FinalSubmissionCoordinator, SubmissionRepository, FakePredictor, FakeChecker]:
    repository = SubmissionRepository(tmp_path / "submission-state.sqlite3")
    repository.initialize()
    predictor = predictor or FakePredictor()
    checker = checker or FakeChecker()
    coordinator = FinalSubmissionCoordinator(
        repository,
        SubmissionJobConfig(
            repository_root=source.project,
            jobs_root=tmp_path / "submission-jobs",
            test_feature_path=source.test_features,
            test_data_view_sha256=source.test_hash,
            environment_sha256="5" * 64,
            expected_test_rows=3,
            prediction_timeout_seconds=30,
        ),
        SubmissionDependencies(
            predictor=predictor,
            csv_builder=build_aligned_csv,
            checker=checker,
        ),
    )
    return coordinator, repository, predictor, checker


def test_submission_job_runs_twice_checked_seals_and_hands_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_fixture(tmp_path)
    coordinator, repository, predictor, checker = _coordinator(tmp_path, source)
    staging_calls: list[Path] = []
    original_stager = coordinator_module.stage_submission_bundle

    def record_finalizer_staging(output_dir, **kwargs):
        staging_calls.append(Path(output_dir).resolve())
        return original_stager(output_dir, **kwargs)

    monkeypatch.setattr(coordinator_module, "stage_submission_bundle", record_finalizer_staging)
    created = coordinator.create(source.production_db, source.run_id)
    replay = coordinator.create(source.production_db, source.run_id)
    assert replay["job_id"] == created["job_id"]

    ready = coordinator.run_until_ready(created["job_id"])

    assert ready["state"] == SubmissionState.READY_FOR_HANDOFF
    assert len(predictor.requests) == 1
    assert len(checker.paths) == 2
    assert staging_calls == [
        (tmp_path / "submission-jobs" / created["job_id"] / "final.staging").resolve()
    ]
    assert checker.paths[0] != checker.paths[1]
    assert checker.paths[1].parent.name == "final.staging"
    sealed = Path(ready["sealed_path"])
    seal = json.loads((sealed / "submission_seal.json").read_text(encoding="utf-8"))
    assert seal["expected_test_rows"] == 3
    assert seal["organizer_checks"] == 2
    assert seal["test_scored"] is False
    assert seal["source_report_sha256"] == ready["source_report_sha256"]
    assert (sealed / "best-valid" / "model" / "model_bundle.json").is_file()
    assert (sealed / "source-report" / "events.jsonl").is_file()
    assert "source-report/resources.json" in seal["artifacts"]
    summary_path = sealed / "final_results_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert seal["final_results_summary_sha256"] == sha256_file(summary_path)
    assert summary["kind"] == "final_results_summary"
    assert summary["submission_columns"] == ["row_id", "user_id", "video_id", "score"]
    assert summary["organizer_checks"] == 2
    assert summary["test_scored_locally"] is False
    assert summary["validation_results"]["best"]["GAUC"] == 0.62
    assert summary["validation_results"]["best"]["nDCG@5"] == 0.58
    assert summary["resource_usage"]["iteration_cap"] == 50
    assert summary["resource_usage"]["llm_total_tokens"] == 10
    assert summary["resource_usage"]["manual_intervention_count"] == 0
    assert summary["artifacts"]["submission_csv"]["sha256"] == ready["csv_sha256"]
    assert summary["artifacts"]["test_predictions"]["sha256"] == ready["prediction_sha256"]
    assert summary["source_identity"]["commit_sha"] == source.commit
    assert summary["source_identity"]["environment"]["runtime_kind"] == "docker"
    assert summary["source_identity"]["environment"]["worker_image_digest"] == (
        "sha256:" + "9" * 64
    )
    assert summary["source_identity"]["environment_evidence"]["sha256"] == sha256_file(
        sealed / "source-report" / "environment_identity.json"
    )

    target = tmp_path / "authorized-handoff"
    handed = coordinator.handoff(
        created["job_id"], target, authorized_seal_sha256=ready["seal_sha256"]
    )
    assert handed["state"] == SubmissionState.HANDED_OFF
    assert sha256_file(target / "submission_seal.json") == ready["seal_sha256"]
    replayed = coordinator.handoff(
        created["job_id"], target, authorized_seal_sha256=ready["seal_sha256"]
    )
    assert replayed["state"] == SubmissionState.HANDED_OFF
    with pytest.raises(SubmissionCoordinatorError, match="cannot be redirected"):
        coordinator.handoff(
            created["job_id"],
            tmp_path / "second-target",
            authorized_seal_sha256=ready["seal_sha256"],
        )
    assert repository.get_handoff(created["job_id"])["status"] == "COMPLETE"


def test_prediction_interruption_resumes_identical_persisted_request(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    predictor = FakePredictor()
    predictor.fail_once = True
    coordinator, repository, _, _ = _coordinator(tmp_path, source, predictor=predictor)
    job = coordinator.create(source.production_db, source.run_id)
    coordinator.advance(job["job_id"])
    coordinator.advance(job["job_id"])
    with pytest.raises(RuntimeError, match="simulated coordinator interruption"):
        coordinator.advance(job["job_id"])
    interrupted = repository.get_job(job["job_id"])
    assert interrupted["state"] == SubmissionState.PREDICTING
    persisted = json.loads(interrupted["prediction_request_json"])

    coordinator.advance(job["job_id"])

    resumed = repository.get_job(job["job_id"])
    assert resumed["state"] == SubmissionState.PREDICTED
    assert predictor.requests[0].model_dump(mode="json") == predictor.requests[1].model_dump(
        mode="json"
    )
    assert persisted["target_view_path"] is None


def test_checker_with_scoring_capability_is_rejected(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    coordinator, _, _, _ = _coordinator(
        tmp_path, source, checker=FakeChecker(score_capability=True)
    )
    job = coordinator.create(source.production_db, source.run_id)
    for _ in range(4):
        coordinator.advance(job["job_id"])
    with pytest.raises(SubmissionCoordinatorError, match="test-check-only"):
        coordinator.advance(job["job_id"])


def test_docker_final_summary_requires_immutable_image_digest(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    identity = source.production_db.parent / "report" / "environment_identity.json"
    identity.write_text(
        json.dumps({"runtime_kind": "docker", "worker_image_digest": "latest"}),
        encoding="utf-8",
    )
    coordinator, repository, _, _ = _coordinator(tmp_path, source)
    job = coordinator.create(source.production_db, source.run_id)

    with pytest.raises(SubmissionCoordinatorError, match="immutable worker image digest"):
        coordinator.run_until_ready(job["job_id"])
    assert repository.get_job(job["job_id"])["state"] == SubmissionState.SECOND_CHECK_VALID


def test_incomplete_production_run_cannot_create_submission_job(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path, state="SEARCHING")
    coordinator, _, _, _ = _coordinator(tmp_path, source)

    with pytest.raises(SubmissionRepositoryError, match="must be COMPLETE"):
        coordinator.create(source.production_db, source.run_id)


def test_incomplete_judge_report_cannot_create_submission_job(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    (source.production_db.parent / "report" / "results.json").unlink()
    coordinator, _, _, _ = _coordinator(tmp_path, source)

    with pytest.raises(SubmissionRepositoryError, match="results.json"):
        coordinator.create(source.production_db, source.run_id)


def test_handoff_requires_exact_seal_authorization(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    coordinator, _, _, _ = _coordinator(tmp_path, source)
    job = coordinator.create(source.production_db, source.run_id)
    ready = coordinator.run_until_ready(job["job_id"])

    with pytest.raises(SubmissionRepositoryError, match="does not match"):
        coordinator.handoff(job["job_id"], tmp_path / "handoff", authorized_seal_sha256="0" * 64)
    assert ready["state"] == SubmissionState.READY_FOR_HANDOFF


def test_source_manifest_drift_is_detected_after_job_creation(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    coordinator, _, _, _ = _coordinator(tmp_path, source)
    job = coordinator.create(source.production_db, source.run_id)
    manifest = source.production_db.parent / "best-valid" / "best_valid_manifest.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["test_prediction_created"] = True
    manifest.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(SubmissionCoordinatorError, match="production source changed"):
        coordinator.advance(job["job_id"])


def test_first_checker_result_is_replayed_after_transition_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_fixture(tmp_path)
    coordinator, repository, _, checker = _coordinator(tmp_path, source)
    job = coordinator.create(source.production_db, source.run_id)
    while repository.get_job(job["job_id"])["state"] != SubmissionState.CSV_BUILT:
        coordinator.advance(job["job_id"])
    original_transition = repository.transition
    interrupted = False

    def transition_with_interruption(job_id, expected, target, **kwargs):
        nonlocal interrupted
        if target == SubmissionState.FIRST_CHECK_VALID and not interrupted:
            interrupted = True
            raise RuntimeError("interrupted after durable checker record")
        return original_transition(job_id, expected, target, **kwargs)

    monkeypatch.setattr(repository, "transition", transition_with_interruption)
    with pytest.raises(RuntimeError, match="durable checker"):
        coordinator.advance(job["job_id"])
    assert repository.get_check(job["job_id"], 1) is not None
    assert len(checker.paths) == 1
    monkeypatch.setattr(repository, "transition", original_transition)

    coordinator.advance(job["job_id"])

    assert repository.get_job(job["job_id"])["state"] == SubmissionState.FIRST_CHECK_VALID
    assert len(checker.paths) == 1


def test_reporting_finalizer_staging_replays_after_coordinator_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_fixture(tmp_path)
    coordinator, repository, _, checker = _coordinator(tmp_path, source)
    job = coordinator.create(source.production_db, source.run_id)
    while repository.get_job(job["job_id"])["state"] != SubmissionState.FIRST_CHECK_VALID:
        coordinator.advance(job["job_id"])
    original_stager = coordinator_module.stage_submission_bundle
    interrupted = False

    def stage_then_interrupt(output_dir, **kwargs):
        nonlocal interrupted
        staged = original_stager(output_dir, **kwargs)
        if not interrupted:
            interrupted = True
            raise RuntimeError("interrupted after reporting finalizer staging")
        return staged

    monkeypatch.setattr(coordinator_module, "stage_submission_bundle", stage_then_interrupt)
    with pytest.raises(RuntimeError, match="reporting finalizer staging"):
        coordinator.advance(job["job_id"])
    assert repository.get_job(job["job_id"])["state"] == SubmissionState.STAGING
    assert len(checker.paths) == 1

    ready = coordinator.run_until_ready(job["job_id"])

    assert ready["state"] == SubmissionState.READY_FOR_HANDOFF
    assert len(checker.paths) == 2


def test_atomic_seal_resumes_after_rename_before_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_fixture(tmp_path)
    coordinator, repository, _, _ = _coordinator(tmp_path, source)
    job = coordinator.create(source.production_db, source.run_id)
    while repository.get_job(job["job_id"])["state"] != SubmissionState.SECOND_CHECK_VALID:
        coordinator.advance(job["job_id"])
    original_transition = repository.transition
    interrupted = False

    def transition_with_interruption(job_id, expected, target, **kwargs):
        nonlocal interrupted
        if target == SubmissionState.SEALED and not interrupted:
            interrupted = True
            raise RuntimeError("interrupted after atomic rename")
        return original_transition(job_id, expected, target, **kwargs)

    monkeypatch.setattr(repository, "transition", transition_with_interruption)
    with pytest.raises(RuntimeError, match="atomic rename"):
        coordinator.advance(job["job_id"])
    assert repository.get_job(job["job_id"])["state"] == SubmissionState.SECOND_CHECK_VALID
    assert (tmp_path / "submission-jobs" / job["job_id"] / "sealed").is_dir()
    monkeypatch.setattr(repository, "transition", original_transition)

    sealed = coordinator.advance(job["job_id"])

    assert sealed["state"] == SubmissionState.SEALED


def test_handoff_rejects_artifact_drift_even_when_seal_file_is_unchanged(
    tmp_path: Path,
) -> None:
    source = _source_fixture(tmp_path)
    coordinator, _, _, _ = _coordinator(tmp_path, source)
    job = coordinator.create(source.production_db, source.run_id)
    ready = coordinator.run_until_ready(job["job_id"])
    (Path(ready["sealed_path"]) / "submission.csv").write_text("drift\n", encoding="utf-8")

    with pytest.raises(SubmissionRepositoryError, match="artifact drifted"):
        coordinator.handoff(
            job["job_id"],
            tmp_path / "handoff",
            authorized_seal_sha256=ready["seal_sha256"],
        )
