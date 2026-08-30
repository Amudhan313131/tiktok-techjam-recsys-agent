"""Resumable orchestration for one validation-winner test prediction and handoff."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from rex.contracts import AttemptStatus, Metrics, RunRequest, RunResult
from rex.data.manifest import sha256_file
from rex.data.views import load_feature_view
from rex.execution.artifacts import atomic_write_json, load_prediction_artifact
from rex.evaluation.submission import build_submission
from rex.models.bundle import LoadedModelBundle, validate_model_bundle
from rex.reporting.finalizer import stage_submission_bundle
from rex.submission.repository import (
    SourceLocator,
    SubmissionRepository,
    SubmissionState,
    discover_completed_source,
    report_fingerprint,
)
from rex.submission.transport import FilesystemHandoff


class SubmissionCoordinatorError(RuntimeError):
    """A submission gate failed closed."""


@dataclass(frozen=True)
class CheckResult:
    valid: bool
    command: tuple[str, ...]
    stdout: str
    stderr: str
    returncode: int


@dataclass(frozen=True)
class VerifiedSource:
    locator: SourceLocator
    manifest: Mapping[str, Any]
    model_bundle: LoadedModelBundle
    config_path: Path
    commit_sha: str
    config_sha256: str
    incumbent_experiment_id: str


@dataclass(frozen=True)
class StagingRequest:
    job_id: str
    output_dir: Path
    source: VerifiedSource
    prediction_path: Path
    submission_path: Path
    first_check_transcript: Path

    @property
    def locator_root(self) -> Path:
        return self.source.locator.best_valid_path.parent


class Predictor(Protocol):
    def __call__(self, request: RunRequest, attempt_dir: Path) -> RunResult: ...


class CsvBuilder(Protocol):
    def __call__(
        self,
        prediction_path: Path,
        csv_path: Path,
        expected_features: Path,
        expected_rows: int,
    ) -> Path: ...


class Checker(Protocol):
    def __call__(self, csv_path: Path) -> CheckResult: ...


class BundleStager(Protocol):
    def __call__(self, request: StagingRequest) -> Path: ...


@dataclass(frozen=True)
class SubmissionDependencies:
    predictor: Predictor
    csv_builder: CsvBuilder
    checker: Checker
    bundle_stager: BundleStager | None = None


@dataclass(frozen=True)
class SubmissionJobConfig:
    repository_root: Path
    jobs_root: Path
    test_feature_path: Path
    test_data_view_sha256: str
    environment_sha256: str
    expected_test_rows: int = 170_588
    prediction_timeout_seconds: int = 3_600
    max_memory_mb: int | None = None

    def __post_init__(self) -> None:
        for name in ("test_data_view_sha256", "environment_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.expected_test_rows <= 0:
            raise ValueError("expected_test_rows must be positive")
        if self.prediction_timeout_seconds <= 0:
            raise ValueError("prediction_timeout_seconds must be positive")


def build_aligned_csv(
    prediction_path: Path,
    csv_path: Path,
    expected_features: Path,
    expected_rows: int,
) -> Path:
    """Compatibility adapter over the canonical submission implementation."""

    return build_submission(
        prediction_path,
        csv_path,
        expected_features=expected_features,
        expected_rows=expected_rows,
    )


class DefaultBundleStager:
    """Delegate production staging to the reporting finalizer boundary."""

    def __call__(self, request: StagingRequest) -> Path:
        return stage_submission_bundle(
            request.output_dir,
            best_valid_dir=request.locator_root,
            source_report_dir=request.source.locator.source_report_path,
            predictions_path=request.prediction_path,
            submission_path=request.submission_path,
            first_check_transcript=request.first_check_transcript,
            expected_commit_sha=request.source.commit_sha,
            expected_config_sha256=request.source.config_sha256,
        )


def _git(cwd: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=cwd, capture_output=True, text=True, timeout=60, check=False
    )
    if completed.returncode:
        raise SubmissionCoordinatorError(
            f"git {' '.join(arguments)} failed: {completed.stderr[-2000:]}"
        )
    return completed.stdout.strip()


def _safe_relative(root: Path, relative_name: str) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise SubmissionCoordinatorError(f"unsafe best-valid artifact name: {relative_name}")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise SubmissionCoordinatorError(
            f"best-valid artifact escapes immutable bundle: {relative_name}"
        ) from error
    return candidate


class FinalSubmissionCoordinator:
    """Advance a final-submission job through durable, replayable gates."""

    _READY_TERMINALS = {
        SubmissionState.READY_FOR_HANDOFF,
        SubmissionState.HANDOFF_IN_PROGRESS,
        SubmissionState.HANDED_OFF,
    }

    def __init__(
        self,
        repository: SubmissionRepository,
        config: SubmissionJobConfig,
        dependencies: SubmissionDependencies,
        *,
        handoff_transport: Callable[[Path, Path, str], Path] | None = None,
    ):
        self.repository = repository
        self.config = config
        self.dependencies = dependencies
        self.handoff_transport = handoff_transport or FilesystemHandoff()

    def create(self, source_database: str | Path, source_run_id: str) -> dict[str, Any]:
        source = discover_completed_source(source_database, source_run_id)
        return self.repository.create_job(source)

    def run_until_ready(self, job_id: str) -> dict[str, Any]:
        while True:
            job = self.repository.get_job(job_id)
            state = SubmissionState(job["state"])
            if state in self._READY_TERMINALS or state in {
                SubmissionState.REJECTED,
                SubmissionState.FAILED,
            }:
                return job
            self.advance(job_id)

    def advance(self, job_id: str) -> dict[str, Any]:
        job = self.repository.get_job(job_id)
        state = SubmissionState(job["state"])
        if state == SubmissionState.CREATED:
            return self._verify_source(job)
        source = self._verified_source(job)
        if state == SubmissionState.SOURCE_VERIFIED:
            return self._prepare_worktree(job, source)
        if state in {SubmissionState.WORKTREE_READY, SubmissionState.PREDICTING}:
            return self._predict(job, source)
        if state == SubmissionState.PREDICTED:
            return self._build_csv(job)
        if state == SubmissionState.CSV_BUILT:
            return self._check(job, ordinal=1)
        if state in {SubmissionState.FIRST_CHECK_VALID, SubmissionState.STAGING}:
            return self._stage_and_check(job, source)
        if state == SubmissionState.SECOND_CHECK_VALID:
            return self._seal(job, source)
        if state == SubmissionState.SEALED:
            return self.repository.transition(
                job_id,
                SubmissionState.SEALED,
                SubmissionState.READY_FOR_HANDOFF,
                evidence={"seal_sha256": job["seal_sha256"]},
            )
        return job

    def handoff(
        self,
        job_id: str,
        target_dir: str | Path,
        *,
        authorized_seal_sha256: str,
    ) -> dict[str, Any]:
        job = self.repository.get_job(job_id)
        state = SubmissionState(job["state"])
        if state == SubmissionState.HANDED_OFF:
            handoff = self.repository.get_handoff(job_id)
            if (
                handoff["authorized_seal_sha256"] != authorized_seal_sha256
                or Path(handoff["target_path"]).resolve() != Path(target_dir).resolve()
            ):
                raise SubmissionCoordinatorError("one-time handoff cannot be redirected")
            return job
        if state == SubmissionState.READY_FOR_HANDOFF:
            self.repository.authorize_handoff(
                job_id, authorized_seal_sha256, Path(target_dir)
            )
            job = self.repository.get_job(job_id)
        if SubmissionState(job["state"]) != SubmissionState.HANDOFF_IN_PROGRESS:
            raise SubmissionCoordinatorError("submission is not ready for handoff")
        handoff = self.repository.get_handoff(job_id)
        if (
            handoff["authorized_seal_sha256"] != authorized_seal_sha256
            or Path(handoff["target_path"]).resolve() != Path(target_dir).resolve()
        ):
            raise SubmissionCoordinatorError("handoff request conflicts with one-time authorization")
        copied = self.handoff_transport(
            Path(job["sealed_path"]), Path(handoff["target_path"]), authorized_seal_sha256
        )
        copied_manifest = copied / FilesystemHandoff.manifest_name
        copied_hash = sha256_file(copied_manifest)
        if copied_hash != authorized_seal_sha256:
            raise SubmissionCoordinatorError("handoff copy does not match authorized seal")
        return self.repository.complete_handoff(job_id, copied_hash)

    def reject(self, job_id: str, *, code: str, summary: str) -> dict[str, Any]:
        """Explicitly close a policy-invalid job without losing its evidence."""

        return self.repository.terminate(
            job_id, SubmissionState.REJECTED, code=code, summary=summary
        )

    def fail(self, job_id: str, *, code: str, summary: str) -> dict[str, Any]:
        """Explicitly close an unrecoverable job; ordinary exceptions remain resumable."""

        return self.repository.terminate(
            job_id, SubmissionState.FAILED, code=code, summary=summary
        )

    def _verify_source(self, job: Mapping[str, Any]) -> dict[str, Any]:
        source = self._strict_source(job)
        return self.repository.transition(
            str(job["job_id"]),
            SubmissionState.CREATED,
            SubmissionState.SOURCE_VERIFIED,
            evidence={
                "best_valid_sha256": source.locator.best_valid_sha256,
                "commit_sha": source.commit_sha,
                "config_sha256": source.config_sha256,
                "test_prediction_created": False,
            },
            updates={
                "source_commit": source.commit_sha,
                "config_sha256": source.config_sha256,
                "incumbent_experiment_id": source.incumbent_experiment_id,
            },
        )

    def _verified_source(self, job: Mapping[str, Any]) -> VerifiedSource:
        source = self._strict_source(job)
        expected = {
            "source_commit": source.commit_sha,
            "config_sha256": source.config_sha256,
            "incumbent_experiment_id": source.incumbent_experiment_id,
        }
        conflicts = {
            key: {"stored": job.get(key), "observed": value}
            for key, value in expected.items()
            if job.get(key) != value
        }
        if conflicts:
            raise SubmissionCoordinatorError(
                "verified source provenance drifted: " + json.dumps(conflicts, sort_keys=True)
            )
        return source

    def _strict_source(self, job: Mapping[str, Any]) -> VerifiedSource:
        locator = discover_completed_source(job["source_database_path"], job["source_run_id"])
        immutable_fields = {
            "source_run_fingerprint": locator.source_run_fingerprint,
            "source_report_path": str(locator.source_report_path),
            "source_report_sha256": locator.source_report_sha256,
            "best_valid_path": str(locator.best_valid_path),
            "best_valid_sha256": locator.best_valid_sha256,
        }
        conflicts = {
            key: {"stored": job[key], "observed": value}
            for key, value in immutable_fields.items()
            if job[key] != value
        }
        if conflicts:
            raise SubmissionCoordinatorError(
                "production source changed after job creation: "
                + json.dumps(conflicts, sort_keys=True)
            )
        try:
            manifest = json.loads(locator.best_valid_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SubmissionCoordinatorError(f"invalid best-valid manifest: {error}") from error
        required = {
            "schema_version",
            "kind",
            "test_prediction_created",
            "run_id",
            "incumbent_experiment_id",
            "commit_sha",
            "config_sha256",
            "metrics",
            "artifacts",
        }
        allowed = required | {"model_bundle_sha256", "test_scored", "validation_metrics"}
        if (
            not isinstance(manifest, dict)
            or not required.issubset(manifest)
            or set(manifest).difference(allowed)
        ):
            raise SubmissionCoordinatorError("best-valid manifest does not satisfy the strict schema")
        if (
            manifest["kind"] != "best_valid"
            or manifest["run_id"] != locator.source_run_id
            or manifest["test_prediction_created"] is not False
        ):
            raise SubmissionCoordinatorError("best-valid manifest identity or split policy is invalid")
        if manifest.get("test_scored", False) is not False:
            raise SubmissionCoordinatorError("best-valid manifest claims test scoring")
        if "validation_metrics" in manifest and manifest["validation_metrics"] != manifest["metrics"]:
            raise SubmissionCoordinatorError("best-valid validation metrics disagree")
        try:
            Metrics.model_validate(manifest["metrics"])
        except Exception as error:
            raise SubmissionCoordinatorError(f"best-valid metrics are invalid: {error}") from error
        artifacts = manifest["artifacts"]
        if not isinstance(artifacts, dict) or not artifacts:
            raise SubmissionCoordinatorError("best-valid manifest has no artifacts")
        root = locator.best_valid_path.parent.resolve()
        for relative_name, raw_ref in artifacts.items():
            if not isinstance(relative_name, str) or not isinstance(raw_ref, dict):
                raise SubmissionCoordinatorError("best-valid artifact record is malformed")
            path = _safe_relative(root, relative_name)
            if path.is_symlink() or not path.is_file():
                raise SubmissionCoordinatorError(f"best-valid artifact is missing: {relative_name}")
            if Path(str(raw_ref.get("path", ""))).resolve() != path:
                raise SubmissionCoordinatorError(
                    f"best-valid artifact path disagrees with manifest key: {relative_name}"
                )
            if raw_ref.get("sha256") != sha256_file(path):
                raise SubmissionCoordinatorError(f"best-valid artifact drifted: {relative_name}")
            if raw_ref.get("size_bytes") != path.stat().st_size:
                raise SubmissionCoordinatorError(
                    f"best-valid artifact size drifted: {relative_name}"
                )
        evidence_reference = artifacts.get("evidence_index.json")
        report_evidence = locator.source_report_path / "evidence_index.json"
        if (
            not isinstance(evidence_reference, dict)
            or evidence_reference.get("sha256") != sha256_file(report_evidence)
        ):
            raise SubmissionCoordinatorError(
                "best-valid evidence index is not the immutable completed-run report"
            )
        commit = str(manifest["commit_sha"])
        config_hash = str(manifest["config_sha256"])
        model_path = _safe_relative(root, "model/model_bundle.json")
        try:
            model = validate_model_bundle(
                model_path,
                expected_commit_sha=commit,
                expected_config_sha256=config_hash,
            )
        except Exception as error:
            raise SubmissionCoordinatorError(f"best-valid model bundle is invalid: {error}") from error
        if manifest.get("model_bundle_sha256", sha256_file(model_path)) != sha256_file(model_path):
            raise SubmissionCoordinatorError("best-valid model-bundle digest disagrees")
        config_names = [
            name
            for name in artifacts
            if Path(name).parent == Path(".") and Path(name).stem == "config"
        ]
        if len(config_names) != 1:
            raise SubmissionCoordinatorError("best-valid bundle must contain exactly one config")
        config_path = _safe_relative(root, config_names[0])
        if sha256_file(config_path) != config_hash:
            raise SubmissionCoordinatorError("best-valid config hash mismatch")

        incumbent = str(manifest["incumbent_experiment_id"])
        source_run = locator.source_run
        if str(source_run.get("search_champion_experiment_id")) != incumbent:
            raise SubmissionCoordinatorError("best-valid winner is not the run's search champion")
        self._verify_database_winner(locator, incumbent, commit, config_hash)
        repository = self.config.repository_root.resolve(strict=True)
        _git(repository, "cat-file", "-e", f"{commit}^{{commit}}")
        return VerifiedSource(
            locator=locator,
            manifest=manifest,
            model_bundle=model,
            config_path=config_path,
            commit_sha=commit,
            config_sha256=config_hash,
            incumbent_experiment_id=incumbent,
        )

    @staticmethod
    def _verify_database_winner(
        locator: SourceLocator,
        incumbent: str,
        commit: str,
        config_sha256: str,
    ) -> None:
        uri = f"file:{locator.source_database_path.as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        try:
            registered = connection.execute(
                "SELECT sha256 FROM artifacts WHERE kind='best_valid_manifest' AND path=?",
                (str(locator.best_valid_path),),
            ).fetchone()
            if registered is not None and registered["sha256"] != locator.best_valid_sha256:
                raise SubmissionCoordinatorError("registered best-valid artifact hash disagrees")
            if incumbent == "baseline":
                if commit != locator.source_run["root_commit"]:
                    raise SubmissionCoordinatorError("baseline best-valid commit is not the run root")
                return
            experiment = connection.execute(
                "SELECT commit_sha,config_sha256 FROM experiments WHERE run_id=? AND experiment_id=?",
                (locator.source_run_id, incumbent),
            ).fetchone()
        finally:
            connection.close()
        if experiment is None:
            raise SubmissionCoordinatorError("best-valid experiment is absent from production DB")
        if experiment["commit_sha"] != commit or experiment["config_sha256"] != config_sha256:
            raise SubmissionCoordinatorError("best-valid provenance disagrees with production DB")

    def _prepare_worktree(
        self, job: Mapping[str, Any], source: VerifiedSource
    ) -> dict[str, Any]:
        root = (self.config.jobs_root / str(job["job_id"])).resolve()
        worktree = root / "worktree"
        repository = self.config.repository_root.resolve(strict=True)
        root.mkdir(parents=True, exist_ok=True)
        if not worktree.exists():
            _git(repository, "worktree", "add", "--detach", str(worktree), source.commit_sha)
        if _git(worktree, "rev-parse", "HEAD") != source.commit_sha:
            raise SubmissionCoordinatorError("submission worktree is at the wrong commit")
        if _git(worktree, "status", "--porcelain", "--untracked-files=normal"):
            raise SubmissionCoordinatorError("submission worktree is not clean")
        return self.repository.transition(
            str(job["job_id"]),
            SubmissionState.SOURCE_VERIFIED,
            SubmissionState.WORKTREE_READY,
            evidence={"detached": True, "commit_sha": source.commit_sha},
            updates={"worktree_path": str(worktree)},
        )

    def _prediction_request(
        self, job: Mapping[str, Any], source: VerifiedSource
    ) -> RunRequest:
        stored = job.get("prediction_request_json")
        if stored:
            return RunRequest.model_validate_json(stored)
        test_view = load_feature_view(self.config.test_feature_path)
        if test_view.sha256 != self.config.test_data_view_sha256:
            raise SubmissionCoordinatorError("test feature view hash does not match authorization")
        if test_view.rows != self.config.expected_test_rows:
            raise SubmissionCoordinatorError(
                f"test view has {test_view.rows} rows, expected {self.config.expected_test_rows}"
            )
        job_root = (self.config.jobs_root / str(job["job_id"])).resolve()
        return RunRequest(
            run_id=str(job["source_run_id"]),
            experiment_id=source.incumbent_experiment_id,
            attempt_id=f"{job['job_id']}-test-predict",
            parent_id=None,
            commit_sha=source.commit_sha,
            plugin=source.model_bundle.manifest.plugin,
            operation="predict",
            config_path=str(source.config_path),
            config_sha256=source.config_sha256,
            seed=source.model_bundle.manifest.seed,
            rung="predict",
            split="test",
            fold=None,
            feature_view_path=str(self.config.test_feature_path.resolve(strict=True)),
            target_view_path=None,
            workspace_path=str(Path(job["worktree_path"]).resolve(strict=True)),
            model_bundle_path=str(source.model_bundle.manifest_path),
            output_dir=str(job_root / "prediction" / "output"),
            deadline_epoch_ms=int(time.time() * 1000)
            + self.config.prediction_timeout_seconds * 1000,
            timeout_seconds=self.config.prediction_timeout_seconds,
            max_memory_mb=self.config.max_memory_mb,
            data_view_sha256=self.config.test_data_view_sha256,
            environment_sha256=self.config.environment_sha256,
        )

    def _predict(self, job: Mapping[str, Any], source: VerifiedSource) -> dict[str, Any]:
        state = SubmissionState(job["state"])
        request = self._prediction_request(job, source)
        if state == SubmissionState.WORKTREE_READY:
            job = self.repository.transition(
                str(job["job_id"]),
                SubmissionState.WORKTREE_READY,
                SubmissionState.PREDICTING,
                evidence={
                    "attempt_id": request.attempt_id,
                    "operation": "predict",
                    "split": "test",
                    "target_view_path": None,
                },
                updates={
                    "prediction_request_json": request.model_dump_json(by_alias=True)
                },
            )
        else:
            request = RunRequest.model_validate_json(str(job["prediction_request_json"]))
        attempt_dir = self.config.jobs_root / str(job["job_id"]) / "prediction"
        result = self.dependencies.predictor(request, attempt_dir)
        if result.status != AttemptStatus.SUCCESS:
            raise SubmissionCoordinatorError(
                f"test prediction failed closed: {result.status}: {result.error_summary}"
            )
        expected_result = {
            "run_id": request.run_id,
            "experiment_id": request.experiment_id,
            "attempt_id": request.attempt_id,
            "commit_sha": request.commit_sha,
            "config_sha256": request.config_sha256,
            "data_view_sha256": request.data_view_sha256,
            "environment_sha256": request.environment_sha256,
        }
        conflicts = {
            name: {"expected": value, "observed": getattr(result, name)}
            for name, value in expected_result.items()
            if getattr(result, name) != value
        }
        if conflicts:
            raise SubmissionCoordinatorError(
                "prediction result provenance mismatch: " + json.dumps(conflicts, sort_keys=True)
            )
        candidates = [artifact for artifact in result.artifacts if artifact.kind == "predictions"]
        if len(candidates) != 1:
            raise SubmissionCoordinatorError("prediction worker must return exactly one prediction")
        prediction = Path(candidates[0].path).resolve(strict=True)
        output_root = Path(request.output_dir).resolve(strict=True)
        try:
            prediction.relative_to(output_root)
        except ValueError as error:
            raise SubmissionCoordinatorError(
                "prediction artifact is outside the authorized output directory"
            ) from error
        if (
            sha256_file(prediction) != candidates[0].sha256
            or prediction.stat().st_size != candidates[0].size_bytes
        ):
            raise SubmissionCoordinatorError("prediction artifact drifted after worker completion")
        arrays = load_prediction_artifact(prediction, self.config.test_feature_path)
        if len(arrays["score"]) != self.config.expected_test_rows:
            raise SubmissionCoordinatorError("prediction artifact has the wrong test row count")
        result_path = attempt_dir / "prediction_result.json"
        atomic_write_json(result_path, result.model_dump(mode="json", by_alias=True))
        return self.repository.transition(
            str(job["job_id"]),
            SubmissionState.PREDICTING,
            SubmissionState.PREDICTED,
            evidence={
                "result_sha256": sha256_file(result_path),
                "prediction_sha256": candidates[0].sha256,
                "rows": len(arrays["score"]),
                "test_scored": False,
            },
            updates={
                "prediction_path": str(prediction),
                "prediction_sha256": candidates[0].sha256,
            },
        )

    def _build_csv(self, job: Mapping[str, Any]) -> dict[str, Any]:
        prediction = Path(str(job["prediction_path"])).resolve(strict=True)
        if sha256_file(prediction) != job["prediction_sha256"]:
            raise SubmissionCoordinatorError("prediction drifted before CSV generation")
        csv_path = (
            self.config.jobs_root / str(job["job_id"]) / "generated" / "submission.csv"
        ).resolve()
        built = self.dependencies.csv_builder(
            prediction,
            csv_path,
            self.config.test_feature_path.resolve(strict=True),
            self.config.expected_test_rows,
        ).resolve(strict=True)
        if built != csv_path:
            raise SubmissionCoordinatorError("CSV builder returned an unexpected output path")
        self._validate_csv_alignment(csv_path)
        return self.repository.transition(
            str(job["job_id"]),
            SubmissionState.PREDICTED,
            SubmissionState.CSV_BUILT,
            evidence={"csv_sha256": sha256_file(csv_path), "rows": self.config.expected_test_rows},
            updates={"csv_path": str(csv_path), "csv_sha256": sha256_file(csv_path)},
        )

    def _validate_csv_alignment(self, csv_path: Path) -> None:
        features = load_feature_view(self.config.test_feature_path)
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = csv.reader(handle)
            try:
                header = next(rows)
            except StopIteration as error:
                raise SubmissionCoordinatorError("submission CSV is empty") from error
            if header != ["row_id", "user_id", "video_id", "score"]:
                raise SubmissionCoordinatorError("submission CSV has an invalid header")
            observed = list(rows)
        if len(observed) != self.config.expected_test_rows:
            raise SubmissionCoordinatorError(
                f"submission row count mismatch: expected {self.config.expected_test_rows}, "
                f"got {len(observed)}"
            )
        for index, row in enumerate(observed):
            if len(row) != 4:
                raise SubmissionCoordinatorError(f"submission row {index} has the wrong width")
            try:
                row_id = int(row[0])
                score = float(row[3])
            except ValueError as error:
                raise SubmissionCoordinatorError(
                    f"submission row {index} contains an invalid number"
                ) from error
            if (
                row_id != index
                or row[1] != str(features.arrays["user_id"][index])
                or row[2] != str(features.arrays["video_id"][index])
            ):
                raise SubmissionCoordinatorError(f"submission alignment drift at row {index}")
            if not math.isfinite(score):
                raise SubmissionCoordinatorError(f"submission score is non-finite at row {index}")

    def _run_checker(
        self, job: Mapping[str, Any], csv_path: Path, ordinal: int
    ) -> tuple[CheckResult, Path]:
        existing = self.repository.get_check(str(job["job_id"]), ordinal)
        if existing is not None:
            transcript = Path(existing["transcript_path"]).resolve(strict=True)
            if (
                Path(existing["csv_path"]).resolve() != csv_path.resolve()
                or existing["csv_sha256"] != sha256_file(csv_path)
                or existing["transcript_sha256"] != sha256_file(transcript)
                or int(existing["returncode"]) != 0
            ):
                raise SubmissionCoordinatorError("recorded organizer check drifted")
            result = CheckResult(
                valid=True,
                command=tuple(json.loads(existing["command_json"])),
                stdout=str(existing["stdout"]),
                stderr=str(existing["stderr"]),
                returncode=int(existing["returncode"]),
            )
            self._require_check_only_command(result.command, csv_path)
            return result, transcript
        result = self.dependencies.checker(csv_path)
        command = tuple(str(value) for value in result.command)
        self._require_check_only_command(command, csv_path)
        if not result.valid or result.returncode != 0:
            raise SubmissionCoordinatorError(
                f"organizer checker {ordinal} rejected submission: "
                f"{result.stdout}\n{result.stderr}"
            )
        transcript = (
            self.config.jobs_root
            / str(job["job_id"])
            / "checks"
            / f"checker-{ordinal}.json"
        ).resolve()
        atomic_write_json(
            transcript,
            {
                "schema_version": "1.0",
                "ordinal": ordinal,
                "csv_path": str(csv_path),
                "csv_sha256": sha256_file(csv_path),
                "command": list(command),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "valid": result.valid,
                "test_scored": False,
            },
        )
        self.repository.record_check(
            str(job["job_id"]),
            ordinal=ordinal,
            csv_path=csv_path,
            command=command,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            transcript_path=transcript,
        )
        return result, transcript

    @staticmethod
    def _require_check_only_command(command: tuple[str, ...], csv_path: Path) -> None:
        flags = [item for item in command if item.startswith("--")]
        forbidden = [item for item in flags if item.startswith(("--score", "--make"))]
        unexpected = set(flags).difference({"--data_dir", "--split", "--check"})
        try:
            split_index = command.index("--split")
            test_split = command[split_index + 1] == "test"
        except (ValueError, IndexError):
            test_split = False
        names_submit = any(Path(item).name == "submit.py" for item in command)
        names_csv = str(csv_path.resolve()) in command
        if (
            command.count("--check") != 1
            or command.count("--split") != 1
            or not test_split
            or forbidden
            or unexpected
            or not names_submit
            or not names_csv
        ):
            raise SubmissionCoordinatorError(
                "organizer checker command must be exactly test-check-only in capability"
            )

    def _check(self, job: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
        csv_path = Path(str(job["csv_path"])).resolve(strict=True)
        if sha256_file(csv_path) != job["csv_sha256"]:
            raise SubmissionCoordinatorError("submission CSV drifted before first check")
        _, transcript = self._run_checker(job, csv_path, ordinal)
        return self.repository.transition(
            str(job["job_id"]),
            SubmissionState.CSV_BUILT,
            SubmissionState.FIRST_CHECK_VALID,
            evidence={"checker_transcript_sha256": sha256_file(transcript)},
        )

    def _stage_and_check(
        self, job: Mapping[str, Any], source: VerifiedSource
    ) -> dict[str, Any]:
        state = SubmissionState(job["state"])
        staging = (self.config.jobs_root / str(job["job_id"]) / "final.staging").resolve()
        if state == SubmissionState.FIRST_CHECK_VALID:
            job = self.repository.transition(
                str(job["job_id"]),
                SubmissionState.FIRST_CHECK_VALID,
                SubmissionState.STAGING,
                evidence={"staging_path": str(staging)},
                updates={"staging_path": str(staging)},
            )
        stager = self.dependencies.bundle_stager or DefaultBundleStager()
        first_transcript = (
            self.config.jobs_root / str(job["job_id"]) / "checks" / "checker-1.json"
        ).resolve(strict=True)
        produced = stager(
            StagingRequest(
                job_id=str(job["job_id"]),
                output_dir=staging,
                source=source,
                prediction_path=Path(str(job["prediction_path"])).resolve(strict=True),
                submission_path=Path(str(job["csv_path"])).resolve(strict=True),
                first_check_transcript=first_transcript,
            )
        ).resolve(strict=True)
        if produced != staging:
            raise SubmissionCoordinatorError("bundle stager returned an unexpected directory")
        copied_report = staging / "source-report"
        if report_fingerprint(copied_report) != job["source_report_sha256"]:
            raise SubmissionCoordinatorError("staged source report differs from completed run")
        copied_csv = staging / "submission.csv"
        if not copied_csv.is_file() or sha256_file(copied_csv) != job["csv_sha256"]:
            raise SubmissionCoordinatorError("staged submission differs from the first checked CSV")
        _, transcript = self._run_checker(job, copied_csv, ordinal=2)
        return self.repository.transition(
            str(job["job_id"]),
            SubmissionState.STAGING,
            SubmissionState.SECOND_CHECK_VALID,
            evidence={"checker_transcript_sha256": sha256_file(transcript)},
        )

    def _seal(self, job: Mapping[str, Any], source: VerifiedSource) -> dict[str, Any]:
        staging = Path(str(job["staging_path"])).resolve()
        sealed = (self.config.jobs_root / str(job["job_id"]) / "sealed").resolve()
        if sealed.exists():
            manifest_path = sealed / FilesystemHandoff.manifest_name
            seal_hash = self._validate_sealed(sealed, job, source)
        else:
            if not staging.is_dir():
                raise SubmissionCoordinatorError("staging directory is missing")
            second = (
                self.config.jobs_root / str(job["job_id"]) / "checks" / "checker-2.json"
            ).resolve(strict=True)
            checks_dir = staging / "checks"
            checks_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(second, checks_dir / "second.json")
            required_files = (
                staging / "submission.csv",
                staging / "test_predictions.npz",
                staging / "best-valid" / "model" / "model_bundle.json",
                staging / "source-report" / "evidence_index.json",
                staging / "source-report" / "events.jsonl",
                staging / "source-report" / "resources.json",
                staging / "checks" / "first.json",
                staging / "checks" / "second.json",
            )
            if any(not path.is_file() for path in required_files):
                raise SubmissionCoordinatorError("staged final bundle is incomplete")
            if sha256_file(staging / "submission.csv") != job["csv_sha256"]:
                raise SubmissionCoordinatorError("staged CSV drifted after second check")
            if sha256_file(staging / "test_predictions.npz") != job["prediction_sha256"]:
                raise SubmissionCoordinatorError("staged predictions drifted")
            validate_model_bundle(
                staging / "best-valid" / "model" / "model_bundle.json",
                expected_commit_sha=source.commit_sha,
                expected_config_sha256=source.config_sha256,
            )
            artifact_hashes: dict[str, dict[str, Any]] = {}
            manifest_path = staging / FilesystemHandoff.manifest_name
            for path in sorted(staging.rglob("*")):
                if path == manifest_path:
                    continue
                if path.is_symlink():
                    raise SubmissionCoordinatorError("sealed bundles may not contain symlinks")
                if path.is_file():
                    relative = path.relative_to(staging).as_posix()
                    artifact_hashes[relative] = {
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
            atomic_write_json(
                manifest_path,
                {
                    "schema_version": "1.0",
                    "kind": "sealed_final_submission",
                    "job_id": job["job_id"],
                    "source_run_id": job["source_run_id"],
                    "source_best_valid_sha256": job["best_valid_sha256"],
                    "source_report_sha256": job["source_report_sha256"],
                    "commit_sha": source.commit_sha,
                    "config_sha256": source.config_sha256,
                    "expected_test_rows": self.config.expected_test_rows,
                    "submission_sha256": job["csv_sha256"],
                    "prediction_sha256": job["prediction_sha256"],
                    "organizer_checks": 2,
                    "test_scored": False,
                    "artifacts": artifact_hashes,
                },
            )
            seal_hash = sha256_file(manifest_path)
            os.replace(staging, sealed)
            seal_hash = self._validate_sealed(sealed, job, source)
        expected = job.get("seal_sha256")
        if expected is not None and expected != seal_hash:
            raise SubmissionCoordinatorError("sealed manifest changed during resume")
        return self.repository.transition(
            str(job["job_id"]),
            SubmissionState.SECOND_CHECK_VALID,
            SubmissionState.SEALED,
            evidence={"seal_sha256": seal_hash, "atomic_rename": True},
            updates={"sealed_path": str(sealed), "seal_sha256": seal_hash},
        )

    def _validate_sealed(
        self, sealed: Path, job: Mapping[str, Any], source: VerifiedSource
    ) -> str:
        manifest_path = sealed / FilesystemHandoff.manifest_name
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise SubmissionCoordinatorError("partial sealed directory has no regular manifest")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SubmissionCoordinatorError(f"sealed manifest is invalid: {error}") from error
        expected = {
            "kind": "sealed_final_submission",
            "job_id": job["job_id"],
            "source_run_id": job["source_run_id"],
            "source_best_valid_sha256": job["best_valid_sha256"],
            "source_report_sha256": job["source_report_sha256"],
            "commit_sha": source.commit_sha,
            "config_sha256": source.config_sha256,
            "expected_test_rows": self.config.expected_test_rows,
            "submission_sha256": job["csv_sha256"],
            "prediction_sha256": job["prediction_sha256"],
            "organizer_checks": 2,
            "test_scored": False,
        }
        conflicts = {
            name: {"expected": value, "observed": manifest.get(name)}
            for name, value in expected.items()
            if manifest.get(name) != value
        }
        if conflicts or not isinstance(manifest.get("artifacts"), dict):
            raise SubmissionCoordinatorError(
                "sealed manifest provenance mismatch: " + json.dumps(conflicts, sort_keys=True)
            )
        recorded = manifest["artifacts"]
        actual: set[str] = set()
        for path in sorted(sealed.rglob("*")):
            if path == manifest_path:
                continue
            if path.is_symlink():
                raise SubmissionCoordinatorError("sealed bundles may not contain symlinks")
            if not path.is_file():
                continue
            relative = path.relative_to(sealed).as_posix()
            actual.add(relative)
            reference = recorded.get(relative)
            if (
                not isinstance(reference, dict)
                or reference.get("sha256") != sha256_file(path)
                or reference.get("size_bytes") != path.stat().st_size
            ):
                raise SubmissionCoordinatorError(f"sealed artifact drifted: {relative}")
        if actual != set(recorded):
            raise SubmissionCoordinatorError("sealed manifest artifact inventory drifted")
        return sha256_file(manifest_path)
