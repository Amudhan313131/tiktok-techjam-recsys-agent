"""Build and validate organizer-aligned submissions without scoring test labels."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from rex.data.manifest import canonical_json_bytes, verify_starter_manifest
from rex.data.views import FeatureView, load_feature_view
from rex.execution.artifacts import ArtifactError, atomic_write_json, load_prediction_artifact
from rex.execution.limits import limits_for_request, resource_limit_preexec
from rex.execution.docker_lease import (
    DockerLeaseError,
    archive_closed_docker_lease,
    read_docker_lease,
    runtime_handle_from_docker_lease,
)
from rex.execution.runtime import (
    ExecutionOutcome,
    ExecutionResult,
    ExecutionRuntime,
    ExecutionRuntimeError,
    ExecutionSpec,
    RuntimeMount,
    RuntimeLifecycleState,
    production_runtime,
)
from rex.execution.runtime_docker import docker_security_policy_sha256
from rex.execution.sandbox import (
    SandboxMode,
    SandboxPolicy,
    production_backend,
    sanitized_environment,
)


TEST_ROW_COUNT = 170_588
SUBMISSION_HEADER = ("row_id", "user_id", "video_id", "score")


class SubmissionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SubmissionValidation:
    valid: bool
    command: tuple[str, ...]
    stdout: str
    stderr: str
    returncode: int
    sandbox_evidence: dict[str, object] = field(default_factory=dict)


def _persist_docker_checker_result(
    path: Path,
    result: ExecutionResult,
    *,
    request_sha256: str,
    execution_sha256: str,
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": "rex.docker-submission-result.v1",
            "request_sha256": request_sha256,
            "execution_sha256": execution_sha256,
            "outcome": result.outcome.value,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "wall_seconds": result.wall_seconds,
            "oom_killed": result.oom_killed,
            "timed_out": result.timed_out,
        },
    )


def _read_docker_checker_result(
    path: Path,
    *,
    request_sha256: str,
    execution_sha256: str,
) -> ExecutionResult | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != "rex.docker-submission-result.v1"
            or payload.get("request_sha256") != request_sha256
            or payload.get("execution_sha256") != execution_sha256
        ):
            return None
        return ExecutionResult(
            outcome=ExecutionOutcome(str(payload["outcome"])),
            exit_code=(None if payload.get("exit_code") is None else int(payload["exit_code"])),
            stdout=str(payload["stdout"]),
            stderr=str(payload["stderr"]),
            wall_seconds=float(payload["wall_seconds"]),
            oom_killed=bool(payload["oom_killed"]),
            timed_out=bool(payload["timed_out"]),
        )
    except (KeyError, OSError, UnicodeError, ValueError, TypeError):
        return None


def build_submission(
    prediction_path: str | Path,
    csv_path: str | Path,
    *,
    expected_features: FeatureView | str | Path,
    expected_rows: int = TEST_ROW_COUNT,
) -> Path:
    """Write a CSV only after exact canonical-view alignment has been proven.

    ``expected_features`` is deliberately mandatory: a prediction artifact is
    self-consistent even when it was generated for the wrong split, so production
    submission construction must compare it with the frozen canonical test view.
    """

    if expected_rows <= 0:
        raise SubmissionError("expected submission row count must be positive")
    view = (
        expected_features
        if isinstance(expected_features, FeatureView)
        else load_feature_view(expected_features)
    )
    if view.rows != expected_rows:
        raise SubmissionError(
            f"canonical feature row count mismatch: expected {expected_rows}, observed {view.rows}"
        )
    try:
        arrays = load_prediction_artifact(prediction_path, expected_features=view)
    except (ArtifactError, OSError, ValueError) as error:
        raise SubmissionError(f"invalid test prediction artifact: {error}") from error
    if len(arrays["score"]) != expected_rows:
        raise SubmissionError(
            f"prediction row count mismatch: expected {expected_rows}, "
            f"observed {len(arrays['score'])}"
        )

    destination = Path(csv_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(SUBMISSION_HEADER)
            for row_id, user_id, video_id, score in zip(
                arrays["row_id"],
                arrays["user_id"],
                arrays["video_id"],
                arrays["score"],
                strict=True,
            ):
                writer.writerow([int(row_id), str(user_id), str(video_id), f"{float(score):.17g}"])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def validate_submission_matches_predictions(
    csv_path: str | Path,
    prediction_path: str | Path,
    *,
    expected_features: FeatureView | str | Path,
    expected_rows: int = TEST_ROW_COUNT,
) -> None:
    """Prove a CSV is an order-preserving rendering of one prediction artifact."""

    view = (
        expected_features
        if isinstance(expected_features, FeatureView)
        else load_feature_view(expected_features)
    )
    if view.rows != expected_rows:
        raise SubmissionError(
            f"canonical feature row count mismatch: expected {expected_rows}, observed {view.rows}"
        )
    try:
        arrays = load_prediction_artifact(prediction_path, expected_features=view)
    except (ArtifactError, OSError, ValueError) as error:
        raise SubmissionError(f"invalid test prediction artifact: {error}") from error
    path = Path(csv_path)
    observed_rows = 0
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            if tuple(next(reader, ())) != SUBMISSION_HEADER:
                raise SubmissionError("submission header must be row_id,user_id,video_id,score")
            for observed_rows, record in enumerate(reader, start=1):
                if len(record) != 4:
                    raise SubmissionError(
                        f"submission row {observed_rows + 1} must contain exactly four fields"
                    )
                if observed_rows > expected_rows:
                    raise SubmissionError(
                        "submission contains more rows than the canonical test view"
                    )
                index = observed_rows - 1
                expected = (
                    str(int(arrays["row_id"][index])),
                    str(arrays["user_id"][index]),
                    str(arrays["video_id"][index]),
                )
                if tuple(record[:3]) != expected:
                    raise SubmissionError(
                        f"submission alignment mismatch at data row {observed_rows}"
                    )
                try:
                    score = float(record[3])
                except ValueError as error:
                    raise SubmissionError(
                        f"submission score is not numeric at data row {observed_rows}"
                    ) from error
                if score != float(arrays["score"][index]):
                    raise SubmissionError(
                        f"submission score differs from prediction artifact at data row {observed_rows}"
                    )
    except (OSError, UnicodeError) as error:
        raise SubmissionError(f"cannot read submission CSV: {error}") from error
    if observed_rows != expected_rows:
        raise SubmissionError(
            f"submission row count mismatch: expected {expected_rows}, observed {observed_rows}"
        )


def _checker_command(
    csv_path: Path, data_dir: Path, split: str, submit_py: Path
) -> tuple[str, ...]:
    command = (
        sys.executable,
        str(submit_py),
        str(csv_path),
        "--data_dir",
        str(data_dir),
        "--split",
        split,
        "--check",
    )
    forbidden = {"--score", "--make"}.intersection(command)
    if forbidden or command[-1] != "--check":  # pragma: no cover - invariant guard
        raise SubmissionError("organizer checker command attempted a forbidden mode")
    return command


def validate_submission(
    csv_path: str | Path,
    *,
    data_dir: str | Path,
    split: str,
    timeout_seconds: int = 180,
    sandbox_mode: SandboxMode = SandboxMode.PRODUCTION,
    execution_runtime: ExecutionRuntime | None = None,
) -> SubmissionValidation:
    """Run only the frozen organizer check mode under a no-network sandbox."""

    if split not in {"valid", "test"}:
        raise SubmissionError(f"unsupported submission split: {split}")
    if timeout_seconds <= 0:
        raise SubmissionError("submission checker timeout must be positive")
    submission = Path(csv_path).resolve()
    data_root = Path(data_dir).resolve()
    if not submission.is_file():
        raise SubmissionError(f"submission CSV is missing: {submission}")
    if not data_root.is_dir():
        raise SubmissionError(f"submission data directory is missing: {data_root}")
    starter = verify_starter_manifest()
    submit_py = starter.root / "submit.py"
    command = _checker_command(submission, data_root, split, submit_py)
    mode = SandboxMode(sandbox_mode)
    runtime_kind = os.environ.get("REX_PRODUCTION_RUNTIME", "docker").strip().lower()
    docker_production = mode == SandboxMode.PRODUCTION and runtime_kind != "native_macos"
    temp_parent: Path | None = None
    if docker_production:
        runs_value = os.environ.get("REX_RUNS_ROOT", "")
        if not runs_value:
            raise SubmissionError("Docker submission checking requires REX_RUNS_ROOT")
        runs_root = Path(runs_value).resolve(strict=True)
        temp_parent = runs_root / "submission-checks"
        temp_parent.mkdir(parents=True, exist_ok=True)
    if (
        mode == SandboxMode.PRODUCTION
        and not docker_production
        and os.environ.get("REX_ALLOW_NATIVE_MACOS_ROLLBACK") != "1"
    ):
        raise SubmissionError("native macOS rollback requires explicit authorization")

    try:
        with tempfile.TemporaryDirectory(
            prefix="rex-submission-check-", dir=temp_parent
        ) as temporary:
            temp_root = Path(temporary).resolve()
            environment = sanitized_environment(
                workspace=starter.root,
                temp_dir=temp_root,
            )
            limits = limits_for_request(timeout_seconds, max_memory_mb=4096)
            if docker_production:
                docker_command = ("python", *command[1:])
                environment = {
                    "HOME": "/tmp",
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": "0",
                    "TMPDIR": "/tmp",
                }
                request_sha = hashlib.sha256(
                    canonical_json_bytes(
                        {
                            "command": list(docker_command),
                            "submission_sha256": hashlib.sha256(
                                submission.read_bytes()
                            ).hexdigest(),
                            "starter_manifest_sha256": starter.manifest_sha256,
                            "split": split,
                        }
                    )
                ).hexdigest()
                execution_sha = hashlib.sha256(
                    canonical_json_bytes(
                        {
                            "request_sha256": request_sha,
                            "docker_security_policy_sha256": docker_security_policy_sha256(),
                        }
                    )
                ).hexdigest()
                runtime = production_runtime() if execution_runtime is None else execution_runtime
                if os.geteuid() == 0:
                    raise SubmissionError("production Docker controller must not run as root")
                lease_root = temp_parent / "leases"
                lease_root.mkdir(parents=True, exist_ok=True)
                lease_path = lease_root / f"{request_sha}.json"
                durable_result_path = lease_root / f"{request_sha}.result.json"
                recovery_path = lease_root / f"{request_sha}.recovery.json"
                specification = ExecutionSpec(
                    command=docker_command,
                    working_directory=str(starter.root.resolve()),
                    mounts=(
                        RuntimeMount(starter.root.resolve(), str(starter.root.resolve()), True),
                        RuntimeMount(data_root, str(data_root), True),
                        RuntimeMount(submission, str(submission), True),
                        RuntimeMount(temp_root, "/output", False),
                    ),
                    environment=environment,
                    timeout_seconds=float(timeout_seconds),
                    memory_bytes=4096 * 1024 * 1024,
                    nano_cpus=1_000_000_000,
                    pids_limit=64,
                    run_id="submission-check",
                    experiment_id=split,
                    attempt_id=f"checker:{request_sha[:24]}",
                    request_sha256=request_sha,
                    execution_sha256=execution_sha,
                    lease_path=lease_path,
                    user=f"{os.geteuid()}:{os.getegid()}",
                )
                handle = None
                recovered_handle = None
                result = None
                recovery_evidence: dict[str, object] | None = None
                try:
                    if lease_path.is_file():
                        prior = read_docker_lease(lease_path)
                        if (
                            prior.request_sha256 != request_sha
                            or prior.execution_sha256 != execution_sha
                        ):
                            raise SubmissionError(
                                "Docker submission lease belongs to a different request"
                            )
                        recovery = runtime.recover(prior)
                        recovered = read_docker_lease(lease_path)
                        recovered_handle = runtime_handle_from_docker_lease(
                            recovered, timeout_seconds=float(timeout_seconds)
                        )
                        status = runtime.inspect(recovered_handle)
                        container_absent = status.state == RuntimeLifecycleState.REMOVED
                        durable_result = _read_docker_checker_result(
                            durable_result_path,
                            request_sha256=request_sha,
                            execution_sha256=execution_sha,
                        )
                        recovery_evidence = {
                            "outcome": recovery.outcome,
                            "container_id": recovery.container_id,
                            "terminated": recovery.terminated,
                            "container_absent": container_absent,
                            "durable_result_reused": durable_result is not None,
                        }
                        atomic_write_json(recovery_path, recovery_evidence)
                        if container_absent:
                            if durable_result is not None:
                                result = durable_result
                            else:
                                recovery_evidence["lease_archive"] = archive_closed_docker_lease(
                                    lease_path,
                                    related_evidence_paths=(recovery_path,),
                                )
                                atomic_write_json(recovery_path, recovery_evidence)
                                recovered_handle = None
                        else:
                            result = durable_result or runtime.collect(recovered_handle)
                            _persist_docker_checker_result(
                                durable_result_path,
                                result,
                                request_sha256=request_sha,
                                execution_sha256=execution_sha,
                            )
                            if (
                                _read_docker_checker_result(
                                    durable_result_path,
                                    request_sha256=request_sha,
                                    execution_sha256=execution_sha,
                                )
                                is None
                            ):
                                raise SubmissionError(
                                    "Docker submission durable result verification failed"
                                )
                            runtime.cleanup(recovered_handle)
                            if result.outcome in {
                                ExecutionOutcome.TIMEOUT,
                                ExecutionOutcome.OOM,
                                ExecutionOutcome.INTERRUPTED,
                            }:
                                recovery_evidence["lease_archive"] = archive_closed_docker_lease(
                                    lease_path,
                                    related_evidence_paths=(recovery_path,),
                                )
                                atomic_write_json(recovery_path, recovery_evidence)
                                durable_result_path.unlink(missing_ok=True)
                                recovered_handle = None
                                result = None
                    if result is None:
                        handle = runtime.launch(specification)
                        result = runtime.collect(handle)
                        _persist_docker_checker_result(
                            durable_result_path,
                            result,
                            request_sha256=request_sha,
                            execution_sha256=execution_sha,
                        )
                        if (
                            _read_docker_checker_result(
                                durable_result_path,
                                request_sha256=request_sha,
                                execution_sha256=execution_sha,
                            )
                            is None
                        ):
                            raise SubmissionError(
                                "Docker submission durable result verification failed"
                            )
                    else:
                        handle = recovered_handle
                except (DockerLeaseError, ExecutionRuntimeError) as error:
                    raise SubmissionError(
                        f"Docker submission checker failed closed: {error}"
                    ) from error
                finally:
                    if handle is not None and handle is not recovered_handle:
                        try:
                            runtime.cleanup(handle)
                        except ExecutionRuntimeError as error:
                            raise SubmissionError(
                                f"Docker submission checker cleanup failed closed: {error}"
                            ) from error
                completed = subprocess.CompletedProcess(
                    args=list(command),
                    returncode=result.exit_code if result.exit_code is not None else 1,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
                sandbox_evidence = {
                    "schema_version": "rex.docker-submission-check.v1",
                    "mode": "production",
                    "backend": "docker",
                    "sandboxed": True,
                    "network_allowed": False,
                    "starter_manifest_sha256": starter.manifest_sha256,
                    "submit_py_sha256": starter.hashes["submit.py"],
                    "logical_command": list(command),
                    "container_id": handle.container_id,
                    "worker_image_digest": handle.worker_image_digest,
                    "docker_security_policy_sha256": docker_security_policy_sha256(),
                    "request_sha256": request_sha,
                    "execution_sha256": execution_sha,
                    "durable_result_path": str(durable_result_path),
                    "lease_sha256": hashlib.sha256(
                        specification.lease_path.read_bytes()
                    ).hexdigest(),
                    "recovery": recovery_evidence,
                    "checked_at_epoch_ms": int(time.time() * 1000),
                }
            elif mode == SandboxMode.PRODUCTION:
                policy = SandboxPolicy(
                    workspace=starter.root.resolve(),
                    read_paths=(starter.root.resolve(), data_root, submission),
                    write_paths=(temp_root,),
                    network_allowed=False,
                    resource_limits=limits,
                )
                prepared = production_backend().prepare(
                    policy,
                    command,
                    environment,
                    temp_root / "checker.sb",
                )
                execution_command = prepared.command
                execution_environment = prepared.environment
                preexec_fn = prepared.preexec_fn
                sandbox_evidence = {
                    **prepared.evidence,
                    "starter_manifest_sha256": starter.manifest_sha256,
                    "submit_py_sha256": starter.hashes["submit.py"],
                    "logical_command": list(command),
                }
            else:
                execution_command = command
                execution_environment = environment
                preexec_fn = resource_limit_preexec(limits)
                sandbox_evidence = {
                    "schema_version": "1.0",
                    "mode": "fixture",
                    "sandboxed": False,
                    "network_allowed": None,
                    "starter_manifest_sha256": starter.manifest_sha256,
                    "submit_py_sha256": starter.hashes["submit.py"],
                    "logical_command": list(command),
                }
            if not docker_production:
                completed = subprocess.run(
                    execution_command,
                    cwd=starter.root,
                    env=execution_environment,
                    preexec_fn=preexec_fn,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
    except subprocess.TimeoutExpired as error:
        raise SubmissionError(
            f"organizer submission checker exceeded {timeout_seconds} seconds"
        ) from error

    return SubmissionValidation(
        valid=completed.returncode == 0,
        command=command,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
        sandbox_evidence=sandbox_evidence,
    )


def require_valid_submission(*args, **kwargs) -> SubmissionValidation:
    result = validate_submission(*args, **kwargs)
    if not result.valid:
        raise SubmissionError(
            f"organizer submission validation failed ({result.returncode}): "
            f"{result.stdout}\n{result.stderr}"
        )
    return result
