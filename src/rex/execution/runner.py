"""Process-group isolated worker runner with timeout, telemetry, and typed fallback results."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path

from pydantic import ValidationError

from rex.contracts import ArtifactRef, AttemptStatus, RunRequest, RunResult
from rex.data.manifest import canonical_json_bytes, repo_root, sha256_file
from rex.execution.artifacts import ArtifactError, artifact_ref, atomic_write_json
from rex.execution.limits import limits_for_request
from rex.execution.lease import (
    AttemptLock,
    WorkerLeaseError,
    begin_worker_lease,
    close_worker_lease,
    command_sha256,
    recover_orphan_worker,
)
from rex.execution.docker_lease import (
    DockerLeaseError,
    archive_closed_docker_lease,
    read_docker_lease,
    runtime_handle_from_docker_lease,
)
from rex.execution.runtime import (
    ExecutionOutcome,
    ExecutionRuntime,
    ExecutionRuntimeError,
    ExecutionSpec,
    RuntimeMount,
    RuntimeLifecycleState,
    production_runtime,
)
from rex.execution.runtime_docker import docker_security_policy_sha256
from rex.execution.sandbox import (
    PreparedSandbox,
    SandboxError,
    SandboxMode,
    SandboxPolicy,
    fixture_environment,
    production_backend,
    sanitized_environment,
)
from rex.execution.telemetry import ResourceTotals
from rex.models.bundle import validate_model_bundle


_LEASE_WRAPPER = """import os
import subprocess
import sys
import time

launch_gate = sys.argv[2]
deadline = time.monotonic() + 2.0
while not os.path.isfile(launch_gate):
    if time.monotonic() >= deadline:
        raise SystemExit(125)
    time.sleep(0.01)
child = subprocess.Popen(sys.argv[3:])
code = child.wait()
if code < 0:
    os.kill(os.getpid(), -code)
raise SystemExit(code)
"""


def _typed_failure(
    stderr: str,
    timed_out: bool,
    memory_exceeded: bool = False,
    *,
    return_code: int | None = None,
    invalid_artifact: bool = False,
) -> AttemptStatus:
    if memory_exceeded:
        return AttemptStatus.OOM
    if timed_out:
        return AttemptStatus.TIMEOUT
    if invalid_artifact:
        return AttemptStatus.INVALID_ARTIFACT
    if return_code is not None and return_code < 0:
        return AttemptStatus.INTERRUPTED
    lowered = stderr.lower()
    if "memoryerror" in lowered or "out of memory" in lowered or "oom" in lowered:
        return AttemptStatus.OOM
    if "syntaxerror" in lowered:
        return AttemptStatus.SYNTAX
    if "modulenotfounderror" in lowered or "importerror" in lowered:
        return AttemptStatus.IMPORT
    if "nan" in lowered or "non-finite" in lowered:
        return AttemptStatus.NAN
    return AttemptStatus.CRASH


def _contract_result(
    request: RunRequest,
    *,
    error_type: str,
    summary: str,
    command: list[str],
    started: float,
    artifacts: list[ArtifactRef] | None = None,
) -> RunResult:
    return RunResult(
        run_id=request.run_id,
        experiment_id=request.experiment_id,
        attempt_id=request.attempt_id,
        status=AttemptStatus.CONTRACT,
        error_type=error_type,
        error_summary=summary,
        command_sha256=hashlib.sha256(canonical_json_bytes(command)).hexdigest(),
        commit_sha=request.commit_sha,
        config_sha256=request.config_sha256,
        data_view_sha256=request.data_view_sha256,
        environment_sha256=request.environment_sha256,
        artifacts=[] if artifacts is None else artifacts,
        wall_seconds=time.monotonic() - started,
    )


def _lease_wrapper_command(
    command: tuple[str, ...], launch_gate: Path
) -> tuple[tuple[str, ...], str]:
    """Keep a stable process-group leader while the sandbox command runs beneath it."""

    token = secrets.token_hex(16)
    return (
        sys.executable,
        "-I",
        "-c",
        _LEASE_WRAPPER,
        token,
        str(launch_gate),
        *command,
    ), token


def _validated_replay(request: RunRequest, result_path: Path) -> RunResult | None:
    """Return a complete prior worker result, or ``None`` for a partial result."""

    if not result_path.is_file():
        return None
    try:
        result = RunResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        expected_request_hash = hashlib.sha256(
            canonical_json_bytes(request.model_dump(mode="json"))
        ).hexdigest()
        if result.command_sha256 != expected_request_hash:
            raise ValueError("worker result request hash mismatch")
        expected = {
            "run_id": request.run_id,
            "experiment_id": request.experiment_id,
            "attempt_id": request.attempt_id,
            "commit_sha": request.commit_sha,
            "config_sha256": request.config_sha256,
            "data_view_sha256": request.data_view_sha256,
            "environment_sha256": request.environment_sha256,
        }
        for field, value in expected.items():
            if getattr(result, field) != value:
                raise ValueError(f"worker result {field} mismatch")
        for ref in result.artifacts:
            artifact_path = Path(ref.path)
            if not artifact_path.is_file() or sha256_file(artifact_path) != ref.sha256:
                raise ValueError(f"worker replay artifact missing or corrupt: {ref.artifact_id}")
        return result
    except (OSError, UnicodeError, ValidationError, ValueError, json.JSONDecodeError):
        return None


def _append_artifacts(result: RunResult, additions: list[ArtifactRef]) -> RunResult:
    observed = {ref.artifact_id for ref in result.artifacts}
    artifacts = [*result.artifacts]
    for ref in additions:
        if ref.artifact_id not in observed:
            artifacts.append(ref)
            observed.add(ref.artifact_id)
    return result.model_copy(update={"artifacts": artifacts})


def _validated_workspace(
    request: RunRequest,
    trusted_worktree_root: str | Path | None,
) -> Path:
    if request.workspace_path is None:
        return repo_root().resolve()
    workspace = Path(request.workspace_path).resolve()
    if trusted_worktree_root is None:
        raise ValueError("trusted_worktree_root is required for worktree-bound execution")
    trusted_root = Path(trusted_worktree_root).resolve()
    try:
        workspace.relative_to(trusted_root)
    except ValueError as error:
        raise ValueError(f"workspace is outside the trusted worktree root: {workspace}") from error
    if not workspace.is_dir():
        raise ValueError(f"workspace does not exist: {workspace}")
    observed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if observed.returncode != 0:
        raise ValueError(f"workspace is not a Git worktree: {observed.stderr.strip()}")
    if observed.stdout.strip() != request.commit_sha:
        raise ValueError(
            f"workspace commit mismatch: expected {request.commit_sha}, "
            f"observed {observed.stdout.strip()}"
        )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=workspace,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise ValueError("workspace must be a clean Git worktree")
    return workspace


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=2)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:  # pragma: no cover - kernel reaping delay
            pass


def _inside(root: Path, candidate: Path, *, label: str) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise SandboxError(f"{label} is outside its trusted root: {candidate}") from error
    return candidate


def _production_plan(
    request: RunRequest,
    *,
    directory: Path,
    workspace: Path,
    command: list[str],
    request_path: Path,
    result_root: Path,
    temp_root: Path,
    policy_path: Path,
    trusted_output_root: str | Path | None,
    environment_overrides: dict[str, str] | None,
) -> PreparedSandbox:
    """Build and verify the least-authority plan for one production worker."""

    if request.workspace_path is None:
        raise SandboxError("production execution requires a verified candidate worktree")
    if trusted_output_root is None:
        raise SandboxError("production execution requires trusted_output_root")
    output_root = Path(trusted_output_root).resolve()
    _inside(output_root, directory, label="attempt directory")
    output_dir = _inside(output_root, Path(request.output_dir), label="worker output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(request.config_path).resolve(strict=True)
    feature_path = Path(request.feature_view_path).resolve(strict=True)
    read_paths: list[Path] = [workspace, request_path, config_path, feature_path]
    if request.target_view_path is not None:
        read_paths.append(Path(request.target_view_path).resolve(strict=True))
    if request.model_bundle_path is not None:
        try:
            bundle = validate_model_bundle(
                request.model_bundle_path,
                expected_plugin=request.plugin,
                expected_config_sha256=request.config_sha256,
                expected_commit_sha=request.commit_sha,
            )
        except ArtifactError as error:
            raise SandboxError(f"production model bundle is invalid: {error}") from error
        read_paths.extend((bundle.manifest_path, *bundle.member_paths))
    elif request.effective_operation == "predict":
        raise SandboxError("production prediction requires an immutable model bundle")
    temp_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    environment = sanitized_environment(
        workspace=workspace,
        temp_dir=temp_root,
        overrides=environment_overrides,
    )
    policy = SandboxPolicy(
        workspace=workspace,
        read_paths=tuple(dict.fromkeys(path.resolve() for path in read_paths)),
        write_paths=(output_dir.resolve(), result_root.resolve(), temp_root.resolve()),
        network_allowed=False,
        resource_limits=limits_for_request(request.timeout_seconds, request.max_memory_mb),
    )
    return production_backend().prepare(policy, command, environment, policy_path)


def _fixture_plan(workspace: Path, command: list[str]) -> PreparedSandbox:
    environment = fixture_environment(workspace)
    return PreparedSandbox(
        command=tuple(command),
        environment=environment,
        preexec_fn=None,
        evidence={
            "schema_version": "1.0",
            "mode": SandboxMode.FIXTURE.value,
            "backend": "trusted-fixture-unsandboxed",
            "sandboxed": False,
            "reason": "unsandboxed execution is restricted to explicit fixture/test mode",
            "environment_keys": sorted(environment),
        },
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _docker_worker_request(
    request: RunRequest,
    attempt_dir: Path,
    *,
    trusted_worktree_root: str | Path | None,
    trusted_output_root: str | Path | None,
    runtime: ExecutionRuntime | None,
) -> RunResult:
    """Execute generated code only in a disposable, fail-closed Docker worker."""

    started = time.monotonic()
    directory = attempt_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    request_root = directory / "input"
    result_root = directory / "worker"
    request_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    request_path = request_root / "request.json"
    docker_request_path = request_root / "docker-request.json"
    result_path = result_root / "result.json"
    stdout_path = directory / "stdout.log"
    stderr_path = directory / "stderr.log"
    evidence_path = directory / "sandbox_evidence.json"
    lease_path = directory / "worker_lease.json"
    recovery_path = directory / "worker_recovery.json"
    replay_path = directory / "worker_replay.json"
    original_payload = request.model_dump(mode="json", by_alias=True)
    request_sha = hashlib.sha256(canonical_json_bytes(original_payload)).hexdigest()
    command = (
        "python",
        "-m",
        "rex.execution.worker",
        "--request",
        "/request.json",
        "--result",
        "/output/.rex-worker-result.json",
    )
    try:
        attempt_lock = AttemptLock.acquire(directory / "worker_lease.lock")
    except WorkerLeaseError as error:
        return _contract_result(
            request,
            error_type="WorkerLeaseConflict",
            summary=str(error),
            command=list(command),
            started=started,
        )
    try:
        if request_path.is_file():
            prior = RunRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
            prior_sha = hashlib.sha256(
                canonical_json_bytes(prior.model_dump(mode="json", by_alias=True))
            ).hexdigest()
            if prior_sha != request_sha:
                raise WorkerLeaseError("attempt directory belongs to a different request hash")
        atomic_write_json(request_path, original_payload)
        replay = _validated_replay(request, result_path)
        workspace = _validated_workspace(request, trusted_worktree_root)
        if trusted_output_root is None:
            raise SandboxError("production Docker execution requires trusted_output_root")
        output_root = Path(trusted_output_root).resolve(strict=True)
        _inside(output_root, directory, label="attempt directory")
        output_dir = _inside(output_root, Path(request.output_dir), label="worker output directory")
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = Path(request.config_path).resolve(strict=True)
        feature_path = Path(request.feature_view_path).resolve(strict=True)
        target_path = (
            Path(request.target_view_path).resolve(strict=True)
            if request.target_view_path is not None
            else None
        )
        bundle_root: Path | None = None
        if request.model_bundle_path is not None:
            bundle = validate_model_bundle(
                request.model_bundle_path,
                expected_plugin=request.plugin,
                expected_config_sha256=request.config_sha256,
                expected_commit_sha=request.commit_sha,
            )
            bundle_root = bundle.manifest_path.parent.resolve()
        elif request.effective_operation == "predict":
            raise SandboxError("production prediction requires an immutable model bundle")
        worker_request = request.model_copy(update={"output_dir": "/output"})
        worker_payload = worker_request.model_dump(mode="json", by_alias=True)
        worker_request_sha = hashlib.sha256(canonical_json_bytes(worker_payload)).hexdigest()
        atomic_write_json(docker_request_path, worker_payload)
        mounts: list[RuntimeMount] = [
            RuntimeMount(workspace, str(workspace), True),
            RuntimeMount(docker_request_path, "/request.json", True),
            RuntimeMount(output_dir, "/output", False),
        ]
        for path in (config_path, feature_path, target_path, bundle_root):
            if path is None or _is_relative_to(path, workspace):
                continue
            if any(item.source == path for item in mounts):
                continue
            mounts.append(RuntimeMount(path, str(path), True))
        environment = {
            "HOME": "/tmp",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": f"{workspace}/src",
            "REX_SOURCE_ROOT": str(workspace),
            "TMPDIR": "/tmp",
            "XDG_CACHE_HOME": "/tmp/cache",
        }
        execution_sha = hashlib.sha256(
            canonical_json_bytes(
                {
                    "request_sha256": request_sha,
                    "worker_request_sha256": worker_request_sha,
                    "command": list(command),
                    "mounts": [
                        {
                            "source": str(item.source),
                            "target": item.target,
                            "read_only": item.read_only,
                        }
                        for item in mounts
                    ],
                    "environment": environment,
                    "environment_sha256": request.environment_sha256,
                    "docker_security_policy_sha256": docker_security_policy_sha256(),
                }
            )
        ).hexdigest()
        selected_runtime = production_runtime() if runtime is None else runtime
        recovered_handle = None
        recovered_result = None
        recovery_archive: dict[str, object] | None = None
        worker_result_path = output_dir / ".rex-worker-result.json"
        if lease_path.is_file():
            prior_lease = read_docker_lease(lease_path)
            if prior_lease.request_sha256 != request_sha:
                raise DockerLeaseError("attempt lease belongs to a different request")
            if prior_lease.execution_sha256 != execution_sha:
                raise DockerLeaseError("attempt lease belongs to a different execution")
            recovery = selected_runtime.recover(prior_lease)
            recovered_lease = read_docker_lease(lease_path)
            prior_handle = runtime_handle_from_docker_lease(
                recovered_lease,
                timeout_seconds=min(float(request.timeout_seconds), 21_600.0),
            )
            prior_status = selected_runtime.inspect(prior_handle)
            container_absent = prior_status.state == RuntimeLifecycleState.REMOVED
            prior_result = None if container_absent else selected_runtime.collect(prior_handle)
            reuse_completed_result = replay is not None or (
                worker_result_path.is_file()
                and prior_result is not None
                and prior_result.outcome
                not in {
                    ExecutionOutcome.TIMEOUT,
                    ExecutionOutcome.OOM,
                    ExecutionOutcome.INTERRUPTED,
                }
            )
            recovery_event = {
                "schema_version": "rex.docker-recovery.v1",
                "outcome": recovery.outcome,
                "container_id": recovery.container_id,
                "terminated": recovery.terminated,
                "detail": recovery.detail,
                "container_absent": container_absent,
                "completed_result_reused": reuse_completed_result,
                "recorded_at_epoch_ms": int(time.time() * 1000),
            }
            if replay is not None:
                if not container_absent:
                    selected_runtime.cleanup(prior_handle)
            elif container_absent:
                atomic_write_json(recovery_path, recovery_event)
                recovery_archive = archive_closed_docker_lease(
                    lease_path, related_evidence_paths=(recovery_path,)
                )
                atomic_write_json(
                    recovery_path,
                    {
                        "schema_version": "rex.docker-recovery.v1",
                        "outcome": "closed-lease-container-absent",
                        "container_id": prior_lease.container_id,
                        "completed_result_reused": replay is not None,
                        "lease_archive": recovery_archive,
                        "recorded_at_epoch_ms": int(time.time() * 1000),
                    },
                )
                worker_result_path.unlink(missing_ok=True)
            elif reuse_completed_result:
                atomic_write_json(recovery_path, recovery_event)
                recovered_handle = prior_handle
                recovered_result = prior_result
            else:
                atomic_write_json(recovery_path, recovery_event)
                selected_runtime.cleanup(prior_handle)
                recovery_archive = archive_closed_docker_lease(
                    lease_path, related_evidence_paths=(recovery_path,)
                )
                atomic_write_json(
                    recovery_path,
                    {
                        "schema_version": "rex.docker-recovery.v1",
                        "outcome": "incomplete-worker-relaunched",
                        "container_id": prior_lease.container_id,
                        "lease_archive": recovery_archive,
                        "recorded_at_epoch_ms": int(time.time() * 1000),
                    },
                )
                worker_result_path.unlink(missing_ok=True)
        if replay is not None:
            atomic_write_json(
                replay_path,
                {
                    "schema_version": "1.0",
                    "outcome": "complete-result-replayed",
                    "request_sha256": request_sha,
                    "result_sha256": sha256_file(result_path),
                    "replayed_at_epoch_ms": int(time.time() * 1000),
                },
            )
            additions = [artifact_ref(replay_path, "worker_replay")]
            for path, kind in (
                (stdout_path, "stdout"),
                (stderr_path, "stderr"),
                (evidence_path, "sandbox_evidence"),
                (lease_path, "worker_lease"),
                (recovery_path, "worker_recovery"),
            ):
                if path.is_file():
                    additions.append(artifact_ref(path, kind))
            archived_path = (
                Path(str(recovery_archive["lease_path"])) if recovery_archive is not None else None
            )
            if archived_path is not None and archived_path.is_file():
                additions.append(artifact_ref(archived_path, "worker_lease_archive"))
            return _append_artifacts(replay, additions)
        if os.geteuid() == 0:
            raise SandboxError("production Docker controller must not run as root")
        specification = ExecutionSpec(
            command=command,
            working_directory=str(workspace),
            mounts=tuple(mounts),
            environment=environment,
            timeout_seconds=min(
                float(request.timeout_seconds),
                max(0.1, (request.deadline_epoch_ms / 1000) - time.time()),
            ),
            memory_bytes=max(64, request.max_memory_mb or 2048) * 1024 * 1024,
            nano_cpus=1_000_000_000,
            pids_limit=128,
            run_id=request.run_id,
            experiment_id=request.experiment_id,
            attempt_id=request.attempt_id,
            request_sha256=request_sha,
            execution_sha256=execution_sha,
            lease_path=lease_path,
            user=f"{os.geteuid()}:{os.getegid()}",
        )
        handle = recovered_handle or selected_runtime.launch(specification)
        runtime_result = recovered_result
        if runtime_result is None:
            runtime_result = selected_runtime.collect(handle)
        stdout_path.write_text(runtime_result.stdout, encoding="utf-8")
        stderr_path.write_text(runtime_result.stderr, encoding="utf-8")
        atomic_write_json(
            evidence_path,
            {
                "schema_version": "rex.docker-worker-evidence.v1",
                "mode": SandboxMode.PRODUCTION.value,
                "backend": "docker",
                "sandboxed": True,
                "container_id": handle.container_id,
                "container_name": handle.container_name,
                "worker_image_digest": handle.worker_image_digest,
                "worker_image_id": handle.worker_image_id,
                "daemon_identity": handle.daemon_identity,
                "request_sha256": request_sha,
                "worker_request_sha256": worker_request_sha,
                "execution_sha256": execution_sha,
                "docker_security_policy_sha256": docker_security_policy_sha256(),
                "outcome": runtime_result.outcome.value,
                "exit_code": runtime_result.exit_code,
                "timed_out": runtime_result.timed_out,
                "oom_killed": runtime_result.oom_killed,
                "environment_keys": sorted(environment),
                "mounts": [{"target": item.target, "read_only": item.read_only} for item in mounts],
            },
        )
        assert runtime_result is not None
        additions: list[ArtifactRef] = [
            artifact_ref(stdout_path, "stdout"),
            artifact_ref(stderr_path, "stderr"),
            artifact_ref(evidence_path, "sandbox_evidence"),
            artifact_ref(lease_path, "worker_lease"),
        ]
        if recovery_path.is_file():
            additions.append(artifact_ref(recovery_path, "worker_recovery"))
        if recovery_archive is not None:
            archived_lease = Path(str(recovery_archive["lease_path"]))
            if archived_lease.is_file():
                additions.append(artifact_ref(archived_lease, "worker_lease_archive"))
        durable_result: RunResult
        if worker_result_path.is_file() and runtime_result.outcome not in {
            ExecutionOutcome.TIMEOUT,
            ExecutionOutcome.OOM,
        }:
            worker_result = RunResult.model_validate_json(
                worker_result_path.read_text(encoding="utf-8")
            )
            if worker_result.command_sha256 != worker_request_sha:
                raise ValueError("Docker worker result request hash mismatch")
            translated: list[ArtifactRef] = []
            worker_output = Path("/output")
            for ref in worker_result.artifacts:
                worker_path = Path(ref.path)
                try:
                    relative = worker_path.relative_to(worker_output)
                except ValueError as error:
                    raise ValueError("Docker worker artifact escaped /output") from error
                translated_path = (output_dir / relative).resolve()
                _inside(output_dir, translated_path, label="Docker worker artifact")
                if not translated_path.is_file() or sha256_file(translated_path) != ref.sha256:
                    raise ValueError("Docker worker artifact is missing or corrupt")
                translated.append(ref.model_copy(update={"path": str(translated_path)}))
            durable_result = worker_result.model_copy(
                update={
                    "command_sha256": request_sha,
                    "artifacts": [*translated, *additions],
                    "wall_seconds": runtime_result.wall_seconds,
                }
            )
        else:
            if runtime_result.outcome == ExecutionOutcome.OOM:
                status = AttemptStatus.OOM
                error_type = "MemoryLimit"
            elif runtime_result.outcome == ExecutionOutcome.TIMEOUT:
                status = AttemptStatus.TIMEOUT
                error_type = "Timeout"
            elif runtime_result.outcome == ExecutionOutcome.INTERRUPTED:
                status = AttemptStatus.INTERRUPTED
                error_type = "Interrupted"
            else:
                status = _typed_failure(
                    runtime_result.stderr,
                    False,
                    return_code=runtime_result.exit_code,
                )
                error_type = "WorkerFailure"
            durable_result = RunResult(
                run_id=request.run_id,
                experiment_id=request.experiment_id,
                attempt_id=request.attempt_id,
                status=status,
                exit_code=runtime_result.exit_code,
                error_type=error_type,
                error_summary=runtime_result.stderr[-4000:] or "Docker worker returned no result",
                command_sha256=request_sha,
                commit_sha=request.commit_sha,
                config_sha256=request.config_sha256,
                data_view_sha256=request.data_view_sha256,
                environment_sha256=request.environment_sha256,
                artifacts=additions,
                wall_seconds=runtime_result.wall_seconds,
            )
        # The result is made durable and revalidated before container cleanup.
        # A controller crash after this point can replay the result without
        # duplicating artifact registration, while still retrying cleanup first.
        atomic_write_json(result_path, durable_result.model_dump(mode="json", by_alias=True))
        if _validated_replay(request, result_path) is None:
            raise ExecutionRuntimeError("durable Docker worker result verification failed")
        try:
            selected_runtime.cleanup(handle)
        except ExecutionRuntimeError as error:
            raise ExecutionRuntimeError(f"Docker worker cleanup failed closed: {error}") from error
        worker_result_path.unlink(missing_ok=True)
        return durable_result
    except (
        DockerLeaseError,
        ExecutionRuntimeError,
        OSError,
        SandboxError,
        subprocess.SubprocessError,
        ValidationError,
        ValueError,
        json.JSONDecodeError,
        WorkerLeaseError,
    ) as error:
        stderr_path.write_text(str(error), encoding="utf-8")
        artifacts = [artifact_ref(stderr_path, "stderr")]
        for path, kind in (
            (evidence_path, "sandbox_evidence"),
            (lease_path, "worker_lease"),
            (recovery_path, "worker_recovery"),
        ):
            if path.is_file():
                artifacts.append(artifact_ref(path, kind))
        return RunResult(
            run_id=request.run_id,
            experiment_id=request.experiment_id,
            attempt_id=request.attempt_id,
            status=AttemptStatus.CONTRACT,
            error_type="DockerRuntimeViolation",
            error_summary=str(error)[-4000:],
            command_sha256=request_sha,
            commit_sha=request.commit_sha,
            config_sha256=request.config_sha256,
            data_view_sha256=request.data_view_sha256,
            environment_sha256=request.environment_sha256,
            artifacts=artifacts,
            wall_seconds=time.monotonic() - started,
        )
    finally:
        attempt_lock.release()


def execute_request(
    request: RunRequest,
    attempt_dir: str | Path,
    *,
    python_executable: str = sys.executable,
    trusted_worktree_root: str | Path | None = None,
    sandbox_mode: SandboxMode | str = SandboxMode.FIXTURE,
    trusted_output_root: str | Path | None = None,
    environment_overrides: dict[str, str] | None = None,
    execution_runtime: ExecutionRuntime | None = None,
) -> RunResult:
    selected_mode = SandboxMode(sandbox_mode)
    runtime_kind = os.environ.get("REX_PRODUCTION_RUNTIME", "docker").strip().lower()
    if selected_mode == SandboxMode.PRODUCTION and runtime_kind != "native_macos":
        if environment_overrides:
            return _contract_result(
                request,
                error_type="DockerRuntimeViolation",
                summary="Docker workers do not accept arbitrary environment overrides",
                command=["python", "-m", "rex.execution.worker"],
                started=time.monotonic(),
            )
        return _docker_worker_request(
            request,
            Path(attempt_dir),
            trusted_worktree_root=trusted_worktree_root,
            trusted_output_root=trusted_output_root,
            runtime=execution_runtime,
        )
    if (
        selected_mode == SandboxMode.PRODUCTION
        and os.environ.get("REX_ALLOW_NATIVE_MACOS_ROLLBACK") != "1"
    ):
        return _contract_result(
            request,
            error_type="SandboxUnavailable",
            summary="native macOS production rollback requires explicit authorization",
            command=[python_executable, "-m", "rex.execution.worker"],
            started=time.monotonic(),
        )
    started = time.monotonic()
    directory = Path(attempt_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    request_root = directory / "input"
    result_root = directory / "worker"
    temp_root = result_root / "tmp"
    request_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    request_path = request_root / "request.json"
    result_path = result_root / "result.json"
    policy_path = directory / "sandbox.sb"
    evidence_path = directory / "sandbox_evidence.json"
    lease_path = directory / "worker_lease.json"
    recovery_path = directory / "worker_recovery.json"
    replay_path = directory / "worker_replay.json"
    launch_gate_path = result_root / "worker_launch_gate.json"
    stdout_path = directory / "stdout.log"
    stderr_path = directory / "stderr.log"
    command = [
        python_executable,
        "-m",
        "rex.execution.worker",
        "--request",
        str(request_path),
        "--result",
        str(result_path),
    ]
    command_sha = hashlib.sha256(canonical_json_bytes(command)).hexdigest()
    request_payload = request.model_dump(mode="json", by_alias=True)
    request_sha = hashlib.sha256(canonical_json_bytes(request_payload)).hexdigest()
    try:
        attempt_lock = AttemptLock.acquire(directory / "worker_lease.lock")
    except WorkerLeaseError as error:
        return _contract_result(
            request,
            error_type="WorkerLeaseConflict",
            summary=str(error),
            command=command,
            started=started,
        )
    if request_path.is_file():
        try:
            prior_request = RunRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
            prior_sha = hashlib.sha256(
                canonical_json_bytes(prior_request.model_dump(mode="json", by_alias=True))
            ).hexdigest()
        except (OSError, UnicodeError, ValidationError, json.JSONDecodeError) as error:
            attempt_lock.release()
            return _contract_result(
                request,
                error_type="WorkerLeaseConflict",
                summary=f"attempt request record is corrupt: {error}",
                command=command,
                started=started,
            )
        if prior_sha != request_sha:
            attempt_lock.release()
            return _contract_result(
                request,
                error_type="WorkerLeaseConflict",
                summary="attempt directory belongs to a different request hash",
                command=command,
                started=started,
            )
    atomic_write_json(request_path, request_payload)
    resources = ResourceTotals()
    timed_out = False
    memory_exceeded = False
    return_code: int | None = None
    invalid_artifact = False
    workspace_error: str | None = None
    sandbox_error: str | None = None
    launch_error: str | None = None
    plan: PreparedSandbox | None = None
    selected_mode = None
    execution_sha: str | None = None
    try:
        workspace = _validated_workspace(request, trusted_worktree_root)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        workspace = repo_root().resolve()
        workspace_error = str(error)
    if workspace_error is None:
        try:
            selected_mode = SandboxMode(sandbox_mode)
            plan = (
                _fixture_plan(workspace, command)
                if selected_mode == SandboxMode.FIXTURE
                else _production_plan(
                    request,
                    directory=directory,
                    workspace=workspace,
                    command=command,
                    request_path=request_path,
                    result_root=result_root,
                    temp_root=temp_root,
                    policy_path=policy_path,
                    trusted_output_root=trusted_output_root,
                    environment_overrides=environment_overrides,
                )
            )
            atomic_write_json(evidence_path, plan.evidence)
        except (OSError, ValueError, SandboxError) as error:
            sandbox_error = str(error)
    if workspace_error is None and sandbox_error is None:
        assert plan is not None
        assert selected_mode is not None
        try:
            requested_executable_sha = sha256_file(Path(python_executable).resolve(strict=True))
        except OSError as error:
            sandbox_error = f"requested worker executable is unreadable: {error}"
            requested_executable_sha = ""
        execution_sha = hashlib.sha256(
            canonical_json_bytes(
                {
                    "request_sha256": request_sha,
                    "sandbox_mode": selected_mode.value,
                    "workspace": str(workspace),
                    "planned_command": list(plan.command),
                    "requested_executable_sha256": requested_executable_sha,
                    "declared_environment_sha256": request.environment_sha256,
                    "environment_overrides_sha256": hashlib.sha256(
                        canonical_json_bytes(environment_overrides or {})
                    ).hexdigest(),
                    "sandbox_policy_sha256": plan.evidence.get("policy_sha256"),
                    "sandbox_profile_sha256": plan.evidence.get("profile_sha256"),
                    "lease_wrapper_sha256": hashlib.sha256(
                        _LEASE_WRAPPER.encode("utf-8")
                    ).hexdigest(),
                }
            )
        ).hexdigest()
    if workspace_error is None and sandbox_error is None:
        assert execution_sha is not None
        try:
            recover_orphan_worker(
                lease_path,
                recovery_path,
                request_sha256=request_sha,
                execution_sha256=execution_sha,
            )
        except WorkerLeaseError as error:
            recovery_ref = (
                artifact_ref(recovery_path, "worker_recovery") if recovery_path.is_file() else None
            )
            attempt_lock.release()
            return _contract_result(
                request,
                error_type="WorkerLeaseConflict",
                summary=str(error),
                command=command,
                started=started,
                artifacts=[] if recovery_ref is None else [recovery_ref],
            )
        replay = _validated_replay(request, result_path)
        if replay is not None:
            atomic_write_json(
                replay_path,
                {
                    "schema_version": "1.0",
                    "outcome": "complete-result-replayed",
                    "request_sha256": request_sha,
                    "result_sha256": sha256_file(result_path),
                    "replayed_at_epoch_ms": int(time.time() * 1000),
                },
            )
            additions: list[ArtifactRef] = [artifact_ref(replay_path, "worker_replay")]
            for candidate, kind in (
                (stdout_path, "stdout"),
                (stderr_path, "stderr"),
                (evidence_path, "sandbox_evidence"),
                (policy_path, "sandbox_profile"),
                (lease_path, "worker_lease"),
                (recovery_path, "worker_recovery"),
            ):
                if candidate.is_file():
                    additions.append(artifact_ref(candidate, kind))
            replay = _append_artifacts(replay, additions)
            attempt_lock.release()
            return replay
        if result_path.exists():
            atomic_write_json(
                replay_path,
                {
                    "schema_version": "1.0",
                    "outcome": "partial-or-invalid-result-rerun",
                    "request_sha256": request_sha,
                    "observed_result_sha256": sha256_file(result_path),
                    "recorded_at_epoch_ms": int(time.time() * 1000),
                },
            )
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        if workspace_error is not None:
            stderr.write(workspace_error.encode("utf-8", errors="replace"))
            stderr.flush()
        elif sandbox_error is not None:
            stderr.write(sandbox_error.encode("utf-8", errors="replace"))
            stderr.flush()
        else:
            assert plan is not None
            assert execution_sha is not None
            active_lease: dict[str, object] | None = None
            try:
                launch_gate_path.unlink(missing_ok=True)
                wrapped_command, identity_token = _lease_wrapper_command(
                    plan.command, launch_gate_path
                )
                process = subprocess.Popen(
                    wrapped_command,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                    env=plan.environment,
                    cwd=workspace,
                    preexec_fn=plan.preexec_fn,
                )
                try:
                    active_lease = begin_worker_lease(
                        lease_path,
                        pid=process.pid,
                        request_sha256=request_sha,
                        execution_sha256=execution_sha,
                        planned_command_sha256=command_sha256(plan.command),
                        identity_token=identity_token,
                    )
                    atomic_write_json(
                        launch_gate_path,
                        {
                            "schema_version": "1.0",
                            "request_sha256": request_sha,
                            "execution_sha256": execution_sha,
                            "identity_token_sha256": hashlib.sha256(
                                identity_token.encode("utf-8")
                            ).hexdigest(),
                            "released_at_epoch_ms": int(time.time() * 1000),
                        },
                    )
                except (OSError, WorkerLeaseError) as error:
                    _kill_process_group(process)
                    if active_lease is not None:
                        close_worker_lease(
                            lease_path,
                            active_lease,
                            reason="launch-handshake-failed",
                            return_code=process.poll(),
                        )
                    raise WorkerLeaseError(
                        f"worker launched without a durable identity lease: {error}"
                    ) from error
                while process.poll() is None:
                    resources.sample(process.pid)
                    elapsed = time.monotonic() - started
                    deadline_reached = int(time.time() * 1000) >= request.deadline_epoch_ms
                    memory_exceeded = bool(
                        request.max_memory_mb
                        and resources.peak_rss_bytes > request.max_memory_mb * 1024 * 1024
                    )
                    if elapsed >= request.timeout_seconds or deadline_reached or memory_exceeded:
                        timed_out = True
                        if memory_exceeded:
                            timed_out = False
                        _kill_process_group(process)
                        break
                    time.sleep(0.05)
                return_code = process.poll()
                resources.sample(process.pid)
                close_worker_lease(
                    lease_path,
                    active_lease,
                    reason=(
                        "memory-limit"
                        if memory_exceeded
                        else ("timeout" if timed_out else "process-exit")
                    ),
                    return_code=return_code,
                )
                launch_gate_path.unlink(missing_ok=True)
            except (OSError, subprocess.SubprocessError, WorkerLeaseError) as error:
                launch_error = str(error)
                stderr.write(launch_error.encode("utf-8", errors="replace"))
                stderr.flush()

    wall_seconds = time.monotonic() - started
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    stdout_ref = artifact_ref(stdout_path, "stdout")
    stderr_ref = artifact_ref(stderr_path, "stderr")
    sandbox_ref = (
        artifact_ref(evidence_path, "sandbox_evidence") if evidence_path.is_file() else None
    )
    profile_ref = artifact_ref(policy_path, "sandbox_profile") if policy_path.is_file() else None
    lease_ref = artifact_ref(lease_path, "worker_lease") if lease_path.is_file() else None
    recovery_ref = (
        artifact_ref(recovery_path, "worker_recovery") if recovery_path.is_file() else None
    )
    replay_ref = artifact_ref(replay_path, "worker_replay") if replay_path.is_file() else None

    if (
        result_path.is_file()
        and not timed_out
        and workspace_error is None
        and sandbox_error is None
        and launch_error is None
    ):
        try:
            result = RunResult.model_validate_json(result_path.read_text(encoding="utf-8"))
            if (
                result.command_sha256
                != hashlib.sha256(canonical_json_bytes(request.model_dump(mode="json"))).hexdigest()
            ):
                raise ValueError("worker result command hash mismatch")
            artifacts = [*result.artifacts, stdout_ref, stderr_ref]
            if sandbox_ref is not None:
                artifacts.append(sandbox_ref)
            if profile_ref is not None:
                artifacts.append(profile_ref)
            if lease_ref is not None:
                artifacts.append(lease_ref)
            if recovery_ref is not None:
                artifacts.append(recovery_ref)
            if replay_ref is not None:
                artifacts.append(replay_ref)
            for ref in result.artifacts:
                artifact_path = Path(ref.path)
                if not artifact_path.is_file() or sha256_file(artifact_path) != ref.sha256:
                    raise ValueError(f"worker artifact missing or corrupt: {ref.artifact_id}")
            result = result.model_copy(
                update={
                    "artifacts": artifacts,
                    "wall_seconds": wall_seconds,
                    "cpu_user_seconds": resources.cpu_user_seconds,
                    "cpu_system_seconds": resources.cpu_system_seconds,
                    "peak_rss_bytes": resources.peak_rss_bytes,
                }
            )
            atomic_write_json(result_path, result.model_dump(mode="json", by_alias=True))
            attempt_lock.release()
            return result
        except (ValidationError, ValueError, json.JSONDecodeError) as error:
            stderr_text += f"\ninvalid worker result: {error}"
            invalid_artifact = True
    elif (
        return_code == 0
        and workspace_error is None
        and sandbox_error is None
        and launch_error is None
        and not timed_out
    ):
        stderr_text += "\nworker exited successfully without a result artifact"
        invalid_artifact = True

    if workspace_error is not None:
        stderr_text = workspace_error
    elif sandbox_error is not None:
        stderr_text = sandbox_error
    elif launch_error is not None:
        stderr_text = launch_error

    failure_artifacts = [stdout_ref, stderr_ref]
    if sandbox_ref is not None:
        failure_artifacts.append(sandbox_ref)
    if profile_ref is not None:
        failure_artifacts.append(profile_ref)
    if lease_ref is not None:
        failure_artifacts.append(lease_ref)
    if recovery_ref is not None:
        failure_artifacts.append(recovery_ref)
    if replay_ref is not None:
        failure_artifacts.append(replay_ref)

    failure = RunResult(
        run_id=request.run_id,
        experiment_id=request.experiment_id,
        attempt_id=request.attempt_id,
        status=(
            AttemptStatus.CONTRACT
            if workspace_error is not None or sandbox_error is not None or launch_error is not None
            else _typed_failure(
                stderr_text,
                timed_out,
                memory_exceeded,
                return_code=return_code,
                invalid_artifact=invalid_artifact,
            )
        ),
        exit_code=return_code,
        signal=-return_code if return_code is not None and return_code < 0 else None,
        error_type=(
            "WorkspaceViolation"
            if workspace_error is not None
            else (
                "SandboxUnavailable"
                if sandbox_error is not None
                else (
                    "SandboxLaunchFailure"
                    if launch_error is not None
                    else (
                        "MemoryLimit"
                        if memory_exceeded
                        else (
                            "Timeout"
                            if timed_out
                            else (
                                "InvalidArtifact"
                                if invalid_artifact
                                else (
                                    "Interrupted"
                                    if return_code is not None and return_code < 0
                                    else "WorkerFailure"
                                )
                            )
                        )
                    )
                )
            )
        ),
        error_summary=stderr_text or "worker exited without a valid result",
        command_sha256=command_sha,
        commit_sha=request.commit_sha,
        config_sha256=request.config_sha256,
        data_view_sha256=request.data_view_sha256,
        environment_sha256=request.environment_sha256,
        artifacts=failure_artifacts,
        wall_seconds=wall_seconds,
        cpu_user_seconds=resources.cpu_user_seconds,
        cpu_system_seconds=resources.cpu_system_seconds,
        peak_rss_bytes=resources.peak_rss_bytes,
    )
    attempt_lock.release()
    return failure
