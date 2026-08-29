"""Process-group isolated worker runner with timeout, telemetry, and typed fallback results."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from pydantic import ValidationError

from rex.contracts import AttemptStatus, RunRequest, RunResult
from rex.data.manifest import canonical_json_bytes, repo_root, sha256_file
from rex.execution.artifacts import artifact_ref, atomic_write_json
from rex.execution.telemetry import ResourceTotals


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


def execute_request(
    request: RunRequest,
    attempt_dir: str | Path,
    *,
    python_executable: str = sys.executable,
    trusted_worktree_root: str | Path | None = None,
) -> RunResult:
    directory = Path(attempt_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    request_path = directory / "request.json"
    result_path = directory / "result.json"
    stdout_path = directory / "stdout.log"
    stderr_path = directory / "stderr.log"
    atomic_write_json(request_path, request.model_dump(mode="json", by_alias=True))
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
    started = time.monotonic()
    resources = ResourceTotals()
    timed_out = False
    memory_exceeded = False
    return_code: int | None = None
    invalid_artifact = False
    workspace_error: str | None = None
    try:
        workspace = _validated_workspace(request, trusted_worktree_root)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        workspace = repo_root().resolve()
        workspace_error = str(error)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        if workspace_error is not None:
            stderr.write(workspace_error.encode("utf-8", errors="replace"))
            stderr.flush()
        else:
            environment = os.environ.copy()
            source_root = str(workspace / "src")
            environment["PYTHONPATH"] = (
                source_root
                if not environment.get("PYTHONPATH")
                else source_root + os.pathsep + environment["PYTHONPATH"]
            )
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            process = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                env=environment,
                cwd=workspace,
            )
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

    wall_seconds = time.monotonic() - started
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    stdout_ref = artifact_ref(stdout_path, "stdout")
    stderr_ref = artifact_ref(stderr_path, "stderr")

    if result_path.is_file() and not timed_out and workspace_error is None:
        try:
            result = RunResult.model_validate_json(result_path.read_text(encoding="utf-8"))
            if result.command_sha256 != hashlib.sha256(
                canonical_json_bytes(request.model_dump(mode="json"))
            ).hexdigest():
                raise ValueError("worker result command hash mismatch")
            artifacts = [*result.artifacts, stdout_ref, stderr_ref]
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
            return result
        except (ValidationError, ValueError, json.JSONDecodeError) as error:
            stderr_text += f"\ninvalid worker result: {error}"
            invalid_artifact = True
    elif return_code == 0 and workspace_error is None and not timed_out:
        stderr_text += "\nworker exited successfully without a result artifact"
        invalid_artifact = True

    if workspace_error is not None:
        stderr_text = workspace_error

    return RunResult(
        run_id=request.run_id,
        experiment_id=request.experiment_id,
        attempt_id=request.attempt_id,
        status=(
            AttemptStatus.CONTRACT
            if workspace_error is not None
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
                "MemoryLimit"
                if memory_exceeded
                else (
                    "Timeout"
                    if timed_out
                    else (
                        "InvalidArtifact"
                        if invalid_artifact
                        else ("Interrupted" if return_code is not None and return_code < 0 else "WorkerFailure")
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
        artifacts=[stdout_ref, stderr_ref],
        wall_seconds=wall_seconds,
        cpu_user_seconds=resources.cpu_user_seconds,
        cpu_system_seconds=resources.cpu_system_seconds,
        peak_rss_bytes=resources.peak_rss_bytes,
    )
