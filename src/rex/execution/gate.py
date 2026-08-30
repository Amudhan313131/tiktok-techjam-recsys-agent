"""Sandboxed static/fixture gates for agent-authored production patches."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rex.data.manifest import canonical_json_bytes
from rex.execution.artifacts import atomic_write_json
from rex.execution.limits import limits_for_request
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
    SandboxError,
    SandboxMode,
    SandboxPolicy,
    fixture_environment,
    production_backend,
    sanitized_environment,
)
from rex.execution.telemetry import ResourceTotals


class GateExecutionError(RuntimeError):
    """A gate could not be launched under its required trust boundary."""


@dataclass(frozen=True)
class GateResult:
    name: str
    command: tuple[str, ...]
    command_sha256: str
    return_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    wall_seconds: float
    peak_rss_bytes: int
    evidence_path: Path
    profile_path: Path | None

    @property
    def successful(self) -> bool:
        return self.return_code == 0 and not self.timed_out


def _persist_docker_gate_result(
    path: Path,
    result: ExecutionResult,
    *,
    request_sha256: str,
    execution_sha256: str,
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": "rex.docker-gate-result.v1",
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


def _read_docker_gate_result(
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
            payload.get("schema_version") != "rex.docker-gate-result.v1"
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


def _inside(root: Path, candidate: Path, *, label: str) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise GateExecutionError(f"{label} is outside its trusted root: {candidate}") from error
    return candidate


def _verified_worktree(workspace: Path, trusted_root: Path) -> Path:
    workspace = _inside(trusted_root, workspace, label="gate workspace")
    if not workspace.is_dir():
        raise GateExecutionError(f"gate workspace does not exist: {workspace}")
    observed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=workspace,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if observed.returncode or Path(observed.stdout.strip()).resolve() != workspace:
        raise GateExecutionError("gate workspace is not a Git worktree root")
    return workspace


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=2)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass


def execute_gate(
    *,
    name: str,
    command: Sequence[str],
    workspace: str | Path,
    artifact_dir: str | Path,
    timeout_seconds: int,
    sandbox_mode: SandboxMode | str,
    trusted_worktree_root: str | Path | None = None,
    trusted_output_root: str | Path | None = None,
    max_memory_mb: int | None = None,
    execution_runtime: ExecutionRuntime | None = None,
) -> GateResult:
    """Execute a candidate-code gate without granting host or secret access.

    Production callers must provide both trusted roots. A dirty worktree is valid
    here because gates intentionally run after patch application and before the
    candidate commit.
    """

    if not command:
        raise GateExecutionError("gate command is empty")
    selected_mode = SandboxMode(sandbox_mode)
    workspace_path = Path(workspace).resolve()
    evidence_root = Path(artifact_dir).resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)
    temp_root = evidence_root / f"{name}-sandbox-temp"
    temp_root.mkdir(parents=True, exist_ok=True)
    profile_path = evidence_root / f"{name}-sandbox.sb"
    evidence_path = evidence_root / f"{name}-sandbox.json"
    limits = limits_for_request(timeout_seconds, max_memory_mb)

    runtime_kind = os.environ.get("REX_PRODUCTION_RUNTIME", "docker").strip().lower()
    if selected_mode == SandboxMode.PRODUCTION and runtime_kind != "native_macos":
        if trusted_worktree_root is None or trusted_output_root is None:
            raise GateExecutionError("production gate requires trusted worktree and output roots")
        workspace_path = _verified_worktree(workspace_path, Path(trusted_worktree_root))
        _inside(Path(trusted_output_root), evidence_root, label="gate artifact directory")
        executable = Path(str(command[0])).name
        if executable.startswith("python"):
            executable = "python"
        docker_command = (executable, *tuple(command)[1:])
        environment = {
            "HOME": "/output",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": f"{workspace_path}/src",
            "PYTHONPYCACHEPREFIX": "/output/pycache",
            "PYTEST_ADDOPTS": "-p no:cacheprovider",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "TMPDIR": "/tmp",
        }
        request_sha = hashlib.sha256(
            canonical_json_bytes(
                {
                    "name": name,
                    "command": list(docker_command),
                    "workspace": str(workspace_path),
                    "timeout_seconds": timeout_seconds,
                    "max_memory_mb": max_memory_mb,
                }
            )
        ).hexdigest()
        execution_sha = hashlib.sha256(
            canonical_json_bytes(
                {
                    "request_sha256": request_sha,
                    "environment": environment,
                    "docker_security_policy_sha256": docker_security_policy_sha256(),
                }
            )
        ).hexdigest()
        selected_runtime = production_runtime() if execution_runtime is None else execution_runtime
        if os.geteuid() == 0:
            raise GateExecutionError("production Docker controller must not run as root")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-.") or "gate"
        lease_path = evidence_root / f"{safe_name}-docker-lease.json"
        durable_result_path = evidence_root / f"{safe_name}-docker-result.json"
        recovery_path = evidence_root / f"{safe_name}-docker-recovery.json"
        specification = ExecutionSpec(
            command=docker_command,
            working_directory=str(workspace_path),
            mounts=(
                RuntimeMount(workspace_path, str(workspace_path), True),
                RuntimeMount(temp_root, "/output", False),
            ),
            environment=environment,
            timeout_seconds=float(timeout_seconds),
            memory_bytes=max(64, max_memory_mb or 2048) * 1024 * 1024,
            nano_cpus=1_000_000_000,
            pids_limit=128,
            run_id="gate",
            experiment_id=safe_name[:128],
            attempt_id=f"gate:{request_sha[:24]}",
            request_sha256=request_sha,
            execution_sha256=execution_sha,
            lease_path=lease_path,
            user=f"{os.geteuid()}:{os.getegid()}",
        )
        started = time.monotonic()
        handle = None
        recovered_handle = None
        result = None
        recovery_evidence: dict[str, object] | None = None
        try:
            if lease_path.is_file():
                prior = read_docker_lease(lease_path)
                if prior.request_sha256 != request_sha or prior.execution_sha256 != execution_sha:
                    raise GateExecutionError("Docker gate lease belongs to a different request")
                recovery = selected_runtime.recover(prior)
                recovered = read_docker_lease(lease_path)
                recovered_handle = runtime_handle_from_docker_lease(
                    recovered, timeout_seconds=float(timeout_seconds)
                )
                status = selected_runtime.inspect(recovered_handle)
                container_absent = status.state == RuntimeLifecycleState.REMOVED
                durable_result = _read_docker_gate_result(
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
                            lease_path, related_evidence_paths=(recovery_path,)
                        )
                        atomic_write_json(recovery_path, recovery_evidence)
                        recovered_handle = None
                else:
                    result = durable_result or selected_runtime.collect(recovered_handle)
                    _persist_docker_gate_result(
                        durable_result_path,
                        result,
                        request_sha256=request_sha,
                        execution_sha256=execution_sha,
                    )
                    if (
                        _read_docker_gate_result(
                            durable_result_path,
                            request_sha256=request_sha,
                            execution_sha256=execution_sha,
                        )
                        is None
                    ):
                        raise GateExecutionError("Docker gate durable result verification failed")
                    selected_runtime.cleanup(recovered_handle)
                    if result.outcome in {
                        ExecutionOutcome.TIMEOUT,
                        ExecutionOutcome.OOM,
                        ExecutionOutcome.INTERRUPTED,
                    }:
                        recovery_evidence["lease_archive"] = archive_closed_docker_lease(
                            lease_path, related_evidence_paths=(recovery_path,)
                        )
                        atomic_write_json(recovery_path, recovery_evidence)
                        durable_result_path.unlink(missing_ok=True)
                        recovered_handle = None
                        result = None
            if result is None:
                handle = selected_runtime.launch(specification)
                result = selected_runtime.collect(handle)
                _persist_docker_gate_result(
                    durable_result_path,
                    result,
                    request_sha256=request_sha,
                    execution_sha256=execution_sha,
                )
                if (
                    _read_docker_gate_result(
                        durable_result_path,
                        request_sha256=request_sha,
                        execution_sha256=execution_sha,
                    )
                    is None
                ):
                    raise GateExecutionError("Docker gate durable result verification failed")
            else:
                handle = recovered_handle
        except (DockerLeaseError, ExecutionRuntimeError) as error:
            raise GateExecutionError(f"Docker gate failed closed: {error}") from error
        finally:
            if handle is not None and handle is not recovered_handle:
                try:
                    selected_runtime.cleanup(handle)
                except ExecutionRuntimeError as error:
                    raise GateExecutionError(
                        f"Docker gate cleanup failed closed: {error}"
                    ) from error
        stdout_path = evidence_root / f"{name}.stdout.log"
        stderr_path = evidence_root / f"{name}.stderr.log"
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        evidence = {
            "schema_version": "rex.docker-gate-evidence.v1",
            "mode": SandboxMode.PRODUCTION.value,
            "backend": "docker",
            "sandboxed": True,
            "gate": name,
            "requested_command": list(command),
            "requested_command_sha256": hashlib.sha256(canonical_json_bytes(command)).hexdigest(),
            "container_id": handle.container_id,
            "worker_image_digest": handle.worker_image_digest,
            "docker_security_policy_sha256": docker_security_policy_sha256(),
            "outcome": result.outcome.value,
            "lease_path": str(specification.lease_path),
            "durable_result_path": str(durable_result_path),
            "recovery": recovery_evidence,
        }
        atomic_write_json(evidence_path, evidence)
        return GateResult(
            name=name,
            command=tuple(command),
            command_sha256=hashlib.sha256(canonical_json_bytes(command)).hexdigest(),
            return_code=result.exit_code,
            timed_out=result.outcome == ExecutionOutcome.TIMEOUT,
            stdout=result.stdout,
            stderr=result.stderr,
            wall_seconds=time.monotonic() - started,
            peak_rss_bytes=0,
            evidence_path=evidence_path,
            profile_path=None,
        )

    if (
        selected_mode == SandboxMode.PRODUCTION
        and os.environ.get("REX_ALLOW_NATIVE_MACOS_ROLLBACK") != "1"
    ):
        raise GateExecutionError("native macOS rollback requires explicit authorization")

    if selected_mode == SandboxMode.PRODUCTION:
        if trusted_worktree_root is None or trusted_output_root is None:
            raise GateExecutionError("production gate requires trusted worktree and output roots")
        workspace_path = _verified_worktree(workspace_path, Path(trusted_worktree_root))
        _inside(Path(trusted_output_root), evidence_root, label="gate artifact directory")
        environment = sanitized_environment(workspace=workspace_path, temp_dir=temp_root)
        environment.update(
            {
                "PYTHONPYCACHEPREFIX": str(temp_root / "pycache"),
                "PYTEST_ADDOPTS": "-p no:cacheprovider",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            }
        )
        executable = shutil.which(command[0], path=environment.get("PATH"))
        if executable is None:
            raise GateExecutionError(f"gate executable is unavailable: {command[0]}")
        # Preserve a virtual-environment launcher path. Resolving the symlink to
        # the base interpreter silently drops that environment's site-packages,
        # so production gates could not import the same pinned dependencies as
        # model workers.
        resolved_executable = Path(executable).absolute()
        resolved_command = (str(resolved_executable), *command[1:])
        policy = SandboxPolicy(
            workspace=workspace_path,
            read_paths=(workspace_path, resolved_executable),
            write_paths=(temp_root,),
            network_allowed=False,
            resource_limits=limits,
        )
        try:
            plan = production_backend().prepare(policy, resolved_command, environment, profile_path)
        except SandboxError as error:
            raise GateExecutionError(str(error)) from error
    else:
        environment = fixture_environment(workspace_path)
        environment.update(
            {
                "PYTHONPYCACHEPREFIX": str(temp_root / "pycache"),
                "PYTEST_ADDOPTS": "-p no:cacheprovider",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            }
        )
        plan_command = tuple(command)
        plan = None
        evidence = {
            "schema_version": "1.0",
            "mode": SandboxMode.FIXTURE.value,
            "backend": "trusted-fixture-unsandboxed",
            "sandboxed": False,
            "reason": "unsandboxed gate execution is restricted to explicit fixture/test mode",
            "environment_keys": sorted(environment),
            "resource_limits": None,
        }

    if selected_mode == SandboxMode.PRODUCTION:
        assert plan is not None
        plan_command = plan.command
        environment = plan.environment
        preexec_fn = plan.preexec_fn
        evidence = plan.evidence
    else:
        preexec_fn = None
    evidence = {
        **evidence,
        "gate": name,
        "requested_command": list(command),
        "requested_command_sha256": hashlib.sha256(canonical_json_bytes(command)).hexdigest(),
    }
    atomic_write_json(evidence_path, evidence)

    started = time.monotonic()
    resources = ResourceTotals()
    timed_out = False
    stdout_path = evidence_root / f"{name}.stdout.log"
    stderr_path = evidence_root / f"{name}.stderr.log"
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        try:
            process = subprocess.Popen(
                plan_command,
                cwd=workspace_path,
                text=True,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
                env=environment,
                preexec_fn=preexec_fn,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise GateExecutionError(f"gate launch failed: {error}") from error
        while process.poll() is None:
            resources.sample(process.pid)
            memory_exceeded = bool(
                max_memory_mb and resources.peak_rss_bytes > max_memory_mb * 1024 * 1024
            )
            if time.monotonic() - started >= timeout_seconds or memory_exceeded:
                timed_out = True
                _kill_process_group(process)
                break
            time.sleep(0.05)
        process.wait(timeout=2)
    resources.sample(process.pid)
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    return GateResult(
        name=name,
        command=tuple(command),
        command_sha256=hashlib.sha256(canonical_json_bytes(command)).hexdigest(),
        return_code=process.returncode,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
        wall_seconds=time.monotonic() - started,
        peak_rss_bytes=resources.peak_rss_bytes,
        evidence_path=evidence_path,
        profile_path=profile_path if profile_path.is_file() else None,
    )
