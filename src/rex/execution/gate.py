"""Sandboxed static/fixture gates for agent-authored production patches."""

from __future__ import annotations

import hashlib
import os
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
            plan = production_backend().prepare(
                policy, resolved_command, environment, profile_path
            )
        except SandboxError as error:
            raise GateExecutionError(str(error)) from error
    else:
        environment = fixture_environment(workspace_path)
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
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
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
