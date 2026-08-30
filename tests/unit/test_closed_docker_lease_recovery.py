from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from rex.contracts import AttemptStatus, RunRequest, RunResult
from rex.data.manifest import canonical_json_bytes, sha256_file
from rex.evaluation import submission as submission_module
from rex.evaluation.submission import SubmissionError, validate_submission
from rex.execution.docker_lease import (
    close_docker_lease,
    create_docker_lease,
    mark_docker_lease_started,
    persist_docker_lease,
    read_docker_lease,
)
from rex.execution.gate import GateExecutionError, execute_gate
from rex.execution.runner import _docker_worker_request
from rex.execution.runtime import (
    DoctorResult,
    ExecutionLease,
    ExecutionOutcome,
    ExecutionResult,
    ExecutionRuntimeError,
    ExecutionSpec,
    RecoveryResult,
    RuntimeHandle,
    RuntimeKind,
    RuntimeLifecycleState,
    RuntimeStatus,
)
from rex.execution.sandbox import SandboxMode


IMAGE = "sha256:" + "a" * 64
DAEMON = "sha256:" + "b" * 64


class ClosedLeaseRuntime:
    """Small exact-lifecycle fake used to simulate controller crash windows."""

    def __init__(self, *, fail_cleanup_once: bool = False) -> None:
        self.launches = 0
        self.collections = 0
        self.cleanups = 0
        self.removed: set[str] = set()
        self.fail_cleanup_once = fail_cleanup_once
        self.specifications: dict[str, ExecutionSpec] = {}

    def doctor(self) -> DoctorResult:  # pragma: no cover - not used by callers
        return DoctorResult(RuntimeKind.DOCKER, True, True)

    def launch(self, specification: ExecutionSpec) -> RuntimeHandle:
        self.launches += 1
        container_id = f"{self.launches:064x}"
        handle = RuntimeHandle(
            runtime_kind=RuntimeKind.DOCKER,
            container_id=container_id,
            container_name=f"rex-fake-{self.launches}",
            worker_image_digest=IMAGE,
            worker_image_id=IMAGE,
            daemon_identity=DAEMON,
            run_id=specification.run_id,
            experiment_id=specification.experiment_id,
            attempt_id=specification.attempt_id,
            request_sha256=specification.request_sha256,
            execution_sha256=specification.execution_sha256,
            lease_path=specification.lease_path,
            timeout_seconds=specification.timeout_seconds,
            started_at_epoch_ms=int(time.time() * 1000),
        )
        lease = create_docker_lease(specification, handle)
        persist_docker_lease(specification.lease_path, lease)
        mark_docker_lease_started(specification.lease_path, lease)
        self.specifications[container_id] = specification
        return handle

    def inspect(self, handle: RuntimeHandle) -> RuntimeStatus:
        state = (
            RuntimeLifecycleState.REMOVED
            if handle.container_id in self.removed
            else RuntimeLifecycleState.EXITED
        )
        return RuntimeStatus(state=state, exit_code=0)

    def terminate(self, handle: RuntimeHandle) -> None:  # pragma: no cover - not used
        return None

    def collect(self, handle: RuntimeHandle) -> ExecutionResult:
        self.collections += 1
        specification = self.specifications[handle.container_id]
        if specification.command[:3] == ("python", "-m", "rex.execution.worker"):
            request_mount = next(
                mount for mount in specification.mounts if mount.target == "/request.json"
            )
            output_mount = next(
                mount for mount in specification.mounts if mount.target == "/output"
            )
            worker_request = RunRequest.model_validate_json(
                request_mount.source.read_text(encoding="utf-8")
            )
            worker_sha = hashlib.sha256(
                canonical_json_bytes(worker_request.model_dump(mode="json", by_alias=True))
            ).hexdigest()
            worker_result = RunResult(
                run_id=worker_request.run_id,
                experiment_id=worker_request.experiment_id,
                attempt_id=worker_request.attempt_id,
                status=AttemptStatus.SUCCESS,
                exit_code=0,
                command_sha256=worker_sha,
                commit_sha=worker_request.commit_sha,
                config_sha256=worker_request.config_sha256,
                data_view_sha256=worker_request.data_view_sha256,
                environment_sha256=worker_request.environment_sha256,
                artifacts=[],
                wall_seconds=0.01,
            )
            (output_mount.source / ".rex-worker-result.json").write_text(
                worker_result.model_dump_json(by_alias=True), encoding="utf-8"
            )
        lease = read_docker_lease(handle.lease_path)
        if lease.state in {"created", "active"}:
            close_docker_lease(handle.lease_path, lease, reason="success", exit_code=0)
        return ExecutionResult(
            outcome=ExecutionOutcome.SUCCESS,
            exit_code=0,
            stdout="ok\n",
            stderr="",
            wall_seconds=0.01,
        )

    def recover(self, lease: ExecutionLease) -> RecoveryResult:
        assert lease.state in {"closed", "recovered"}
        return RecoveryResult("already-closed", lease.container_id, False)

    def cleanup(self, handle: RuntimeHandle) -> None:
        self.cleanups += 1
        if self.fail_cleanup_once:
            self.fail_cleanup_once = False
            raise ExecutionRuntimeError("controlled cleanup failure")
        self.removed.add(handle.container_id)


def _git_workspace(path: Path) -> tuple[Path, str]:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "rex@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "REX Test"], cwd=path, check=True)
    (path / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=path, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return path, commit


def _runner_request(workspace: Path, commit: str, runs: Path) -> RunRequest:
    config = workspace / "config.json"
    features = workspace / "features.npz"
    targets = workspace / "targets.npz"
    config.write_text("{}\n", encoding="utf-8")
    features.write_bytes(b"features")
    targets.write_bytes(b"targets")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "inputs"], cwd=workspace, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workspace, check=True, text=True, capture_output=True
    ).stdout.strip()
    return RunRequest(
        run_id="run",
        experiment_id="experiment",
        attempt_id="attempt-1",
        commit_sha=commit,
        plugin="rex.models.experimental.fixture:FixturePlugin",
        config_path=str(config),
        config_sha256=sha256_file(config),
        seed=1,
        rung="cheap",
        split="train",
        feature_view_path=str(features),
        target_view_path=str(targets),
        output_dir=str(runs / "model-output"),
        workspace_path=str(workspace),
        deadline_epoch_ms=int((time.time() + 30) * 1000),
        timeout_seconds=20,
        data_view_sha256=sha256_file(features),
        environment_sha256="0" * 64,
    )


def test_runner_archives_closed_missing_result_and_relaunches(tmp_path: Path) -> None:
    workspace, commit = _git_workspace(tmp_path / "worktrees" / "candidate")
    runs = tmp_path / "runs"
    runs.mkdir()
    request = _runner_request(workspace, commit, runs)
    attempt = runs / "attempt"
    runtime = ClosedLeaseRuntime()

    first = _docker_worker_request(
        request,
        attempt,
        trusted_worktree_root=tmp_path / "worktrees",
        trusted_output_root=runs,
        runtime=runtime,
    )
    assert first.status == AttemptStatus.SUCCESS
    (attempt / "worker" / "result.json").unlink()

    second = _docker_worker_request(
        request,
        attempt,
        trusted_worktree_root=tmp_path / "worktrees",
        trusted_output_root=runs,
        runtime=runtime,
    )

    assert second.status == AttemptStatus.SUCCESS
    assert runtime.launches == 2
    assert list((attempt / "lease-archive").glob("worker_lease-*.json"))
    recovery = json.loads((attempt / "worker_recovery.json").read_text(encoding="utf-8"))
    assert recovery["outcome"] == "closed-lease-container-absent"


def test_runner_cleanup_failure_fails_closed_then_replays_once(tmp_path: Path) -> None:
    workspace, commit = _git_workspace(tmp_path / "worktrees" / "candidate")
    runs = tmp_path / "runs"
    runs.mkdir()
    request = _runner_request(workspace, commit, runs)
    attempt = runs / "attempt"
    runtime = ClosedLeaseRuntime(fail_cleanup_once=True)

    failed = _docker_worker_request(
        request,
        attempt,
        trusted_worktree_root=tmp_path / "worktrees",
        trusted_output_root=runs,
        runtime=runtime,
    )
    assert failed.status == AttemptStatus.CONTRACT
    assert "cleanup failed closed" in (failed.error_summary or "")
    assert (attempt / "worker" / "result.json").is_file()

    replayed = _docker_worker_request(
        request,
        attempt,
        trusted_worktree_root=tmp_path / "worktrees",
        trusted_output_root=runs,
        runtime=runtime,
    )

    assert replayed.status == AttemptStatus.SUCCESS
    assert runtime.launches == 1
    assert runtime.cleanups == 2


def test_gate_cleanup_failure_is_closed_and_recoverable(tmp_path: Path) -> None:
    workspace, _commit = _git_workspace(tmp_path / "worktrees" / "candidate")
    artifacts = tmp_path / "runs" / "gate"
    runtime = ClosedLeaseRuntime(fail_cleanup_once=True)
    arguments = dict(
        name="syntax",
        command=("python", "-c", "print('ok')"),
        workspace=workspace,
        artifact_dir=artifacts,
        timeout_seconds=10,
        sandbox_mode=SandboxMode.PRODUCTION,
        trusted_worktree_root=tmp_path / "worktrees",
        trusted_output_root=tmp_path / "runs",
        execution_runtime=runtime,
    )

    with pytest.raises(GateExecutionError, match="cleanup failed closed"):
        execute_gate(**arguments)
    assert (artifacts / "syntax-docker-result.json").is_file()

    result = execute_gate(**arguments)

    assert result.successful
    assert runtime.launches == 1
    assert runtime.cleanups == 2


def test_submission_archives_closed_missing_result_and_relaunches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    starter = tmp_path / "starter"
    starter.mkdir()
    submit_py = starter / "submit.py"
    submit_py.write_text("raise SystemExit(0)\n", encoding="utf-8")
    monkeypatch.setattr(
        submission_module,
        "verify_starter_manifest",
        lambda: SimpleNamespace(
            root=starter,
            manifest_sha256="c" * 64,
            hashes={"submit.py": sha256_file(submit_py)},
        ),
    )
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("REX_RUNS_ROOT", str(runs))
    data = tmp_path / "data"
    data.mkdir()
    csv_path = tmp_path / "submission.csv"
    csv_path.write_text("row_id,user_id,video_id,score\n", encoding="utf-8")
    runtime = ClosedLeaseRuntime()

    first = validate_submission(
        csv_path,
        data_dir=data,
        split="test",
        sandbox_mode=SandboxMode.PRODUCTION,
        execution_runtime=runtime,
    )
    Path(str(first.sandbox_evidence["durable_result_path"])).unlink()

    second = validate_submission(
        csv_path,
        data_dir=data,
        split="test",
        sandbox_mode=SandboxMode.PRODUCTION,
        execution_runtime=runtime,
    )

    assert second.valid
    assert runtime.launches == 2
    assert list((runs / "submission-checks" / "leases" / "lease-archive").glob("*.json"))


def test_submission_cleanup_failure_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    starter = tmp_path / "starter"
    starter.mkdir()
    submit_py = starter / "submit.py"
    submit_py.write_text("raise SystemExit(0)\n", encoding="utf-8")
    monkeypatch.setattr(
        submission_module,
        "verify_starter_manifest",
        lambda: SimpleNamespace(
            root=starter,
            manifest_sha256="c" * 64,
            hashes={"submit.py": sha256_file(submit_py)},
        ),
    )
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("REX_RUNS_ROOT", str(runs))
    data = tmp_path / "data"
    data.mkdir()
    csv_path = tmp_path / "submission.csv"
    csv_path.write_text("row_id,user_id,video_id,score\n", encoding="utf-8")
    runtime = ClosedLeaseRuntime(fail_cleanup_once=True)

    with pytest.raises(SubmissionError, match="cleanup failed closed"):
        validate_submission(
            csv_path,
            data_dir=data,
            split="test",
            sandbox_mode=SandboxMode.PRODUCTION,
            execution_runtime=runtime,
        )

    assert list((runs / "submission-checks" / "leases").glob("*.result.json"))
