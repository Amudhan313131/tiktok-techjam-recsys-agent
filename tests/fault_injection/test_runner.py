from __future__ import annotations

import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import psutil

from rex.contracts import AttemptStatus, RunRequest
from rex.data.manifest import sha256_file
from rex.execution.lease import (
    WorkerLeaseError,
    begin_worker_lease,
    recover_orphan_worker,
)
from rex.execution.runner import _typed_failure, execute_request


HASH = "0" * 64


def request(feature_target_paths, tmp_path: Path, config: dict, **overrides) -> RunRequest:
    features, targets = feature_target_paths
    config_path = tmp_path / f"config-{len(list(tmp_path.glob('config-*')))}.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    values = {
        "run_id": "run",
        "experiment_id": "experiment",
        "attempt_id": f"attempt-{time.time_ns()}",
        "commit_sha": "fixture",
        "plugin": "rex.models.experimental.fixture:FixturePlugin",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "seed": 1,
        "rung": "fixture",
        "split": "train",
        "feature_view_path": str(features),
        "target_view_path": str(targets),
        "output_dir": str(tmp_path / f"output-{time.time_ns()}"),
        "deadline_epoch_ms": int((time.time() + 20) * 1000),
        "timeout_seconds": 10,
        "data_view_sha256": sha256_file(features),
        "environment_sha256": HASH,
    }
    values.update(overrides)
    return RunRequest(**values)


def test_successful_worker_emits_valid_checkpoint(feature_target_paths, tmp_path: Path) -> None:
    result = execute_request(request(feature_target_paths, tmp_path, {}), tmp_path / "attempt-success")
    assert result.status == AttemptStatus.SUCCESS
    assert any(artifact.kind == "checkpoint" for artifact in result.artifacts)
    assert any(artifact.kind == "stdout" for artifact in result.artifacts)


def test_timeout_kills_worker_process_group(feature_target_paths, tmp_path: Path) -> None:
    result = execute_request(
        request(feature_target_paths, tmp_path, {"sleep_seconds": 3}, timeout_seconds=1),
        tmp_path / "attempt-timeout",
    )
    assert result.status == AttemptStatus.TIMEOUT
    assert result.wall_seconds < 3


def test_memory_limit_is_typed_oom(feature_target_paths, tmp_path: Path) -> None:
    result = execute_request(
        request(feature_target_paths, tmp_path, {"allocate_mb": 8}, max_memory_mb=1),
        tmp_path / "attempt-oom",
    )
    assert result.status == AttemptStatus.OOM


def test_worker_classifies_explicit_memory_error(feature_target_paths, tmp_path: Path) -> None:
    result = execute_request(
        request(feature_target_paths, tmp_path, {"raise_memory_error": True}),
        tmp_path / "attempt-memory-error",
    )
    assert result.status == AttemptStatus.OOM


def test_worker_classifies_floating_point_failure_as_nan(
    feature_target_paths, tmp_path: Path
) -> None:
    result = execute_request(
        request(feature_target_paths, tmp_path, {"raise_floating_point": True}),
        tmp_path / "attempt-floating-point",
    )
    assert result.status == AttemptStatus.NAN
    assert result.error_type == "FloatingPointError"
    assert result.error_summary == "fixture non-finite loss"


def test_worker_classifies_unhandled_runtime_failure_as_crash(
    feature_target_paths, tmp_path: Path
) -> None:
    result = execute_request(
        request(feature_target_paths, tmp_path, {"raise_crash": True}),
        tmp_path / "attempt-runtime-crash",
    )
    assert result.status == AttemptStatus.CRASH
    assert result.error_type == "RuntimeError"
    assert result.error_summary == "fixture crash"


def test_missing_worker_result_is_invalid_artifact(feature_target_paths, tmp_path: Path) -> None:
    result = execute_request(
        request(feature_target_paths, tmp_path, {}),
        tmp_path / "attempt-missing-result",
        python_executable="/usr/bin/true",
    )
    assert result.status == AttemptStatus.INVALID_ARTIFACT
    assert result.error_type == "InvalidArtifact"
    assert "without a result artifact" in (result.error_summary or "")


def test_malformed_worker_result_is_invalid_artifact(feature_target_paths, tmp_path: Path) -> None:
    fake_python = tmp_path / "malformed-worker"
    fake_python.write_text(
        "#!/bin/sh\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = \"--result\" ]; then shift; printf '{broken' > \"$1\"; exit 0; fi\n"
        "  shift\n"
        "done\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    result = execute_request(
        request(feature_target_paths, tmp_path, {}),
        tmp_path / "attempt-malformed-result",
        python_executable=str(fake_python),
    )
    assert result.status == AttemptStatus.INVALID_ARTIFACT
    assert result.error_type == "InvalidArtifact"
    assert "invalid worker result" in (result.error_summary or "")


def test_timeout_stops_spawned_descendant_process_group(
    feature_target_paths, tmp_path: Path
) -> None:
    fake_python = tmp_path / "worker-with-descendant"
    child_pid_path = tmp_path / "worker-with-descendant.child"
    fake_python.write_text(
        "#!/bin/sh\n"
        "sleep 30 &\n"
        "child=$!\n"
        f"printf '%s\\n' \"$child\" > {str(child_pid_path)!r}\n"
        "wait \"$child\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    result = execute_request(
        request(feature_target_paths, tmp_path, {}, timeout_seconds=1),
        tmp_path / "attempt-descendant-timeout",
        python_executable=str(fake_python),
    )
    assert result.status == AttemptStatus.TIMEOUT
    child_pid = int(child_pid_path.read_text(encoding="utf-8").strip())

    # A killed child can briefly remain as a zombie until its new parent reaps it;
    # both a missing process and a zombie prove it is no longer executing work.
    for _ in range(40):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(child_pid)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if not state or state.startswith("Z"):
            break
        time.sleep(0.05)
    else:
        raise AssertionError(f"descendant process {child_pid} survived worker timeout")


def test_model_bundle_predicts_without_config_mutation(feature_target_paths, tmp_path: Path) -> None:
    fit = execute_request(
        request(feature_target_paths, tmp_path, {}),
        tmp_path / "attempt-bundle-fit",
    )
    assert fit.status == AttemptStatus.SUCCESS
    bundle = next(artifact for artifact in fit.artifacts if artifact.kind == "model_bundle")

    prediction_request = request(
        feature_target_paths,
        tmp_path,
        {},
        operation="predict",
        rung="predict",
        split="valid",
        target_view_path=None,
        model_bundle_path=bundle.path,
    )
    predicted = execute_request(prediction_request, tmp_path / "attempt-bundle-predict")
    assert predicted.status == AttemptStatus.SUCCESS
    assert any(artifact.kind == "predictions" for artifact in predicted.artifacts)
    assert "model_artifact_path" not in json.loads(Path(prediction_request.config_path).read_text())


def test_corrupt_bundle_member_is_invalid_artifact(feature_target_paths, tmp_path: Path) -> None:
    fit = execute_request(
        request(feature_target_paths, tmp_path, {}),
        tmp_path / "attempt-corrupt-fit",
    )
    bundle = next(artifact for artifact in fit.artifacts if artifact.kind == "model_bundle")
    checkpoint = next(artifact for artifact in fit.artifacts if artifact.kind == "checkpoint")
    Path(checkpoint.path).write_text("corrupt", encoding="utf-8")

    prediction_request = request(
        feature_target_paths,
        tmp_path,
        {},
        operation="predict",
        rung="predict",
        split="valid",
        target_view_path=None,
        model_bundle_path=bundle.path,
    )
    result = execute_request(prediction_request, tmp_path / "attempt-corrupt-predict")
    assert result.status == AttemptStatus.INVALID_ARTIFACT


def test_missing_bundle_member_is_invalid_artifact(feature_target_paths, tmp_path: Path) -> None:
    fit = execute_request(
        request(feature_target_paths, tmp_path, {}),
        tmp_path / "attempt-missing-fit",
    )
    bundle = next(artifact for artifact in fit.artifacts if artifact.kind == "model_bundle")
    checkpoint = next(artifact for artifact in fit.artifacts if artifact.kind == "checkpoint")
    Path(checkpoint.path).unlink()

    prediction_request = request(
        feature_target_paths,
        tmp_path,
        {},
        operation="predict",
        rung="predict",
        split="valid",
        target_view_path=None,
        model_bundle_path=bundle.path,
    )
    result = execute_request(prediction_request, tmp_path / "attempt-missing-predict")
    assert result.status == AttemptStatus.INVALID_ARTIFACT


def test_nonfinite_prediction_is_typed_nan(feature_target_paths, tmp_path: Path) -> None:
    config = {"nan_scores": True}
    fit = execute_request(
        request(feature_target_paths, tmp_path, config),
        tmp_path / "attempt-nan-fit",
    )
    bundle = next(artifact for artifact in fit.artifacts if artifact.kind == "model_bundle")
    result = execute_request(
        request(
            feature_target_paths,
            tmp_path,
            config,
            operation="predict",
            rung="predict",
            split="valid",
            target_view_path=None,
            model_bundle_path=bundle.path,
        ),
        tmp_path / "attempt-nan-predict",
    )
    assert result.status == AttemptStatus.NAN


def test_external_worker_signal_is_typed_interrupted() -> None:
    assert _typed_failure("", False, return_code=-signal.SIGKILL) == AttemptStatus.INTERRUPTED


def test_completed_worker_result_is_replayed_without_retraining(
    feature_target_paths, tmp_path: Path
) -> None:
    run_request = request(feature_target_paths, tmp_path, {})
    attempt_dir = tmp_path / "attempt-replay"
    first = execute_request(run_request, attempt_dir)
    assert first.status == AttemptStatus.SUCCESS
    checkpoint = next(item for item in first.artifacts if item.kind == "checkpoint")
    result_path = attempt_dir / "worker" / "result.json"
    stdout_path = attempt_dir / "stdout.log"
    observed = {
        "checkpoint": Path(checkpoint.path).stat().st_mtime_ns,
        "result": result_path.stat().st_mtime_ns,
        "stdout": stdout_path.stat().st_mtime_ns,
    }

    replayed = execute_request(run_request, attempt_dir)

    assert replayed.status == AttemptStatus.SUCCESS
    assert Path(checkpoint.path).stat().st_mtime_ns == observed["checkpoint"]
    assert result_path.stat().st_mtime_ns == observed["result"]
    assert stdout_path.stat().st_mtime_ns == observed["stdout"]
    replay_ref = next(item for item in replayed.artifacts if item.kind == "worker_replay")
    replay = json.loads(Path(replay_ref.path).read_text(encoding="utf-8"))
    assert replay["outcome"] == "complete-result-replayed"


def test_same_attempt_with_conflicting_request_hash_fails_closed(
    feature_target_paths, tmp_path: Path
) -> None:
    original = request(
        feature_target_paths,
        tmp_path,
        {},
        attempt_id="attempt-request-conflict",
    )
    attempt_dir = tmp_path / "attempt-request-conflict"
    first = execute_request(original, attempt_dir)
    assert first.status == AttemptStatus.SUCCESS
    checkpoint = next(item for item in first.artifacts if item.kind == "checkpoint")
    checkpoint_mtime = Path(checkpoint.path).stat().st_mtime_ns
    values = original.model_dump()
    values["seed"] = original.seed + 1

    conflict = execute_request(RunRequest(**values), attempt_dir)

    assert conflict.status == AttemptStatus.CONTRACT
    assert conflict.error_type == "WorkerLeaseConflict"
    assert "different request hash" in (conflict.error_summary or "")
    assert Path(checkpoint.path).stat().st_mtime_ns == checkpoint_mtime


def _stopped_or_zombie(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return not state or state.startswith("Z")


def test_resume_reaps_verified_orphan_group_after_coordinator_sigkill(
    feature_target_paths, tmp_path: Path
) -> None:
    first_run_marker = tmp_path / "orphan-first-run.marker"
    descendant_path = tmp_path / "orphan-descendant.pid"
    fake_worker = tmp_path / "orphan-worker"
    fake_worker.write_text(
        "#!/bin/sh\n"
        f"if [ ! -e {shlex.quote(str(first_run_marker))} ]; then\n"
        f"  touch {shlex.quote(str(first_run_marker))}\n"
        "  sleep 30 &\n"
        "  child=$!\n"
        f"  printf '%s\\n' \"$child\" > {shlex.quote(str(descendant_path))}\n"
        "  wait \"$child\"\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_worker.chmod(0o755)
    run_request = request(
        feature_target_paths,
        tmp_path,
        {},
        attempt_id="attempt-orphan-recovery",
        timeout_seconds=20,
        deadline_epoch_ms=int((time.time() + 30) * 1000),
    )
    serialized_request = tmp_path / "orphan-request.json"
    serialized_request.write_text(run_request.model_dump_json(), encoding="utf-8")
    attempt_dir = tmp_path / "attempt-orphan-recovery"
    coordinator_code = (
        "import sys; from pathlib import Path; from rex.contracts import RunRequest; "
        "from rex.execution.runner import execute_request; "
        "request=RunRequest.model_validate_json(Path(sys.argv[1]).read_text()); "
        "execute_request(request, sys.argv[2], python_executable=sys.argv[3])"
    )
    coordinator = subprocess.Popen(
        [
            sys.executable,
            "-c",
            coordinator_code,
            str(serialized_request),
            str(attempt_dir),
            str(fake_worker),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "REX_VOLATILE_AMBIENT": "child-coordinator-only"},
    )
    lease_path = attempt_dir / "worker_lease.json"
    deadline = time.monotonic() + 8
    lease: dict[str, object] | None = None
    while time.monotonic() < deadline:
        if lease_path.is_file() and descendant_path.is_file():
            candidate = json.loads(lease_path.read_text(encoding="utf-8"))
            if candidate.get("state") == "active":
                lease = candidate
                break
        time.sleep(0.05)
    assert lease is not None, "coordinator never persisted an active worker lease"
    old_worker_pid = int(lease["pid"])
    descendant_pid = int(descendant_path.read_text(encoding="utf-8").strip())
    assert not _stopped_or_zombie(old_worker_pid)
    assert not _stopped_or_zombie(descendant_pid)
    observed_command = psutil.Process(old_worker_pid).cmdline()
    observed_token_hashes = {
        hashlib.sha256(argument.encode("utf-8")).hexdigest()
        for argument in observed_command
    }
    assert lease["identity_token_sha256"] in observed_token_hashes

    os.kill(coordinator.pid, signal.SIGKILL)
    coordinator.wait(timeout=5)
    assert not _stopped_or_zombie(old_worker_pid)

    resumed = execute_request(run_request, attempt_dir, python_executable=str(fake_worker))

    assert resumed.status == AttemptStatus.INVALID_ARTIFACT, resumed.error_summary
    recovery_ref = next(item for item in resumed.artifacts if item.kind == "worker_recovery")
    recovery = json.loads(Path(recovery_ref.path).read_text(encoding="utf-8"))
    event = recovery["events"][-1]
    assert event["outcome"] == "orphan-process-group-terminated"
    assert event["pid"] == old_worker_pid
    for _ in range(40):
        if _stopped_or_zombie(old_worker_pid) and _stopped_or_zombie(descendant_pid):
            break
        time.sleep(0.05)
    else:
        raise AssertionError("verified orphan worker process group survived recovery")


def test_pid_reuse_identity_mismatch_never_signals_process(tmp_path: Path) -> None:
    process = subprocess.Popen(["sleep", "30"], start_new_session=True)
    lease_path = tmp_path / "reused-pid-lease.json"
    recovery_path = tmp_path / "reused-pid-recovery.json"
    try:
        marker = begin_worker_lease(
            lease_path,
            pid=process.pid,
            request_sha256="1" * 64,
            execution_sha256="2" * 64,
            planned_command_sha256="3" * 64,
        )
        marker["create_time"] = float(marker["create_time"]) - 60
        lease_path.write_text(json.dumps(marker), encoding="utf-8")
        with pytest.raises(WorkerLeaseError, match="PID was reused"):
            recover_orphan_worker(
                lease_path,
                recovery_path,
                request_sha256="1" * 64,
                execution_sha256="2" * 64,
            )
        assert process.poll() is None
        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
        assert recovery["events"][-1]["outcome"] == "recovery-refused"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def test_foreign_active_lease_fails_closed_without_signaling(tmp_path: Path) -> None:
    process = subprocess.Popen(["sleep", "30"], start_new_session=True)
    lease_path = tmp_path / "foreign-lease.json"
    recovery_path = tmp_path / "foreign-recovery.json"
    try:
        marker = begin_worker_lease(
            lease_path,
            pid=process.pid,
            request_sha256="4" * 64,
            execution_sha256="5" * 64,
            planned_command_sha256="6" * 64,
        )
        marker["hostname"] = "different-host.invalid"
        lease_path.write_text(json.dumps(marker), encoding="utf-8")
        with pytest.raises(WorkerLeaseError, match="foreign host or boot"):
            recover_orphan_worker(
                lease_path,
                recovery_path,
                request_sha256="4" * 64,
                execution_sha256="5" * 64,
            )
        assert process.poll() is None
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
