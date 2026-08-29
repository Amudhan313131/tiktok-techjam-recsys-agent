from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from rex.contracts import AttemptStatus, RunRequest
from rex.data.manifest import sha256_file
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
