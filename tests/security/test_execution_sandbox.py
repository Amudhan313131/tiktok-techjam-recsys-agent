from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rex.contracts import AttemptStatus, RunRequest
from rex.data.manifest import sha256_file
from rex.execution.gate import execute_gate
from rex.execution.runner import execute_request
from rex.execution.sandbox import (
    SandboxError,
    SandboxMode,
    production_backend,
    sanitized_environment,
)
from rex.execution.sandbox_macos import MacOSSandboxBackend


HASH = "0" * 64


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _sandbox_worktree(tmp_path: Path) -> tuple[Path, Path, str]:
    trusted_root = tmp_path / "trusted-worktrees"
    project = trusted_root / "candidate"
    repository = Path(__file__).resolve().parents[2]
    shutil.copytree(
        repository / "src" / "rex",
        project / "src" / "rex",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(repository / "configs" / "frozen", project / "configs" / "frozen")
    probe = project / "src" / "rex" / "models" / "experimental" / "sandbox_probe.py"
    probe.write_text(
        """import errno
import json
import os
from pathlib import Path
import socket
import numpy as np

class SandboxProbe:
    def fit(self, train_features, train_targets, config, seed, output_dir):
        observations = {}
        try:
            Path(config['undeclared_read']).read_text(encoding='utf-8')
            observations['read_denied'] = False
        except OSError as error:
            observations['read_denied'] = error.errno in {errno.EPERM, errno.EACCES}
        try:
            Path(config['undeclared_write']).write_text('escaped', encoding='utf-8')
            observations['write_denied'] = False
        except OSError as error:
            observations['write_denied'] = error.errno in {errno.EPERM, errno.EACCES}
        try:
            socket.create_connection(('127.0.0.1', 9), timeout=0.1)
            observations['network_denied'] = False
        except OSError as error:
            observations['network_denied'] = error.errno in {errno.EPERM, errno.EACCES}
        observations['secret_absent'] = 'REX_TEST_SECRET' not in os.environ
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / 'probe.json'
        path.write_text(json.dumps(observations, sort_keys=True), encoding='utf-8')
        return path

    def predict(self, model_artifact, features, config, output_dir):
        return np.zeros(features.rows, dtype=np.float64)
""",
        encoding="utf-8",
    )
    _git(project, "init")
    _git(project, "config", "user.email", "rex@example.invalid")
    _git(project, "config", "user.name", "REX Sandbox Test")
    _git(project, "add", "--all")
    _git(project, "commit", "-m", "sandbox fixture")
    return trusted_root, project, _git(project, "rev-parse", "HEAD")


def _request(
    feature_target_paths: tuple[Path, Path],
    tmp_path: Path,
    project: Path,
    commit: str,
) -> RunRequest:
    features, targets = feature_target_paths
    secret = tmp_path / "undeclared-secret.txt"
    secret.write_text("do not expose", encoding="utf-8")
    config_path = tmp_path / "sandbox-config.json"
    config_path.write_text(
        json.dumps(
            {
                "undeclared_read": str(secret),
                "undeclared_write": str(tmp_path / "escaped.txt"),
            }
        ),
        encoding="utf-8",
    )
    return RunRequest(
        run_id="run",
        experiment_id="sandbox-experiment",
        attempt_id=f"attempt-{time.time_ns()}",
        commit_sha=commit,
        plugin="rex.models.experimental.sandbox_probe:SandboxProbe",
        config_path=str(config_path),
        config_sha256=sha256_file(config_path),
        seed=1,
        rung="cheap",
        split="shadow",
        feature_view_path=str(features),
        target_view_path=str(targets),
        workspace_path=str(project),
        output_dir=str(tmp_path / "outputs" / "fit"),
        deadline_epoch_ms=int((time.time() + 30) * 1000),
        timeout_seconds=15,
        data_view_sha256=sha256_file(features),
        environment_sha256=HASH,
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox backend test")
def test_macos_sandbox_doctor_proves_undeclared_reads_are_denied() -> None:
    result = MacOSSandboxBackend().doctor()
    assert result.available
    assert result.safe_for_production, result.detail


def test_sanitized_environment_drops_credentials_and_user_home(tmp_path: Path) -> None:
    environment = sanitized_environment(
        {
            "PATH": "/usr/bin",
            "LANG": "en_US.UTF-8",
            "OPENAI_API_KEY": "secret",
            "ANTHROPIC_API_KEY": "secret",
            "SSH_AUTH_SOCK": "/tmp/agent",
            "HOME": "/Users/example",
        },
        workspace=tmp_path / "workspace",
        temp_dir=tmp_path / "private-temp",
    )
    assert environment["PATH"] == "/usr/bin"
    assert environment["HOME"] == str(tmp_path / "private-temp")
    assert "OPENAI_API_KEY" not in environment
    assert "ANTHROPIC_API_KEY" not in environment
    assert "SSH_AUTH_SOCK" not in environment


def test_production_backend_fails_closed_on_unsupported_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "UnsupportedOS")
    with pytest.raises(SandboxError, match="no production sandbox backend"):
        production_backend()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox backend test")
def test_production_worker_has_only_declared_capabilities(
    feature_target_paths: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REX_PRODUCTION_RUNTIME", "native_macos")
    monkeypatch.setenv("REX_ALLOW_NATIVE_MACOS_ROLLBACK", "1")
    trusted_root, project, commit = _sandbox_worktree(tmp_path)
    request = _request(feature_target_paths, tmp_path, project, commit)
    monkeypatch.setenv("REX_TEST_SECRET", "must-not-reach-worker")

    result = execute_request(
        request,
        tmp_path / "attempt",
        trusted_worktree_root=trusted_root,
        trusted_output_root=tmp_path,
        sandbox_mode=SandboxMode.PRODUCTION,
    )

    assert result.status == AttemptStatus.SUCCESS, result.error_summary
    checkpoint = next(item for item in result.artifacts if item.kind == "checkpoint")
    observations = json.loads(Path(checkpoint.path).read_text(encoding="utf-8"))
    assert observations == {
        "network_denied": True,
        "read_denied": True,
        "secret_absent": True,
        "write_denied": True,
    }
    assert not (tmp_path / "escaped.txt").exists()
    evidence_ref = next(item for item in result.artifacts if item.kind == "sandbox_evidence")
    evidence = json.loads(Path(evidence_ref.path).read_text(encoding="utf-8"))
    assert evidence["sandboxed"] is True
    assert evidence["backend"] == "macos-sandbox-exec"
    assert evidence["policy"]["network_allowed"] is False
    assert "REX_TEST_SECRET" not in evidence["environment_keys"]
    assert len(evidence["policy_sha256"]) == 64
    assert len(evidence["profile_sha256"]) == 64
    profile_ref = next(item for item in result.artifacts if item.kind == "sandbox_profile")
    assert sha256_file(profile_ref.path) == evidence["profile_sha256"]

    bundle = next(item for item in result.artifacts if item.kind == "model_bundle")
    values = request.model_dump()
    values.update(
        {
            "attempt_id": f"attempt-predict-{time.time_ns()}",
            "operation": "predict",
            "rung": "predict",
            "split": "valid",
            "target_view_path": None,
            "model_bundle_path": bundle.path,
            "output_dir": str(tmp_path / "outputs" / "predict"),
        }
    )
    prediction = execute_request(
        RunRequest(**values),
        tmp_path / "attempt-predict",
        trusted_worktree_root=trusted_root,
        trusted_output_root=tmp_path,
        sandbox_mode=SandboxMode.PRODUCTION,
    )
    assert prediction.status == AttemptStatus.SUCCESS, prediction.error_summary
    assert any(item.kind == "predictions" for item in prediction.artifacts)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox backend test")
def test_production_candidate_gate_cannot_escape_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REX_PRODUCTION_RUNTIME", "native_macos")
    monkeypatch.setenv("REX_ALLOW_NATIVE_MACOS_ROLLBACK", "1")
    trusted_root, project, _commit = _sandbox_worktree(tmp_path)
    secret = tmp_path / "gate-secret.txt"
    escaped = tmp_path / "gate-escaped.txt"
    secret.write_text("do not expose", encoding="utf-8")
    script = f"""
import errno, json, os, socket
from pathlib import Path
observed = {{}}
try:
    Path({str(secret)!r}).read_text()
    observed['read_denied'] = False
except OSError as error:
    observed['read_denied'] = error.errno in {{errno.EPERM, errno.EACCES}}
try:
    Path({str(escaped)!r}).write_text('escaped')
    observed['write_denied'] = False
except OSError as error:
    observed['write_denied'] = error.errno in {{errno.EPERM, errno.EACCES}}
try:
    socket.create_connection(('127.0.0.1', 9), timeout=0.1)
    observed['network_denied'] = False
except OSError as error:
    observed['network_denied'] = error.errno in {{errno.EPERM, errno.EACCES}}
observed['credential_absent'] = 'OPENAI_API_KEY' not in os.environ
print(json.dumps(observed, sort_keys=True))
"""
    result = execute_gate(
        name="fixture",
        command=(sys.executable, "-c", script),
        workspace=project,
        artifact_dir=tmp_path / "gate-artifacts",
        timeout_seconds=10,
        sandbox_mode=SandboxMode.PRODUCTION,
        trusted_worktree_root=trusted_root,
        trusted_output_root=tmp_path,
    )
    assert result.successful, result.stderr
    assert json.loads(result.stdout) == {
        "credential_absent": True,
        "network_denied": True,
        "read_denied": True,
        "write_denied": True,
    }
    assert not escaped.exists()
    assert result.profile_path is not None
    evidence = json.loads(result.evidence_path.read_text(encoding="utf-8"))
    assert evidence["sandboxed"] is True
    assert evidence["gate"] == "fixture"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox backend test")
def test_production_sandbox_can_load_pinned_lightgbm_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REX_PRODUCTION_RUNTIME", "native_macos")
    monkeypatch.setenv("REX_ALLOW_NATIVE_MACOS_ROLLBACK", "1")
    lightgbm = pytest.importorskip("lightgbm")
    trusted_root, project, _commit = _sandbox_worktree(tmp_path)

    result = execute_gate(
        name="lightgbm-runtime",
        command=(
            sys.executable,
            "-c",
            "import lightgbm; print(lightgbm.__version__)",
        ),
        workspace=project,
        artifact_dir=tmp_path / "lightgbm-gate",
        timeout_seconds=15,
        sandbox_mode=SandboxMode.PRODUCTION,
        trusted_worktree_root=trusted_root,
        trusted_output_root=tmp_path,
    )

    assert result.successful, result.stderr
    assert result.stdout.strip() == lightgbm.__version__


def test_production_execution_fails_closed_without_verified_worktree(
    feature_target_paths: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REX_PRODUCTION_RUNTIME", "native_macos")
    monkeypatch.setenv("REX_ALLOW_NATIVE_MACOS_ROLLBACK", "1")
    features, targets = feature_target_paths
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    request = RunRequest(
        run_id="run",
        experiment_id="experiment",
        attempt_id="attempt",
        commit_sha="fixture",
        plugin="rex.models.experimental.fixture:FixturePlugin",
        config_path=str(config_path),
        config_sha256=sha256_file(config_path),
        seed=1,
        rung="cheap",
        split="shadow",
        feature_view_path=str(features),
        target_view_path=str(targets),
        output_dir=str(tmp_path / "output"),
        deadline_epoch_ms=int((time.time() + 20) * 1000),
        timeout_seconds=10,
        data_view_sha256=sha256_file(features),
        environment_sha256=HASH,
    )
    result = execute_request(
        request,
        tmp_path / "attempt",
        sandbox_mode=SandboxMode.PRODUCTION,
        trusted_output_root=tmp_path,
    )
    assert result.status == AttemptStatus.CONTRACT
    assert result.error_type == "SandboxUnavailable"
    assert "verified candidate worktree" in (result.error_summary or "")


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox backend test")
def test_production_worker_rejects_output_outside_trusted_root(
    feature_target_paths: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REX_PRODUCTION_RUNTIME", "native_macos")
    monkeypatch.setenv("REX_ALLOW_NATIVE_MACOS_ROLLBACK", "1")
    trusted_root, project, commit = _sandbox_worktree(tmp_path)
    request = _request(feature_target_paths, tmp_path, project, commit)
    values = request.model_dump()
    values["output_dir"] = str(tmp_path.parent / "undeclared-output")
    result = execute_request(
        RunRequest(**values),
        tmp_path / "attempt-outside-output",
        trusted_worktree_root=trusted_root,
        trusted_output_root=tmp_path,
        sandbox_mode=SandboxMode.PRODUCTION,
    )
    assert result.status == AttemptStatus.CONTRACT
    assert result.error_type == "SandboxUnavailable"
    assert "worker output directory is outside its trusted root" in (result.error_summary or "")
