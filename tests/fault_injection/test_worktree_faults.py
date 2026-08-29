from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from rex.contracts import AttemptStatus, RunRequest
from rex.data.manifest import sha256_file
from rex.execution.runner import execute_request


HASH = "0" * 64


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _workspace(trusted_root: Path) -> tuple[Path, str]:
    workspace = trusted_root / "candidate"
    workspace.mkdir(parents=True)
    (workspace / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _git(workspace, "init")
    _git(workspace, "config", "user.email", "rex@example.invalid")
    _git(workspace, "config", "user.name", "REX Fault Fixture")
    _git(workspace, "add", "tracked.txt")
    _git(workspace, "commit", "-m", "fixture")
    return workspace, _git(workspace, "rev-parse", "HEAD")


def _request(feature_target_paths, tmp_path: Path, workspace: Path, commit: str) -> RunRequest:
    features, targets = feature_target_paths
    config_path = tmp_path / f"worktree-fault-{time.time_ns()}.json"
    config_path.write_text(json.dumps({}), encoding="utf-8")
    return RunRequest(
        run_id="run",
        experiment_id="worktree-fault",
        attempt_id=f"attempt-{time.time_ns()}",
        commit_sha=commit,
        plugin="rex.models.experimental.fixture:FixturePlugin",
        config_path=str(config_path),
        config_sha256=sha256_file(config_path),
        seed=0,
        rung="fixture",
        split="train",
        feature_view_path=str(features),
        target_view_path=str(targets),
        workspace_path=str(workspace),
        output_dir=str(tmp_path / f"output-{time.time_ns()}"),
        deadline_epoch_ms=int((time.time() + 20) * 1000),
        timeout_seconds=10,
        data_view_sha256=sha256_file(features),
        environment_sha256=HASH,
    )


def test_runner_rejects_worktree_commit_mismatch(feature_target_paths, tmp_path: Path) -> None:
    trusted_root = tmp_path / "trusted"
    workspace, _ = _workspace(trusted_root)
    result = execute_request(
        _request(feature_target_paths, tmp_path, workspace, "f" * 40),
        tmp_path / "commit-mismatch",
        trusted_worktree_root=trusted_root,
    )
    assert result.status == AttemptStatus.CONTRACT
    assert result.error_type == "WorkspaceViolation"
    assert "commit mismatch" in (result.error_summary or "")


def test_runner_rejects_dirty_worktree(feature_target_paths, tmp_path: Path) -> None:
    trusted_root = tmp_path / "trusted"
    workspace, commit = _workspace(trusted_root)
    (workspace / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    result = execute_request(
        _request(feature_target_paths, tmp_path, workspace, commit),
        tmp_path / "dirty-worktree",
        trusted_worktree_root=trusted_root,
    )
    assert result.status == AttemptStatus.CONTRACT
    assert result.error_type == "WorkspaceViolation"
    assert "clean Git worktree" in (result.error_summary or "")
