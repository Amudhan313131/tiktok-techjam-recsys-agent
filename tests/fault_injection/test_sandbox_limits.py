from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rex.execution.gate import execute_gate
from rex.execution.limits import ResourceLimits, limits_for_request, resource_limit_preexec
from rex.execution.sandbox import SandboxMode


def test_posix_limits_are_applied_before_worker_exec() -> None:
    limits = ResourceLimits(
        cpu_seconds=7,
        address_space_bytes=None,
        open_files=32,
        processes=16,
        file_size_bytes=1024 * 1024,
        core_size_bytes=0,
    )
    script = (
        "import json,resource; print(json.dumps({"
        "'cpu': resource.getrlimit(resource.RLIMIT_CPU)[0],"
        "'files': resource.getrlimit(resource.RLIMIT_NOFILE)[0],"
        "'fsize': resource.getrlimit(resource.RLIMIT_FSIZE)[0],"
        "'core': resource.getrlimit(resource.RLIMIT_CORE)[0]}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        timeout=5,
        check=True,
        preexec_fn=resource_limit_preexec(limits),
    )
    observed = json.loads(completed.stdout)
    assert observed == {"cpu": 7, "files": 32, "fsize": 1024 * 1024, "core": 0}


def test_process_limit_is_part_of_production_limit_contract() -> None:
    limits = ResourceLimits(cpu_seconds=1, address_space_bytes=None, processes=11)
    assert limits.to_dict()["processes"] == 11
    assert hasattr(resource, "RLIMIT_NPROC")


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS resource contract")
def test_macos_uses_runner_rss_monitor_instead_of_broken_rlimit_as() -> None:
    limits = limits_for_request(timeout_seconds=60, max_memory_mb=512)
    assert limits.address_space_bytes is None


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox backend test")
def test_sandboxed_gate_timeout_kills_spawned_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REX_PRODUCTION_RUNTIME", "native_macos")
    monkeypatch.setenv("REX_ALLOW_NATIVE_MACOS_ROLLBACK", "1")
    trusted_root = tmp_path / "worktrees"
    workspace = trusted_root / "candidate"
    workspace.mkdir(parents=True)
    for arguments in (
        ("init",),
        ("config", "user.email", "rex@example.invalid"),
        ("config", "user.name", "REX Sandbox Test"),
        ("commit", "--allow-empty", "-m", "base"),
    ):
        completed = subprocess.run(
            ["git", *arguments],
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
    artifacts = tmp_path / "gate-artifacts"
    script = (
        "import os, pathlib, subprocess, sys, time; "
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "pathlib.Path(os.environ['TMPDIR'], 'child.pid').write_text(str(child.pid)); "
        "time.sleep(30)"
    )
    result = execute_gate(
        name="timeout",
        command=(sys.executable, "-c", script),
        workspace=workspace,
        artifact_dir=artifacts,
        timeout_seconds=1,
        sandbox_mode=SandboxMode.PRODUCTION,
        trusted_worktree_root=trusted_root,
        trusted_output_root=tmp_path,
    )
    assert result.timed_out
    child_pid = int((artifacts / "timeout-sandbox-temp" / "child.pid").read_text(encoding="utf-8"))
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
        raise AssertionError(f"sandboxed gate descendant {child_pid} survived timeout")
