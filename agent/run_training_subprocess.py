"""Compatibility wrapper for the process-group isolated typed runner."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rex.contracts import RunRequest, RunResult  # noqa: E402
from rex.execution.runner import execute_request  # noqa: E402


def run_training(request: RunRequest, attempt_dir: str | Path) -> RunResult:
    return execute_request(request, attempt_dir)
