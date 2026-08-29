"""Compatibility wrapper around the immutable organizer submission checker."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rex.evaluation.submission import validate_submission  # noqa: E402


def validate(csv_path: str, *, data_dir: str, split: str) -> dict[str, object]:
    result = validate_submission(csv_path, data_dir=data_dir, split=split)
    return {
        "valid": result.valid,
        "output": result.stdout + result.stderr,
        "command": list(result.command),
        "returncode": result.returncode,
    }
