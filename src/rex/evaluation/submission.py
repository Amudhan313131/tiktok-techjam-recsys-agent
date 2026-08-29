"""Build and validate organizer-aligned submissions without scoring test labels."""

from __future__ import annotations

import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from rex.data.manifest import verify_starter_manifest
from rex.execution.artifacts import load_prediction_artifact


class SubmissionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SubmissionValidation:
    valid: bool
    command: tuple[str, ...]
    stdout: str
    stderr: str
    returncode: int


def build_submission(prediction_path: str | Path, csv_path: str | Path) -> Path:
    arrays = load_prediction_artifact(prediction_path)
    destination = Path(csv_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, user_id, video_id, score in zip(
            arrays["row_id"], arrays["user_id"], arrays["video_id"], arrays["score"], strict=True
        ):
            writer.writerow([int(row_id), str(user_id), str(video_id), f"{float(score):.17g}"])
    temporary.replace(destination)
    return destination


def validate_submission(
    csv_path: str | Path,
    *,
    data_dir: str | Path,
    split: str,
    timeout_seconds: int = 180,
) -> SubmissionValidation:
    if split not in {"valid", "test"}:
        raise SubmissionError(f"unsupported submission split: {split}")
    starter = verify_starter_manifest()
    command = (
        sys.executable,
        str(starter.root / "submit.py"),
        str(Path(csv_path).resolve()),
        "--data_dir",
        str(Path(data_dir).resolve()),
        "--split",
        split,
        "--check",
    )
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds)
    return SubmissionValidation(
        valid=completed.returncode == 0,
        command=command,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


def require_valid_submission(*args, **kwargs) -> SubmissionValidation:
    result = validate_submission(*args, **kwargs)
    if not result.valid:
        raise SubmissionError(
            f"organizer submission validation failed ({result.returncode}): "
            f"{result.stdout}\n{result.stderr}"
        )
    return result
