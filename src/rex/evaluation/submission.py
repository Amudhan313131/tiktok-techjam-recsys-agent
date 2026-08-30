"""Build and validate organizer-aligned submissions without scoring test labels."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from rex.data.manifest import verify_starter_manifest
from rex.data.views import FeatureView, load_feature_view
from rex.execution.artifacts import ArtifactError, load_prediction_artifact
from rex.execution.limits import limits_for_request, resource_limit_preexec
from rex.execution.sandbox import (
    SandboxMode,
    SandboxPolicy,
    production_backend,
    sanitized_environment,
)


TEST_ROW_COUNT = 170_588
SUBMISSION_HEADER = ("row_id", "user_id", "video_id", "score")


class SubmissionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SubmissionValidation:
    valid: bool
    command: tuple[str, ...]
    stdout: str
    stderr: str
    returncode: int
    sandbox_evidence: dict[str, object] = field(default_factory=dict)


def build_submission(
    prediction_path: str | Path,
    csv_path: str | Path,
    *,
    expected_features: FeatureView | str | Path,
    expected_rows: int = TEST_ROW_COUNT,
) -> Path:
    """Write a CSV only after exact canonical-view alignment has been proven.

    ``expected_features`` is deliberately mandatory: a prediction artifact is
    self-consistent even when it was generated for the wrong split, so production
    submission construction must compare it with the frozen canonical test view.
    """

    if expected_rows <= 0:
        raise SubmissionError("expected submission row count must be positive")
    view = (
        expected_features
        if isinstance(expected_features, FeatureView)
        else load_feature_view(expected_features)
    )
    if view.rows != expected_rows:
        raise SubmissionError(
            f"canonical feature row count mismatch: expected {expected_rows}, observed {view.rows}"
        )
    try:
        arrays = load_prediction_artifact(prediction_path, expected_features=view)
    except (ArtifactError, OSError, ValueError) as error:
        raise SubmissionError(f"invalid test prediction artifact: {error}") from error
    if len(arrays["score"]) != expected_rows:
        raise SubmissionError(
            f"prediction row count mismatch: expected {expected_rows}, "
            f"observed {len(arrays['score'])}"
        )

    destination = Path(csv_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(SUBMISSION_HEADER)
            for row_id, user_id, video_id, score in zip(
                arrays["row_id"],
                arrays["user_id"],
                arrays["video_id"],
                arrays["score"],
                strict=True,
            ):
                writer.writerow([int(row_id), str(user_id), str(video_id), f"{float(score):.17g}"])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def validate_submission_matches_predictions(
    csv_path: str | Path,
    prediction_path: str | Path,
    *,
    expected_features: FeatureView | str | Path,
    expected_rows: int = TEST_ROW_COUNT,
) -> None:
    """Prove a CSV is an order-preserving rendering of one prediction artifact."""

    view = (
        expected_features
        if isinstance(expected_features, FeatureView)
        else load_feature_view(expected_features)
    )
    if view.rows != expected_rows:
        raise SubmissionError(
            f"canonical feature row count mismatch: expected {expected_rows}, observed {view.rows}"
        )
    try:
        arrays = load_prediction_artifact(prediction_path, expected_features=view)
    except (ArtifactError, OSError, ValueError) as error:
        raise SubmissionError(f"invalid test prediction artifact: {error}") from error
    path = Path(csv_path)
    observed_rows = 0
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            if tuple(next(reader, ())) != SUBMISSION_HEADER:
                raise SubmissionError("submission header must be row_id,user_id,video_id,score")
            for observed_rows, record in enumerate(reader, start=1):
                if len(record) != 4:
                    raise SubmissionError(
                        f"submission row {observed_rows + 1} must contain exactly four fields"
                    )
                if observed_rows > expected_rows:
                    raise SubmissionError("submission contains more rows than the canonical test view")
                index = observed_rows - 1
                expected = (
                    str(int(arrays["row_id"][index])),
                    str(arrays["user_id"][index]),
                    str(arrays["video_id"][index]),
                )
                if tuple(record[:3]) != expected:
                    raise SubmissionError(
                        f"submission alignment mismatch at data row {observed_rows}"
                    )
                try:
                    score = float(record[3])
                except ValueError as error:
                    raise SubmissionError(
                        f"submission score is not numeric at data row {observed_rows}"
                    ) from error
                if score != float(arrays["score"][index]):
                    raise SubmissionError(
                        f"submission score differs from prediction artifact at data row {observed_rows}"
                    )
    except (OSError, UnicodeError) as error:
        raise SubmissionError(f"cannot read submission CSV: {error}") from error
    if observed_rows != expected_rows:
        raise SubmissionError(
            f"submission row count mismatch: expected {expected_rows}, observed {observed_rows}"
        )


def _checker_command(csv_path: Path, data_dir: Path, split: str, submit_py: Path) -> tuple[str, ...]:
    command = (
        sys.executable,
        str(submit_py),
        str(csv_path),
        "--data_dir",
        str(data_dir),
        "--split",
        split,
        "--check",
    )
    forbidden = {"--score", "--make"}.intersection(command)
    if forbidden or command[-1] != "--check":  # pragma: no cover - invariant guard
        raise SubmissionError("organizer checker command attempted a forbidden mode")
    return command


def validate_submission(
    csv_path: str | Path,
    *,
    data_dir: str | Path,
    split: str,
    timeout_seconds: int = 180,
    sandbox_mode: SandboxMode = SandboxMode.PRODUCTION,
) -> SubmissionValidation:
    """Run only the frozen organizer check mode under a no-network sandbox."""

    if split not in {"valid", "test"}:
        raise SubmissionError(f"unsupported submission split: {split}")
    if timeout_seconds <= 0:
        raise SubmissionError("submission checker timeout must be positive")
    submission = Path(csv_path).resolve()
    data_root = Path(data_dir).resolve()
    if not submission.is_file():
        raise SubmissionError(f"submission CSV is missing: {submission}")
    if not data_root.is_dir():
        raise SubmissionError(f"submission data directory is missing: {data_root}")
    starter = verify_starter_manifest()
    submit_py = starter.root / "submit.py"
    command = _checker_command(submission, data_root, split, submit_py)
    mode = SandboxMode(sandbox_mode)

    try:
        with tempfile.TemporaryDirectory(prefix="rex-submission-check-") as temporary:
            temp_root = Path(temporary).resolve()
            environment = sanitized_environment(
                workspace=starter.root,
                temp_dir=temp_root,
            )
            limits = limits_for_request(timeout_seconds, max_memory_mb=4096)
            if mode == SandboxMode.PRODUCTION:
                policy = SandboxPolicy(
                    workspace=starter.root.resolve(),
                    read_paths=(starter.root.resolve(), data_root, submission),
                    write_paths=(temp_root,),
                    network_allowed=False,
                    resource_limits=limits,
                )
                prepared = production_backend().prepare(
                    policy,
                    command,
                    environment,
                    temp_root / "checker.sb",
                )
                execution_command = prepared.command
                execution_environment = prepared.environment
                preexec_fn = prepared.preexec_fn
                sandbox_evidence = {
                    **prepared.evidence,
                    "starter_manifest_sha256": starter.manifest_sha256,
                    "submit_py_sha256": starter.hashes["submit.py"],
                    "logical_command": list(command),
                }
            else:
                execution_command = command
                execution_environment = environment
                preexec_fn = resource_limit_preexec(limits)
                sandbox_evidence = {
                    "schema_version": "1.0",
                    "mode": "fixture",
                    "sandboxed": False,
                    "network_allowed": None,
                    "starter_manifest_sha256": starter.manifest_sha256,
                    "submit_py_sha256": starter.hashes["submit.py"],
                    "logical_command": list(command),
                }
            completed = subprocess.run(
                execution_command,
                cwd=starter.root,
                env=execution_environment,
                preexec_fn=preexec_fn,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
    except subprocess.TimeoutExpired as error:
        raise SubmissionError(
            f"organizer submission checker exceeded {timeout_seconds} seconds"
        ) from error

    return SubmissionValidation(
        valid=completed.returncode == 0,
        command=command,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
        sandbox_evidence=sandbox_evidence,
    )


def require_valid_submission(*args, **kwargs) -> SubmissionValidation:
    result = validate_submission(*args, **kwargs)
    if not result.valid:
        raise SubmissionError(
            f"organizer submission validation failed ({result.returncode}): "
            f"{result.stdout}\n{result.stderr}"
        )
    return result
