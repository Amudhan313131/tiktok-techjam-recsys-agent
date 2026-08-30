from __future__ import annotations

import pytest
from pydantic import ValidationError

from rex.contracts import (
    FinalSubmissionSpec,
    RehearsalR3Spec,
    SubmissionCheckEvidence,
)


SHA = "a" * 64


def test_r3_caps_wall_time_and_requires_paid_api_authorization() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 21600"):
        RehearsalR3Spec(
            rehearsal_id="r3",
            run_id="run",
            source_commit="1234567",
            repository=".",
            data_dir="data",
            artifact_root="runs",
            provider_mode="codex_cli",
            wall_seconds=21_601,
        )
    with pytest.raises(ValidationError, match="paid-API authorization"):
        RehearsalR3Spec(
            rehearsal_id="r3",
            run_id="run",
            source_commit="1234567",
            repository=".",
            data_dir="data",
            artifact_root="runs",
            provider_mode="openai_api",
        )


def test_submission_requires_explicit_test_prediction_authorization() -> None:
    with pytest.raises(ValidationError, match="explicit test-prediction authorization"):
        FinalSubmissionSpec(
            job_id="submission-1",
            source_run_id="run-1",
            best_valid_manifest_path="best.json",
            best_valid_manifest_sha256=SHA,
            test_feature_path="test.npz",
            test_feature_sha256=SHA,
            data_dir="data",
            output_dir="final",
        )


def test_checker_contract_prohibits_scoring() -> None:
    common = {
        "ordinal": 1,
        "checker_sha256": SHA,
        "csv_sha256": SHA,
        "stdout_sha256": SHA,
        "stderr_sha256": SHA,
        "returncode": 0,
        "valid": True,
    }
    with pytest.raises(ValidationError, match="score/make"):
        SubmissionCheckEvidence(command=["submit.py", "--check", "--score"], **common)
    with pytest.raises(ValidationError, match="exactly once"):
        SubmissionCheckEvidence(command=["submit.py", "--split", "test"], **common)

