from __future__ import annotations

from pathlib import Path

from rex.rehearsal import rehearsal_requirements, run_production_rehearsal, run_r1, run_r2


def _case(result: dict[str, object], name: str) -> dict[str, object]:
    return next(item for item in result["cases"] if item["name"] == name)


def test_r1_connected_fault_matrix_recovers_without_replacing_incumbent_unsafely(
    tmp_path: Path,
) -> None:
    result = run_r1(None, tmp_path / "r1")

    assert result["all_cases_passed"] is True
    assert result["case_count"] == 10
    assert result["max_repair_number_observed"] == 2
    assert result["repair_limit_respected"] is True
    assert result["real_data_accessed"] is False
    assert result["six_hour_dress_rehearsal"] is False
    assert result["test_prediction_created"] is False
    assert result["final_submission_created"] is False

    preparation = _case(result, "preparation_interrupt")
    assert preparation["interruption_recovered"] is True
    assert preparation["interrupted_snapshot"]["state"] == "SEARCHING"
    assert preparation["interrupted_snapshot"]["champion"] == "baseline"

    timeout = _case(result, "training_timeout_descendants")
    assert timeout["repair_actions"] == [
        {"phase": "cheap", "repair_number": 1, "action": "reduce_workload"}
    ]

    stale = _case(result, "database_interrupt_stale_takeover")
    assert stale["interruption_recovered"] is True
    assert stale["stale_takeover_recorded"] is True
    assert stale["transaction_rollback_preserved_incumbent"] is True

    finalization = _case(result, "finalization_interrupt")
    assert finalization["interrupted_snapshot"]["state"] == "FINALIZING"
    assert finalization["interrupted_snapshot"]["champion"] == "rehearsal-e01"

    persistent = _case(result, "persistent_candidate_then_continue")
    assert persistent["max_repair_number"] == 2
    assert persistent["snapshot"]["experiments"] == [
        {"experiment_id": "rehearsal-e01", "state": "FAILED_FINAL"},
        {"experiment_id": "rehearsal-e02", "state": "PROMOTED"},
    ]
    assert persistent["snapshot"]["promotions"] == [
        {"previous_experiment_id": "baseline", "experiment_id": "rehearsal-e02"}
    ]


def test_r2_recovers_provider_faults_and_rejects_authority_escalation(
    tmp_path: Path,
) -> None:
    result = run_r2(None, tmp_path / "r2")

    assert result["all_cases_passed"] is True
    assert result["r1_passed"] is True
    assert result["provider_timeout_injected_once"] is True
    assert result["provider_schema_failure_injected_once"] is True
    assert result["final_submission_created"] is False

    timeout = _case(result, "llm_interrupt")
    schema = _case(result, "llm_schema_failure")
    assert timeout["interruption_recovered"] is True
    assert schema["interruption_recovered"] is True
    assert timeout["interrupted_snapshot"]["state"] == "SEARCHING"
    assert schema["interrupted_snapshot"]["experiments"] == [
        {"experiment_id": "rehearsal-e01", "state": "FULL_COMPLETE"}
    ]

    security = _case(result, "protected_patch_and_workspace_escape")
    assert security["acceptance"] == {
        "protected_patch_rejected": True,
        "patch_path_traversal_rejected": True,
        "sandbox_workspace_escape_rejected": True,
    }


def test_production_rehearsal_dispatch_and_scope(tmp_path: Path) -> None:
    result = run_production_rehearsal("R1", tmp_path / "dispatch")
    requirements = rehearsal_requirements("R2")

    assert result["level"] == "R1"
    assert result["all_cases_passed"] is True
    assert requirements["data"] is False
    assert requirements["live_llm"] is False
    assert requirements["provider_faults"] is True
    assert requirements["final_submission"] is False
