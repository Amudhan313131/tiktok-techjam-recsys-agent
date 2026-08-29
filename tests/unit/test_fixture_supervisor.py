from __future__ import annotations

import json
from pathlib import Path

import pytest

from rex.contracts import ExperimentState
from rex.control.fixture_supervisor import (
    FixtureRunConfig,
    FixtureScriptProvider,
    next_fixture_action,
)


def test_fixture_dispatcher_covers_running_and_terminal_states() -> None:
    assert next_fixture_action(ExperimentState.FIXTURE_VALID) == "start_cheap"
    assert next_fixture_action(ExperimentState.CHEAP_RUNNING) == "resume_cheap"
    assert next_fixture_action(ExperimentState.FULL_RUNNING) == "resume_full"
    assert next_fixture_action(ExperimentState.FAILED_REPAIRABLE) == "repair_or_fail"
    assert next_fixture_action(ExperimentState.REJECTED) == "terminal"


def test_production_configuration_is_rejected_by_fixture_loader(tmp_path: Path) -> None:
    path = tmp_path / "production.yaml"
    path.write_text(
        "execution_mode: production_science_disabled\n"
        "budget_config: configs/budget.yaml\n"
        "protected_paths: configs/security/protected_paths.yaml\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="production science is disabled"):
        FixtureRunConfig.load(path)


def test_fixed_fixture_provider_is_role_aware_and_evidence_bound() -> None:
    provider = FixtureScriptProvider()
    proposal = provider.generate(
        role="proposal",
        system="",
        prompt=json.dumps({"fixture_iteration_number": 2}),
        schema={},
    )
    patch = provider.generate(
        role="patch",
        system="",
        prompt=json.dumps({"proposal": proposal.value}),
        schema={},
    )
    diagnosis = provider.generate(
        role="diagnosis",
        system="",
        prompt=json.dumps(
            {
                "experiment_id": proposal.value["experiment_id"],
                "artifact_ids": ["fixture-evidence"],
                "fixture_primary": 0.51,
            }
        ),
        schema={},
    )
    assert proposal.provider == "fixed"
    assert proposal.value["experiment_id"] == "fixture-002"
    assert "DEFAULT_BIAS" in patch.value["patch"]
    assert diagnosis.value["evidence_artifact_ids"] == ["fixture-evidence"]
