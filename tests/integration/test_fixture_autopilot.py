from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from rex.contracts import ExperimentProposal, RunState
from rex.control.budget import deadline_epoch_ms
from rex.control.fixture_supervisor import (
    FixtureAutopilot,
    FixtureRunConfig,
    FixtureScriptProvider,
    _create_disposable_source,
    _write_fixture_views,
)
from rex.data.manifest import repo_root, sha256_file
from rex.rehearsal import run_fixture_rehearsal
from rex.store.db import Database
from rex.store.repository import ExperimentRepository


HASH = "0" * 64


def _searching_fixture_run(config: FixtureRunConfig, run_id: str):
    run_dir = config.runs_dir / run_id
    run_dir.mkdir(parents=True)
    database = Database(run_dir / "state.sqlite3")
    database.initialize()
    repository = ExperimentRepository(database)
    _, root_commit = _create_disposable_source(run_dir)
    features, _ = _write_fixture_views(run_dir)
    repository.create_run(
        run_id=run_id,
        deadline_epoch_ms=deadline_epoch_ms(900),
        root_commit=root_commit,
        environment_sha256=HASH,
        data_manifest_sha256=sha256_file(features),
        evaluator_sha256=HASH,
    )
    repository.transition_run(run_id, RunState.INITIALIZING, RunState.BASELINE_VERIFYING)
    repository.transition_run(run_id, RunState.BASELINE_VERIFYING, RunState.SEARCHING)
    return repository


def test_connected_fixture_rehearsal_recovers_and_never_promotes(tmp_path: Path) -> None:
    result = run_fixture_rehearsal(
        repo_root() / "configs/run/fixture.yaml",
        tmp_path / "runs",
    )

    assert result["state"] == "COMPLETE"
    assert result["stop_reason"] == "fixture_hypothesis_cap"
    assert result["hypothesis_count"] == 4
    assert result["non_improvement_streak"] == 1
    assert result["event_chain_valid"]
    assert result["provider_interruption_recovered"]
    assert result["worker_nan_recovered"]
    assert result["worker_repair_limit_enforced"]
    assert result["protected_patch_rejected"]
    assert result["production_promotion_blocked"]
    assert not result["production_science_enabled"]
    assert not result["confirmation_enabled"]
    assert not result["final_submission_enabled"]
    assert [item["state"] for item in result["experiments"]] == [
        "ABANDONED",
        "REJECTED",
        "ABANDONED",
        "FAILED_FINAL",
    ]
    report_dir = Path(str(result["run_dir"])) / "report"
    event_types = {
        json.loads(line)["event_type"]
        for line in (report_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert "run.transition" in event_types
    assert "session.closed" in event_types
    evidence = json.loads((report_dir / "evidence_index.json").read_text(encoding="utf-8"))
    assert evidence["runs"][0]["state"] == "COMPLETE"
    assert evidence["process_sessions"][0]["exit_reason"] == "fixture_complete"
    assert evidence["artifact_links"]
    assert evidence["llm_calls"]
    assert {item["action"] for item in evidence["interventions"]} == {
        "controlled_provider_interruption",
        "protected_patch_probe",
        "controlled_worker_nan",
        "controlled_worker_repair_exhausted",
    }


def test_resume_closes_precommit_candidate_and_continues(tmp_path: Path) -> None:
    config = replace(
        FixtureRunConfig.load(repo_root() / "configs/run/fixture.yaml"),
        runs_dir=tmp_path / "runs",
    )
    run_id = "resume-precommit"
    repository = _searching_fixture_run(config, run_id)
    scripted = FixtureScriptProvider()
    proposal = scripted.generate(
        role="proposal",
        system="",
        prompt=json.dumps({"fixture_iteration_number": 1}),
        schema={},
    )
    repository.create_experiment(
        run_id,
        ExperimentProposal.model_validate(proposal.value),
        repository.get_run(run_id)["root_commit"],
        max_hypotheses=3,
        experiment_kind="fixture",
    )

    result = FixtureAutopilot(config, FixtureScriptProvider()).run(run_id=run_id)

    assert result["state"] == "COMPLETE"
    assert [item["state"] for item in result["experiments"]] == [
        "ABANDONED",
        "REJECTED",
        "ABANDONED",
    ]


def test_resume_finishes_a_run_already_in_finalizing(tmp_path: Path) -> None:
    config = replace(
        FixtureRunConfig.load(repo_root() / "configs/run/fixture.yaml"),
        runs_dir=tmp_path / "runs",
    )
    run_id = "resume-finalizing"
    repository = _searching_fixture_run(config, run_id)
    repository.transition_run(run_id, RunState.SEARCHING, RunState.FINALIZING, "controlled-kill")

    result = FixtureAutopilot(config, FixtureScriptProvider()).run(run_id=run_id)

    assert result["state"] == "COMPLETE"
    assert result["stop_reason"] == "controlled-kill"
    assert result["hypothesis_count"] == 0
