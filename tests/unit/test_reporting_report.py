from __future__ import annotations

import json
from pathlib import Path

from rex.contracts import AttemptStatus, ExperimentProposal, Metrics, Operator, RunState
from rex.control.budget import deadline_epoch_ms
from rex.execution.artifacts import artifact_ref
from rex.reporting.report import build_report
from rex.store.db import Database
from rex.store.repository import ExperimentRepository


HASH = "0" * 64


def test_report_includes_run_level_baseline_and_pre_experiment_llm_evidence(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "state.sqlite3")
    database.initialize()
    repository = ExperimentRepository(database)
    run_id = "run-level-evidence"
    repository.create_run(
        run_id=run_id,
        deadline_epoch_ms=deadline_epoch_ms(100),
        root_commit="root",
        environment_sha256=HASH,
        data_manifest_sha256=HASH,
        evaluator_sha256=HASH,
    )
    repository.transition_run(
        run_id, RunState.INITIALIZING, RunState.BASELINE_VERIFYING
    )

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text('{"accepted":true}\n', encoding="utf-8")
    baseline_ref = artifact_ref(baseline_path, "baseline_gate")
    repository.register_artifact(baseline_ref)
    repository.establish_baseline(
        run_id=run_id,
        metrics=Metrics(
            GAUC=0.6,
            **{"nDCG@5": 0.5},
            primary=0.55,
            users=10,
            rows=20,
            evaluator_sha256=HASH,
            split="valid",
        ),
        evidence_artifact_ids=[baseline_ref.artifact_id],
    )

    failure_path = tmp_path / "llm-failure.json"
    failure_path.write_text('{"error":"timeout"}\n', encoding="utf-8")
    failure_ref = artifact_ref(failure_path, "llm_failure")
    repository.register_artifact(failure_ref)
    repository.record_llm_call(
        call_id="proposal-failure",
        run_id=run_id,
        experiment_id=None,
        role="proposal_or_patch",
        provider="test",
        model="test",
        request_artifact_id=None,
        response_artifact_id=failure_ref.artifact_id,
        schema_valid=False,
        input_tokens=0,
        output_tokens=0,
        wall_seconds=0.1,
        error="timeout",
    )

    unrelated_path = tmp_path / "unrelated.json"
    unrelated_path.write_text("{}\n", encoding="utf-8")
    unrelated_ref = artifact_ref(unrelated_path, "unrelated")
    repository.register_artifact(unrelated_ref)

    experiment_id = "experiment-with-control-evidence"
    repository.create_experiment(
        run_id,
        ExperimentProposal(
            experiment_id=experiment_id,
            parent_id=None,
            operator=Operator.LOSS,
            hypothesis="The treatment should improve ranking.",
            mechanism="A controlled test mechanism.",
            primary_change="one loss setting",
            files_to_change=["configs/experiments/example.yaml"],
            expected_metric_effects={"primary": "increase"},
            falsifier="No improvement on the cheap fold.",
            leakage_analysis="Train labels only.",
            estimated_seconds=10,
            cheap_rung={"fold": "A"},
            full_rung={"folds": ["A", "B", "C"]},
        ),
        "root",
    )
    repository.reserve_experiment_repair(
        experiment_id=experiment_id,
        phase="cheap",
        failure_status=AttemptStatus.TIMEOUT,
        plan={"action": "retry"},
    )
    with database.transaction() as connection:
        connection.execute(
            "INSERT INTO search_promotions(run_id,previous_experiment_id,experiment_id,"
            "primary_units,evidence_json,created_at) VALUES(?,?,?,?,?,?)",
            (run_id, "baseline", experiment_id, 550000, "[]", "2026-08-30T00:00:00+00:00"),
        )
        connection.execute(
            "INSERT INTO convergence_transactions(experiment_id,run_id,outcome,delta_units,"
            "created_at) VALUES(?,?,?,?,?)",
            (experiment_id, run_id, "promoted", 1000, "2026-08-30T00:00:00+00:00"),
        )

    output = tmp_path / "report"
    build_report(database, run_id, output)
    evidence = json.loads((output / "evidence_index.json").read_text(encoding="utf-8"))

    assert evidence["baseline_gates"][0]["run_id"] == run_id
    assert {item["artifact_id"] for item in evidence["artifacts"]} == {
        baseline_ref.artifact_id,
        failure_ref.artifact_id,
    }
    assert {item["artifact_id"] for item in evidence["artifact_links"]} == {
        baseline_ref.artifact_id,
        failure_ref.artifact_id,
    }
    assert evidence["llm_calls"][0]["experiment_id"] is None
    assert evidence["experiment_repairs"][0]["experiment_id"] == experiment_id
    assert evidence["search_promotions"][0]["experiment_id"] == experiment_id
    assert evidence["convergence_transactions"][0]["experiment_id"] == experiment_id
    assert unrelated_ref.artifact_id not in {
        item["artifact_id"] for item in evidence["artifacts"]
    }
