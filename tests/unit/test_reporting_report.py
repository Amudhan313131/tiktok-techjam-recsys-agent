from __future__ import annotations

import json
from pathlib import Path

import pytest

from rex.contracts import AttemptStatus, ExperimentProposal, Metrics, Operator, RunState
from rex.control.budget import deadline_epoch_ms
from rex.data.manifest import sha256_file
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
    repository.transition_run(run_id, RunState.INITIALIZING, RunState.BASELINE_VERIFYING)

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
    assert unrelated_ref.artifact_id not in {item["artifact_id"] for item in evidence["artifacts"]}


def test_report_is_complete_self_contained_and_idempotent(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.sqlite3")
    database.initialize()
    repository = ExperimentRepository(database)
    run_id = "judge-report"
    repository.create_run(
        run_id=run_id,
        deadline_epoch_ms=deadline_epoch_ms(100),
        root_commit="root",
        environment_sha256=HASH,
        data_manifest_sha256=HASH,
        evaluator_sha256=HASH,
    )
    repository.transition_run(run_id, RunState.INITIALIZING, RunState.BASELINE_VERIFYING)
    baseline_model = tmp_path / "baseline-model.json"
    baseline_model.write_text('{"model":"fm"}\n', encoding="utf-8")
    baseline_ref = artifact_ref(baseline_model, "model_bundle")
    repository.register_artifact(baseline_ref)
    environment_path = tmp_path / "environment-identity.json"
    environment_path.write_text(
        json.dumps(
            {
                "runtime_kind": "docker",
                "worker_image_digest": "sha256:" + "9" * 64,
                "container_platform": "linux/arm64",
            }
        ),
        encoding="utf-8",
    )
    environment_ref = artifact_ref(environment_path, "runtime_environment_identity")
    repository.register_artifact(environment_ref)
    repository.establish_baseline(
        run_id=run_id,
        metrics=Metrics(
            GAUC=0.6674,
            **{"nDCG@5": 0.5366},
            primary=0.602,
            users=10,
            rows=20,
            evaluator_sha256=HASH,
            split="valid",
        ),
        evidence_artifact_ids=[baseline_ref.artifact_id, environment_ref.artifact_id],
    )

    experiment_id = "candidate-001"
    repository.create_experiment(
        run_id,
        ExperimentProposal(
            experiment_id=experiment_id,
            parent_id=None,
            operator=Operator.LOSS,
            hypothesis="Pairwise weighting should improve top-five ranking quality.",
            mechanism="Weighting informative pairs should focus learning on ranking errors.",
            primary_change="pair weighting",
            files_to_change=["configs/experiments/example.yaml"],
            expected_metric_effects={"nDCG@5": "increase"},
            falsifier="Validation primary does not improve.",
            leakage_analysis="Uses training targets only.",
            estimated_seconds=10,
            cheap_rung={"fold": "A"},
            full_rung={"folds": ["A", "B", "C"]},
        ),
        "root",
    )
    patch_path = tmp_path / "candidate.diff"
    patch_text = "--- a/configs/experiments/example.yaml\n+++ b/configs/experiments/example.yaml\n@@ -1 +1 @@\n-loss: bce\n+loss: pairwise\n"
    patch_path.write_text(patch_text, encoding="utf-8")
    patch_ref = artifact_ref(patch_path, "patch")
    repository.register_artifact(patch_ref, experiment_id=experiment_id)
    checkpoint_path = tmp_path / "candidate.bin"
    checkpoint_path.write_bytes(b"candidate-checkpoint")
    checkpoint_ref = artifact_ref(checkpoint_path, "checkpoint")
    repository.register_artifact(checkpoint_ref, experiment_id=experiment_id)
    repository.record_metrics(
        experiment_id,
        Metrics(
            GAUC=0.671,
            **{"nDCG@5": 0.549},
            primary=0.61,
            users=10,
            rows=20,
            evaluator_sha256=HASH,
            split="valid",
        ),
    )
    repair = repository.reserve_experiment_repair(
        experiment_id=experiment_id,
        phase="cheap",
        failure_status=AttemptStatus.TIMEOUT,
        plan={"action": "retry_with_smaller_batch"},
    )
    repository.complete_experiment_repair(
        str(repair["repair_id"]), evidence_artifact_ids=[patch_ref.artifact_id]
    )
    repository.record_llm_call(
        call_id="candidate-llm",
        run_id=run_id,
        experiment_id=experiment_id,
        role="patch",
        provider="test",
        model="test",
        request_artifact_id=None,
        response_artifact_id=None,
        schema_valid=False,
        input_tokens=12,
        output_tokens=3,
        wall_seconds=0.5,
        error="interrupted once",
    )
    repository.record_resource_usage(
        resource_key="candidate-gpu",
        run_id=run_id,
        experiment_id=experiment_id,
        scope="training",
        wall_seconds=20,
        gpu_seconds=7200,
    )
    repository.record_intervention(
        intervention_id="controlled-fault",
        run_id=run_id,
        actor="fixture_rehearsal",
        action="controlled_sigkill",
        reason="prove recovery",
    )
    repository.record_intervention(
        intervention_id="human-help",
        run_id=run_id,
        actor="human",
        action="restart_docker",
        reason="daemon was unavailable",
    )
    with database.transaction() as connection:
        connection.execute(
            "UPDATE runs SET search_champion_experiment_id=?,best_ever_experiment_id=?,"
            "best_primary_units=?,state='COMPLETE',stop_reason='epsilon_plateau' WHERE run_id=?",
            (experiment_id, experiment_id, 610000, run_id),
        )
        connection.execute(
            "INSERT INTO attempts(attempt_id,experiment_id,rung,repair_number,status,"
            "wall_seconds,error_type,error_summary) VALUES(?,?,?,?,?,?,?,?)",
            (
                "candidate-timeout",
                experiment_id,
                "cheap",
                0,
                "timeout",
                10.0,
                "timeout",
                "worker exceeded its limit",
            ),
        )
        connection.execute(
            "INSERT INTO process_sessions(session_id,run_id,pid,host,started_at,ended_at,"
            "monotonic_seconds,last_heartbeat,exit_reason) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                "session-1",
                run_id,
                123,
                "host",
                "2026-08-30T00:00:00+00:00",
                "2026-08-30T00:01:00+00:00",
                60.0,
                "2026-08-30T00:01:00+00:00",
                "dead_process_takeover",
            ),
        )

    output = tmp_path / "report"
    first = build_report(database, run_id, output)
    first_hashes = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in output.iterdir()
        if path.is_file()
    }
    second = build_report(database, run_id, output)
    second_hashes = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in output.iterdir()
        if path.is_file()
    }
    assert first_hashes == second_hashes
    assert first["resources"] == second["resources"]

    iteration = json.loads((output / "iteration_logs.json").read_text(encoding="utf-8"))[0]
    assert iteration["scientific_intent"]["why"].startswith("Weighting informative")
    assert iteration["applied_change"]["embedded_diffs"][0]["diff"] == patch_text
    assert iteration["resulting_metrics"][0]["gauc"] == 0.671
    assert {event["kind"] for event in iteration["error_recovery_events"]} == {
        "attempt_failure",
        "bounded_repair",
        "llm_error",
    }

    resources = json.loads((output / "resources.json").read_text(encoding="utf-8"))
    assert resources["llm_input_tokens"] == 12
    assert resources["llm_output_tokens"] == 3
    assert resources["llm_total_tokens"] == 15
    assert resources["gpu_hours"] == 2.0
    assert resources["iterations_used"] == 1
    assert resources["iteration_cap"] == 50
    assert resources["manual_interventions"] == 1

    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert results["validation_best_experiment_id"] == experiment_id
    assert results["validation_best"] == {
        "GAUC": 0.671,
        "nDCG@5": 0.549,
        "primary": 0.61,
        "split": "valid",
        "fold": None,
        "seed": None,
        "evaluator_sha256": HASH,
    }
    assert results["delta_over_official_baseline"]["primary"] == pytest.approx(0.0084)
    assert results["hidden_test_scored_locally"] is False

    intervention = json.loads((output / "manual_interventions.json").read_text(encoding="utf-8"))
    assert intervention["manual_intervention_count"] == 1
    assert len(intervention["automated_control_events_excluded"]) == 1
    artifact_summary = json.loads((output / "artifact_summary.json").read_text(encoding="utf-8"))
    assert artifact_summary["checkpoint_artifacts"][0]["sha256"] == checkpoint_ref.sha256
    assert artifact_summary["final_submission"]["test_scored_locally"] is False
    environment = json.loads((output / "environment_identity.json").read_text(encoding="utf-8"))
    assert environment["worker_image_digest"] == "sha256:" + "9" * 64
    assert (
        artifact_summary["runtime_environment_identity"]["source_sha256"] == environment_ref.sha256
    )
    markdown = (output / "experiments.md").read_text(encoding="utf-8")
    assert "```diff" in markdown
    assert "12 input + 3 output" in markdown
    assert "dead_process_takeover" in markdown

    with database.transaction() as connection:
        connection.execute(
            "INSERT INTO metrics(experiment_id,split,evaluator_sha256,gauc,ndcg5,"
            "primary_score,primary_units,rows,users) VALUES(?,?,?,?,?,?,?,?,?)",
            (experiment_id, "test", HASH, 0.9, 0.9, 0.9, 900000, 20, 10),
        )
    with pytest.raises(RuntimeError, match="hidden-test metrics"):
        build_report(database, run_id, output)
