from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rex.contracts import (
    AttemptStatus,
    ExperimentProposal,
    ExperimentState,
    Metrics,
    Operator,
    RunResult,
    RunState,
    ValidationPhase,
)
from rex.control.budget import deadline_epoch_ms
from rex.execution.artifacts import artifact_ref
from rex.store.db import Database
from rex.store.event_log import export_events, verify_event_chain
from rex.store.repository import ExperimentRepository, RepositoryError


HASH = "0" * 64


def proposal(experiment_id: str) -> ExperimentProposal:
    return ExperimentProposal(
        experiment_id=experiment_id,
        parent_id=None,
        operator=Operator.LOSS,
        hypothesis="Pairwise ranking should improve within-user impression ordering.",
        mechanism="Same-user pairs align gradients with the official ranking metric.",
        primary_change="pairwise loss",
        files_to_change=["src/rex/losses/experimental/pair.py"],
        expected_metric_effects={"primary": "increase"},
        falsifier="Primary delta below one thousandth on the cheap fold.",
        leakage_analysis="Train labels only and complete user groups.",
        estimated_seconds=30,
        cheap_rung={"fold": "A"},
        full_rung={"folds": ["A", "B", "C"]},
    )


def repository(tmp_path: Path) -> tuple[ExperimentRepository, Database, str]:
    database = Database(tmp_path / "state.sqlite3")
    database.initialize()
    repo = ExperimentRepository(database)
    run_id = "run"
    repo.create_run(
        run_id=run_id,
        deadline_epoch_ms=deadline_epoch_ms(100),
        root_commit="root",
        environment_sha256=HASH,
        data_manifest_sha256=HASH,
        evaluator_sha256=HASH,
    )
    return repo, database, run_id


def test_transition_replay_is_exactly_once(tmp_path: Path) -> None:
    repo, _, run_id = repository(tmp_path)
    repo.create_experiment(run_id, proposal("e1"), "root")
    repo.transition_experiment(
        "e1", ExperimentState.PROPOSED, ExperimentState.WORKTREE_READY, idempotency_key="once"
    )
    repo.transition_experiment(
        "e1", ExperimentState.PROPOSED, ExperimentState.WORKTREE_READY, idempotency_key="once"
    )
    assert repo.get_experiment("e1")["state"] == ExperimentState.WORKTREE_READY


def test_event_export_detects_tampering(tmp_path: Path) -> None:
    repo, database, run_id = repository(tmp_path)
    path = tmp_path / "events.jsonl"
    export_events(database, run_id, path)
    assert verify_event_chain(path)
    content = path.read_text(encoding="utf-8").replace("run.created", "run.changed")
    path.write_text(content, encoding="utf-8")
    assert not verify_event_chain(path)


def test_run_state_compare_and_swap(tmp_path: Path) -> None:
    repo, _, run_id = repository(tmp_path)
    repo.transition_run(run_id, RunState.INITIALIZING, RunState.BASELINE_VERIFYING)
    assert repo.get_run(run_id)["state"] == RunState.BASELINE_VERIFYING


def test_migration_backfills_legacy_artifact_links_with_original_path(tmp_path: Path) -> None:
    repo, database, run_id = repository(tmp_path)
    repo.create_experiment(run_id, proposal("legacy-artifact"), "root")
    payload = tmp_path / "legacy.bin"
    payload.write_bytes(b"legacy")
    ref = artifact_ref(payload, "fixture")
    repo.register_artifact(ref, experiment_id="legacy-artifact")
    with database.connect() as connection:
        connection.execute("DROP TABLE artifact_links")
        connection.execute("PRAGMA user_version = 1")

    database.initialize()

    with database.connect() as connection:
        row = connection.execute(
            "SELECT artifact_id,artifact_path FROM artifact_links WHERE artifact_id=?",
            (ref.artifact_id,),
        ).fetchone()
    assert row is not None
    assert row["artifact_path"] == str(payload)


def test_promotion_requires_all_linked_artifacts(tmp_path: Path) -> None:
    repo, _, run_id = repository(tmp_path)
    repo.create_experiment(run_id, proposal("e1"), "root")
    chain = [
        ExperimentState.WORKTREE_READY,
        ExperimentState.PATCHED,
        ExperimentState.STATIC_VALID,
        ExperimentState.FIXTURE_VALID,
        ExperimentState.CHEAP_RUNNING,
        ExperimentState.CHEAP_COMPLETE,
        ExperimentState.FULL_RESERVED,
        ExperimentState.FULL_RUNNING,
        ExperimentState.FULL_COMPLETE,
        ExperimentState.DIAGNOSED,
        ExperimentState.CONFIRMING,
        ExperimentState.CONFIRMED,
        ExperimentState.SUBMISSION_BUILDING,
        ExperimentState.SUBMISSION_VALID,
    ]
    current = ExperimentState.PROPOSED
    for index, target in enumerate(chain):
        repo.transition_experiment("e1", current, target, idempotency_key=f"t{index}")
        current = target
    refs = []
    for kind in ("checkpoint", "predictions", "submission", "validator"):
        path = tmp_path / kind
        path.write_text(kind, encoding="utf-8")
        ref = artifact_ref(path, kind)
        repo.register_artifact(ref, experiment_id="e1")
        refs.append(ref)
    result = repo.promote_validated(
        run_id=run_id,
        experiment_id="e1",
        primary=0.601,
        epsilon=0.002,
        patience=3,
        checkpoint_artifact_id=refs[0].artifact_id,
        prediction_artifact_id=refs[1].artifact_id,
        submission_artifact_id=refs[2].artifact_id,
        validator_artifact_id=refs[3].artifact_id,
        idempotency_key="promote",
    )
    assert result["is_new_best"]
    assert repo.get_experiment("e1")["state"] == ExperimentState.PROMOTED


def test_three_cheap_rejections_trigger_convergence(tmp_path: Path) -> None:
    repo, _, run_id = repository(tmp_path)
    for number in range(3):
        experiment_id = f"reject-{number}"
        repo.create_experiment(run_id, proposal(experiment_id), "root")
        current = ExperimentState.PROPOSED
        for index, target in enumerate(
            (
                ExperimentState.WORKTREE_READY,
                ExperimentState.PATCHED,
                ExperimentState.STATIC_VALID,
                ExperimentState.FIXTURE_VALID,
                ExperimentState.CHEAP_RUNNING,
                ExperimentState.CHEAP_COMPLETE,
            )
        ):
            repo.transition_experiment(
                experiment_id, current, target, idempotency_key=f"{experiment_id}:{index}"
            )
            current = target
        result = repo.reject_candidate(
            run_id=run_id,
            experiment_id=experiment_id,
            expected_state=ExperimentState.CHEAP_COMPLETE,
            reason="cheap delta below promotion threshold",
            patience=3,
            idempotency_key=f"{experiment_id}:reject",
        )
    assert result["converged"] is True
    assert repo.get_run(run_id)["non_improvement_streak"] == 3
    assert repo.get_run(run_id)["stop_reason"] == "epsilon_plateau"


def test_failed_transaction_counts_once_and_preserves_baseline_incumbent(tmp_path: Path) -> None:
    repo, _, run_id = repository(tmp_path)
    repo.transition_run(run_id, RunState.INITIALIZING, RunState.BASELINE_VERIFYING)
    repo.establish_baseline(
        run_id=run_id,
        metrics=Metrics(
            GAUC=0.5,
            **{"nDCG@5": 0.5},
            primary=0.5,
            users=2,
            rows=4,
            evaluator_sha256=HASH,
            split="valid",
        ),
        evidence_artifact_ids=[],
    )
    repo.transition_run(run_id, RunState.BASELINE_VERIFYING, RunState.SEARCHING)
    repo.create_experiment(run_id, proposal("failed"), "root")
    current = ExperimentState.PROPOSED
    for index, target in enumerate(
        (
            ExperimentState.WORKTREE_READY,
            ExperimentState.PATCHED,
            ExperimentState.STATIC_VALID,
            ExperimentState.FIXTURE_VALID,
            ExperimentState.CHEAP_RUNNING,
            ExperimentState.FAILED_FINAL,
        )
    ):
        repo.transition_experiment("failed", current, target, idempotency_key=f"failed:{index}")
        current = target

    first = repo.count_failed_transaction(
        run_id=run_id,
        experiment_id="failed",
        reason="controlled failure",
        patience=3,
    )
    second = repo.count_failed_transaction(
        run_id=run_id,
        experiment_id="failed",
        reason="controlled failure",
        patience=3,
    )

    run = repo.get_run(run_id)
    assert first["idempotent"] is False
    assert second["idempotent"] is True
    assert run["non_improvement_streak"] == 0
    assert first["converged"] is False
    assert run["best_primary_units"] == 500_000_000
    assert run["best_ever_experiment_id"] == "baseline"
    assert run["search_champion_experiment_id"] == "baseline"


def test_atomic_validation_phase_locks_one_finalist_and_never_reopens(tmp_path: Path) -> None:
    repo, _, run_id = repository(tmp_path)
    repo.transition_run(run_id, RunState.INITIALIZING, RunState.BASELINE_VERIFYING)
    repo.establish_baseline(
        run_id=run_id,
        metrics=Metrics(
            GAUC=0.5,
            **{"nDCG@5": 0.5},
            primary=0.5,
            users=2,
            rows=4,
            evaluator_sha256=HASH,
            split="valid",
        ),
        evidence_artifact_ids=[],
    )
    repo.transition_run(run_id, RunState.BASELINE_VERIFYING, RunState.SEARCHING)
    repo.create_experiment(
        run_id,
        proposal("finalist"),
        "root",
        method_card_id="E21",
        experiment_kind="production_search",
    )
    current = ExperimentState.PROPOSED
    for index, target in enumerate(
        (
            ExperimentState.WORKTREE_READY,
            ExperimentState.PATCHED,
            ExperimentState.STATIC_VALID,
            ExperimentState.FIXTURE_VALID,
            ExperimentState.CHEAP_RUNNING,
            ExperimentState.CHEAP_COMPLETE,
            ExperimentState.FULL_RESERVED,
            ExperimentState.FULL_RUNNING,
            ExperimentState.FULL_COMPLETE,
            ExperimentState.DIAGNOSED,
        )
    ):
        repo.transition_experiment("finalist", current, target, idempotency_key=f"phase:{index}")
        current = target
    evidence_path = tmp_path / "full-shadow.json"
    evidence_path.write_text("{}\n", encoding="utf-8")
    evidence = artifact_ref(evidence_path, "production_full_result")
    repo.register_artifact(evidence, experiment_id="finalist")
    repo.record_shadow_evaluation(
        run_id=run_id,
        experiment_id="finalist",
        family="field_weighted_fm",
        primary=0.505,
        supported=True,
        evidence_artifact_ids=[evidence.artifact_id],
        epsilon=0.002,
        patience=3,
        valid_family_count=1,
        minimum_valid_families=3,
    )

    first = repo.lock_validation_finalist(
        run_id=run_id,
        experiment_id="finalist",
        evidence_artifact_ids=[evidence.artifact_id],
        idempotency_key="lock-finalist",
    )
    replay = repo.lock_validation_finalist(
        run_id=run_id,
        experiment_id="finalist",
        evidence_artifact_ids=[evidence.artifact_id],
        idempotency_key="lock-finalist",
    )
    assert first["idempotent"] is False
    assert replay["idempotent"] is True
    assert repo.get_run(run_id)["validation_phase"] == ValidationPhase.FINALIST_LOCKED

    repo.transition_experiment(
        "finalist",
        ExperimentState.DIAGNOSED,
        ExperimentState.OFFICIAL_VALID_RUNNING,
        idempotency_key="official-running",
    )
    repo.record_metrics(
        "finalist",
        Metrics(
            GAUC=0.51,
            **{"nDCG@5": 0.51},
            primary=0.51,
            users=2,
            rows=4,
            evaluator_sha256=HASH,
            split="valid",
        ),
        max_official_evaluations=1,
    )
    repo.transition_experiment(
        "finalist",
        ExperimentState.OFFICIAL_VALID_RUNNING,
        ExperimentState.OFFICIAL_VALID_COMPLETE,
        idempotency_key="official-complete",
    )
    repo.promote_search_candidate(
        run_id=run_id,
        experiment_id="finalist",
        primary=0.51,
        evidence_artifact_ids=[evidence.artifact_id],
        epsilon=0.002,
        patience=3,
        idempotency_key="official-result",
    )
    run = repo.get_run(run_id)
    assert run["validation_phase"] == ValidationPhase.OFFICIAL_EVALUATED
    assert run["official_evaluation_count"] == 1
    assert run["finalist_experiment_id"] == "finalist"
    with pytest.raises(RepositoryError, match="cannot lock"):
        repo.lock_validation_finalist(
            run_id=run_id,
            experiment_id="finalist",
            evidence_artifact_ids=[evidence.artifact_id],
            idempotency_key="reopen",
        )


def test_shadow_tracker_uses_epsilon_and_resets_on_meaningful_improvement(
    tmp_path: Path,
) -> None:
    repo, _, run_id = repository(tmp_path)

    def record(experiment_id: str, score: float, family: str) -> dict:
        repo.create_experiment(
            run_id,
            proposal(experiment_id),
            "root",
            method_card_id=experiment_id,
            experiment_kind="production_search",
        )
        current = ExperimentState.PROPOSED
        for index, target in enumerate(
            (
                ExperimentState.WORKTREE_READY,
                ExperimentState.PATCHED,
                ExperimentState.STATIC_VALID,
                ExperimentState.FIXTURE_VALID,
                ExperimentState.CHEAP_RUNNING,
                ExperimentState.CHEAP_COMPLETE,
                ExperimentState.FULL_RESERVED,
                ExperimentState.FULL_RUNNING,
                ExperimentState.FULL_COMPLETE,
                ExperimentState.DIAGNOSED,
            )
        ):
            repo.transition_experiment(
                experiment_id,
                current,
                target,
                idempotency_key=f"{experiment_id}:shadow:{index}",
            )
            current = target
        path = tmp_path / f"{experiment_id}.json"
        path.write_text("{}\n", encoding="utf-8")
        ref = artifact_ref(path, "production_full_result")
        repo.register_artifact(ref, experiment_id=experiment_id)
        return repo.record_shadow_evaluation(
            run_id=run_id,
            experiment_id=experiment_id,
            family=family,
            primary=score,
            supported=True,
            evidence_artifact_ids=[ref.artifact_id],
            epsilon=0.002,
            patience=3,
            valid_family_count=3,
            minimum_valid_families=3,
        )

    assert record("e1", 0.600, "fm")["non_improvement_streak"] == 0
    assert record("e2", 0.601, "tree")["non_improvement_streak"] == 1
    assert record("e3", 0.604, "history")["non_improvement_streak"] == 0
    run = repo.get_run(run_id)
    assert run["shadow_champion_experiment_id"] == "e3"
    assert run["shadow_best_primary_units"] == 604_000_000


def test_process_session_lease_heartbeat_stale_takeover_and_close(tmp_path: Path) -> None:
    repo, database, run_id = repository(tmp_path)
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert not repo.open_process_session(
        session_id="s1", run_id=run_id, pid=101, host="fixture", now=started
    )["idempotent"]
    assert repo.open_process_session(
        session_id="s1", run_id=run_id, pid=101, host="fixture", now=started
    )["idempotent"]
    repo.heartbeat_process_session("s1", 5.0, now=started + timedelta(seconds=5))
    with pytest.raises(RepositoryError, match="active process session"):
        repo.open_process_session(
            session_id="s2",
            run_id=run_id,
            pid=202,
            host="fixture",
            stale_after_seconds=10,
            now=started + timedelta(seconds=6),
        )
    assert (
        repo.list_stale_process_sessions(
            run_id, stale_after_seconds=10, now=started + timedelta(seconds=16)
        )[0]["session_id"]
        == "s1"
    )
    takeover = repo.open_process_session(
        session_id="s2",
        run_id=run_id,
        pid=202,
        host="fixture",
        stale_after_seconds=10,
        now=started + timedelta(seconds=16),
    )
    assert takeover["stale_session_ids"] == ["s1"]
    assert not repo.close_process_session("s2", exit_reason="complete", monotonic_seconds=20)[
        "idempotent"
    ]
    assert repo.close_process_session("s2", exit_reason="complete", monotonic_seconds=20)[
        "idempotent"
    ]
    with pytest.raises(RepositoryError, match="conflicting replay"):
        repo.close_process_session("s2", exit_reason="fatal", monotonic_seconds=20)
    with database.connect() as connection:
        assert (
            connection.execute(
                "SELECT exit_reason FROM process_sessions WHERE session_id='s1'"
            ).fetchone()["exit_reason"]
            == "stale_takeover"
        )


def _result(experiment_id: str = "e1", *, wall_seconds: float = 1.5) -> RunResult:
    return RunResult(
        run_id="run",
        experiment_id=experiment_id,
        attempt_id=f"attempt-{experiment_id}",
        status=AttemptStatus.SUCCESS,
        command_sha256=HASH,
        commit_sha="root",
        config_sha256=HASH,
        data_view_sha256=HASH,
        environment_sha256=HASH,
        wall_seconds=wall_seconds,
    )


def test_attempt_replay_is_exactly_once_and_conflicts_fail(tmp_path: Path) -> None:
    repo, database, run_id = repository(tmp_path)
    repo.create_experiment(run_id, proposal("e1"), "root")
    result = _result()
    repo.record_attempt(result, rung="fixture")
    repo.record_attempt(result, rung="fixture")
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM resource_usage").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM event_outbox WHERE event_type='attempt.completed'"
            ).fetchone()[0]
            == 1
        )
    with pytest.raises(RepositoryError, match="conflicting replay"):
        repo.record_attempt(_result(wall_seconds=2.0), rung="fixture")


def test_reserved_attempt_completes_once(tmp_path: Path) -> None:
    repo, database, run_id = repository(tmp_path)
    repo.create_experiment(run_id, proposal("e1"), "root")
    assert not repo.reserve_attempt(
        attempt_id="attempt-e1",
        experiment_id="e1",
        rung="fixture",
        command_sha256=HASH,
        commit_sha="root",
    )["idempotent"]
    assert repo.reserve_attempt(
        attempt_id="attempt-e1",
        experiment_id="e1",
        rung="fixture",
        command_sha256=HASH,
        commit_sha="root",
    )["idempotent"]
    repo.record_attempt(_result(), rung="fixture")
    repo.record_attempt(_result(), rung="fixture")
    with database.connect() as connection:
        assert (
            connection.execute(
                "SELECT status FROM attempts WHERE attempt_id='attempt-e1'"
            ).fetchone()["status"]
            == AttemptStatus.SUCCESS
        )
        assert connection.execute("SELECT COUNT(*) FROM resource_usage").fetchone()[0] == 1


def test_metric_and_llm_replays_do_not_double_count(tmp_path: Path) -> None:
    repo, database, run_id = repository(tmp_path)
    repo.create_experiment(run_id, proposal("e1"), "root")
    metric = Metrics(
        GAUC=0.6,
        **{"nDCG@5": 0.4},
        primary=0.5,
        users=10,
        rows=20,
        evaluator_sha256=HASH,
        split="valid",
        fold=None,
        seed=None,
    )
    repo.record_metrics("e1", metric)
    repo.record_metrics("e1", metric)
    repo.record_llm_call(
        call_id="call",
        run_id=run_id,
        experiment_id="e1",
        role="proposal",
        provider="fake",
        model="fixture",
        request_artifact_id=None,
        response_artifact_id=None,
        schema_valid=True,
        input_tokens=3,
        output_tokens=4,
        wall_seconds=0.5,
    )
    repo.record_llm_call(
        call_id="call",
        run_id=run_id,
        experiment_id="e1",
        role="proposal",
        provider="fake",
        model="fixture",
        request_artifact_id=None,
        response_artifact_id=None,
        schema_valid=True,
        input_tokens=3,
        output_tokens=4,
        wall_seconds=0.5,
    )
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 1
        assert repo.get_run(run_id)["official_evaluation_count"] == 1
        assert connection.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM resource_usage").fetchone()[0] == 1
    with pytest.raises(RepositoryError, match="conflicting replay"):
        repo.record_llm_call(
            call_id="call",
            run_id=run_id,
            experiment_id="e1",
            role="proposal",
            provider="fake",
            model="fixture",
            request_artifact_id=None,
            response_artifact_id=None,
            schema_valid=True,
            input_tokens=30,
            output_tokens=4,
            wall_seconds=0.5,
        )


def test_artifact_content_can_link_to_multiple_experiments_safely(tmp_path: Path) -> None:
    repo, database, run_id = repository(tmp_path)
    repo.create_experiment(run_id, proposal("e1"), "root")
    repo.create_experiment(run_id, proposal("e2"), "root")
    path = tmp_path / "shared.log"
    path.write_text("same evidence", encoding="utf-8")
    ref = artifact_ref(path, "fixture_log")
    first = repo.register_artifact(ref, experiment_id="e1")
    second = repo.register_artifact(ref, experiment_id="e2")
    assert first != second
    assert repo.register_artifact(ref, experiment_id="e1") == first
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM artifact_links").fetchone()[0] == 2
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM event_outbox WHERE event_type='artifact.registered'"
            ).fetchone()[0]
            == 2
        )


def test_event_export_is_atomic_deterministic_rebuild(tmp_path: Path) -> None:
    _, database, run_id = repository(tmp_path)
    path = tmp_path / "events.jsonl"
    count = export_events(database, run_id, path)
    first = path.read_bytes()
    assert export_events(database, run_id, path) == count
    assert path.read_bytes() == first
    assert len(path.read_text(encoding="utf-8").splitlines()) == count
    assert verify_event_chain(path)


def test_schema_migration_and_workspace_provenance(tmp_path: Path) -> None:
    repo, database, run_id = repository(tmp_path)
    repo.create_experiment(
        run_id,
        proposal("e1"),
        "root",
        method_card_id="F01",
        experiment_kind="FIXTURE",
    )
    assert not repo.record_experiment_workspace(
        "e1", workspace_path="/tmp/worktree", branch_name="fixture/e1", commit_sha="abc"
    )["idempotent"]
    assert repo.record_experiment_workspace(
        "e1", workspace_path="/tmp/worktree", branch_name="fixture/e1", commit_sha="abc"
    )["idempotent"]
    experiment = repo.get_experiment("e1")
    assert experiment["method_card_id"] == "F01"
    assert experiment["experiment_kind"] == "FIXTURE"
    assert experiment["workspace_path"] == "/tmp/worktree"
    with database.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 6
        assert connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 6


def test_migration_adds_production_run_columns_to_old_database(tmp_path: Path) -> None:
    database = Database(tmp_path / "old-state.sqlite3")
    with database.connect() as connection:
        connection.execute(
            "CREATE TABLE runs ("
            "run_id TEXT PRIMARY KEY,state TEXT NOT NULL,created_at TEXT NOT NULL,"
            "updated_at TEXT NOT NULL,deadline_epoch_ms INTEGER NOT NULL,root_commit TEXT NOT NULL,"
            "environment_sha256 TEXT NOT NULL,data_manifest_sha256 TEXT NOT NULL,"
            "evaluator_sha256 TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO runs(run_id,state,created_at,updated_at,deadline_epoch_ms,root_commit,"
            "environment_sha256,data_manifest_sha256,evaluator_sha256) VALUES(?,?,?,?,?,?,?,?,?)",
            ("legacy", "INITIALIZING", "then", "then", 1, "root", HASH, HASH, HASH),
        )
        connection.execute("PRAGMA user_version = 1")

    database.initialize()

    with database.connect() as connection:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(runs)").fetchall()}
        run = connection.execute("SELECT * FROM runs WHERE run_id='legacy'").fetchone()
        assert {
            "hypothesis_count",
            "official_evaluation_count",
            "non_improvement_streak",
            "best_primary_units",
            "best_ever_experiment_id",
            "search_champion_experiment_id",
            "shadow_best_primary_units",
            "shadow_champion_experiment_id",
            "validation_phase",
            "finalist_experiment_id",
            "official_evaluated_at",
            "stop_reason",
        } <= columns
        assert run["hypothesis_count"] == 0
        assert run["official_evaluation_count"] == 0
        assert run["non_improvement_streak"] == 0
        assert run["validation_phase"] == ValidationPhase.DISCOVERY
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 6


def test_migration_adds_immutable_repair_revision_columns(tmp_path: Path) -> None:
    database = Database(tmp_path / "old-repairs.sqlite3")
    database.initialize()
    with database.connect() as connection:
        connection.execute("DROP TABLE experiment_repairs")
        connection.execute(
            "CREATE TABLE experiment_repairs ("
            "repair_id TEXT PRIMARY KEY,experiment_id TEXT NOT NULL,"
            "repair_number INTEGER NOT NULL,phase TEXT NOT NULL,failure_status TEXT NOT NULL,"
            "plan_json TEXT NOT NULL,evidence_json TEXT,created_at TEXT NOT NULL,"
            "completed_at TEXT,UNIQUE(experiment_id,repair_number))"
        )
        connection.execute("PRAGMA user_version = 3")

    database.initialize()

    with database.connect() as connection:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(experiment_repairs)").fetchall()
        }
        assert {
            "previous_commit_sha",
            "repaired_commit_sha",
            "previous_config_sha256",
            "repaired_config_sha256",
            "effective_config_artifact_id",
        } <= columns
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 6


def test_repair_revision_atomically_supersedes_effective_commit_and_config(
    tmp_path: Path,
) -> None:
    repo, database, run_id = repository(tmp_path)
    original = tmp_path / "original.yaml"
    original.write_text("batch_size: 1024\n", encoding="utf-8")
    repaired = tmp_path / "repaired.yaml"
    repaired.write_text("batch_size: 512\n", encoding="utf-8")
    original_ref = artifact_ref(original, "experiment_config")
    repaired_ref = artifact_ref(repaired, "repaired_experiment_config")
    repo.create_experiment(
        run_id,
        proposal("repair-revision"),
        "root",
        commit_sha="candidate-v1",
        config_sha256=original_ref.sha256,
    )
    repo.register_artifact(original_ref, experiment_id="repair-revision")
    repo.transition_experiment(
        "repair-revision",
        ExperimentState.PROPOSED,
        ExperimentState.WORKTREE_READY,
        idempotency_key="repair-revision:worktree",
    )
    repo.transition_experiment(
        "repair-revision",
        ExperimentState.WORKTREE_READY,
        ExperimentState.FAILED_REPAIRABLE,
        idempotency_key="repair-revision:failed",
    )
    reservation = repo.reserve_experiment_repair(
        experiment_id="repair-revision",
        phase="cheap",
        failure_status=AttemptStatus.NAN,
        plan={"action": "request_constrained_patch"},
    )
    repo.transition_experiment(
        "repair-revision",
        ExperimentState.FAILED_REPAIRABLE,
        ExperimentState.REPAIRING,
        idempotency_key="repair-revision:repairing",
    )
    repo.register_artifact(repaired_ref, experiment_id="repair-revision")

    first = repo.apply_experiment_repair_revision(
        reservation["repair_id"],
        repaired_commit_sha="candidate-v2",
        effective_config_artifact_id=repaired_ref.artifact_id,
    )
    replay = repo.apply_experiment_repair_revision(
        reservation["repair_id"],
        repaired_commit_sha="candidate-v2",
        effective_config_artifact_id=repaired_ref.artifact_id,
    )

    assert not first["idempotent"]
    assert replay["idempotent"]
    experiment = repo.get_experiment("repair-revision")
    assert experiment["commit_sha"] == "candidate-v2"
    assert experiment["config_sha256"] == repaired_ref.sha256
    assert repo.get_run(run_id)["hypothesis_count"] == 1
    with database.connect() as connection:
        revision = connection.execute(
            "SELECT * FROM experiment_repairs WHERE repair_id=?",
            (reservation["repair_id"],),
        ).fetchone()
    assert revision["previous_commit_sha"] == "candidate-v1"
    assert revision["repaired_commit_sha"] == "candidate-v2"
    assert revision["previous_config_sha256"] == original_ref.sha256
    assert revision["repaired_config_sha256"] == repaired_ref.sha256
    assert revision["effective_config_artifact_id"] == repaired_ref.artifact_id
    with pytest.raises(RepositoryError, match="conflicting replay"):
        repo.apply_experiment_repair_revision(
            reservation["repair_id"],
            repaired_commit_sha="candidate-v3",
            effective_config_artifact_id=repaired_ref.artifact_id,
        )
