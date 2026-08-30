from __future__ import annotations

import os
import socket
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rex.agents.recovery import decide_repair
from rex.contracts import AttemptStatus, ExperimentProposal, Operator, RunResult
from rex.control.budget import deadline_epoch_ms
from rex.store.db import Database
from rex.store.event_log import export_events, verify_event_chain
from rex.store.repository import ExperimentRepository, RepositoryError


HASH = "0" * 64


def _repository(tmp_path: Path) -> tuple[ExperimentRepository, Database, str]:
    database = Database(tmp_path / "fault-state.sqlite3")
    database.initialize()
    repository = ExperimentRepository(database)
    run_id = "fault-run"
    repository.create_run(
        run_id=run_id,
        deadline_epoch_ms=deadline_epoch_ms(100),
        root_commit="fixture-root",
        environment_sha256=HASH,
        data_manifest_sha256=HASH,
        evaluator_sha256=HASH,
    )
    return repository, database, run_id


def _proposal(experiment_id: str) -> ExperimentProposal:
    return ExperimentProposal(
        experiment_id=experiment_id,
        parent_id=None,
        operator=Operator.REPAIR,
        hypothesis="A fixture failure exercises bounded repair.",
        mechanism="The repair dispatcher counts attempts without changing the incumbent.",
        primary_change="fixture-only failure",
        files_to_change=["src/rex/models/experimental/fixture.py"],
        expected_metric_effects={"fixture": "none"},
        falsifier="The fixture unexpectedly succeeds.",
        leakage_analysis="No competition data is used.",
        estimated_seconds=1,
        cheap_rung={"fixture": True},
        full_rung={"fixture": True},
    )


def _nan_result(experiment_id: str, attempt_id: str) -> RunResult:
    return RunResult(
        run_id="fault-run",
        experiment_id=experiment_id,
        attempt_id=attempt_id,
        status=AttemptStatus.NAN,
        error_type="WorkerFailure",
        error_summary="fixture non-finite loss",
        command_sha256=HASH,
        commit_sha="fixture-root",
        config_sha256=HASH,
        data_view_sha256=HASH,
        environment_sha256=HASH,
        wall_seconds=0.01,
    )


def test_stale_session_takeover_leaves_exactly_one_active_owner(tmp_path: Path) -> None:
    repository, database, run_id = _repository(tmp_path)
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    repository.open_process_session(
        session_id="crashed-owner", run_id=run_id, pid=101, host="fixture", now=started
    )
    takeover = repository.open_process_session(
        session_id="recovery-owner",
        run_id=run_id,
        pid=202,
        host="fixture",
        stale_after_seconds=10,
        now=started + timedelta(seconds=11),
    )
    assert takeover["stale_session_ids"] == ["crashed-owner"]
    with database.connect() as connection:
        active = connection.execute(
            "SELECT session_id FROM process_sessions WHERE run_id=? AND ended_at IS NULL",
            (run_id,),
        ).fetchall()
        stale = connection.execute(
            "SELECT exit_reason FROM process_sessions WHERE session_id='crashed-owner'"
        ).fetchone()
    assert [row["session_id"] for row in active] == ["recovery-owner"]
    assert stale["exit_reason"] == "stale_takeover"


def test_live_same_host_pid_blocks_process_session_takeover(tmp_path: Path) -> None:
    repository, _, run_id = _repository(tmp_path)
    repository.open_process_session(
        session_id="live-owner",
        run_id=run_id,
        pid=os.getpid(),
        host=socket.gethostname(),
    )

    with pytest.raises(RepositoryError, match="already has active process session"):
        repository.open_process_session(
            session_id="unsafe-takeover",
            run_id=run_id,
            pid=os.getpid(),
            host=socket.gethostname(),
            stale_after_seconds=900,
        )


def test_dead_same_host_pid_is_taken_over_without_waiting_for_timeout(tmp_path: Path) -> None:
    repository, database, run_id = _repository(tmp_path)
    process = subprocess.Popen(["true"])
    dead_pid = process.pid
    assert process.wait(timeout=5) == 0
    repository.open_process_session(
        session_id="dead-owner",
        run_id=run_id,
        pid=dead_pid,
        host=socket.gethostname(),
    )

    takeover = repository.open_process_session(
        session_id="immediate-recovery",
        run_id=run_id,
        pid=os.getpid(),
        host=socket.gethostname(),
        stale_after_seconds=900,
    )

    assert takeover["stale_session_ids"] == ["dead-owner"]
    with database.connect() as connection:
        stale = connection.execute(
            "SELECT exit_reason FROM process_sessions WHERE session_id='dead-owner'"
        ).fetchone()
    assert stale["exit_reason"] == "dead_process_takeover"


def test_event_export_rebuilds_atomically_without_replay_duplicates(tmp_path: Path) -> None:
    _, database, run_id = _repository(tmp_path)
    destination = tmp_path / "events.jsonl"
    first_count = export_events(database, run_id, destination)
    first_bytes = destination.read_bytes()

    # Simulate bytes from an interrupted prior export, then resume twice.
    destination.write_text("partial-event\n", encoding="utf-8")
    second_count = export_events(database, run_id, destination)
    second_bytes = destination.read_bytes()
    third_count = export_events(database, run_id, destination)

    assert first_count == second_count == third_count == 1
    assert second_bytes == first_bytes == destination.read_bytes()
    assert len(destination.read_text(encoding="utf-8").splitlines()) == 1
    assert verify_event_chain(destination)


def test_two_repair_limit_preserves_seeded_fixture_incumbent(tmp_path: Path) -> None:
    repository, database, run_id = _repository(tmp_path)
    repository.create_experiment(run_id, _proposal("broken-candidate"), "fixture-root")
    with database.transaction() as connection:
        connection.execute(
            "UPDATE runs SET best_primary_units=?,best_ever_experiment_id=?,"
            "search_champion_experiment_id=? WHERE run_id=?",
            (600_000, "fixture-incumbent", "fixture-incumbent", run_id),
        )

    for repair_number in range(3):
        result = _nan_result("broken-candidate", f"failure-{repair_number}")
        repository.record_attempt(result, rung="fixture", repair_number=repair_number)

    assert decide_repair(AttemptStatus.NAN, 0).repair_number == 1
    assert decide_repair(AttemptStatus.NAN, 1).repair_number == 2
    assert not decide_repair(AttemptStatus.NAN, 2).repair
    with database.connect() as connection:
        repair_rows = connection.execute(
            "SELECT repair_number,status FROM attempts WHERE experiment_id=? ORDER BY repair_number",
            ("broken-candidate",),
        ).fetchall()
    assert [(row["repair_number"], row["status"]) for row in repair_rows] == [
        (0, AttemptStatus.NAN),
        (1, AttemptStatus.NAN),
        (2, AttemptStatus.NAN),
    ]
    run = repository.get_run(run_id)
    assert run["best_primary_units"] == 600_000
    assert run["best_ever_experiment_id"] == "fixture-incumbent"
    assert run["search_champion_experiment_id"] == "fixture-incumbent"
