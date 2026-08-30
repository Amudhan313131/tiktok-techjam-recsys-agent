from __future__ import annotations

import json
import os
import subprocess
import sys
import sqlite3
from pathlib import Path

import pytest

import rex.rehearsal_clean as rehearsal_clean
from rex.rehearsal_clean import (
    R3Envelope,
    R3EnvelopeError,
    R3Options,
    compact_status_snapshot,
    validate_hash_lock,
    validate_r3_manifest_with_runtime,
)
from rex.store.db import Database


def test_hash_lock_requires_exact_pins_and_sha256(tmp_path: Path) -> None:
    lock = tmp_path / "requirements-lock.txt"
    lock.write_text("numpy==2.3.5\n", encoding="utf-8")
    with pytest.raises(R3EnvelopeError, match="missing sha256 hash"):
        validate_hash_lock(lock)

    lock.write_text(
        "numpy==2.3.5 \\\n" "    --hash=sha256:" + "a" * 64 + "\n",
        encoding="utf-8",
    )
    evidence = validate_hash_lock(lock)
    assert evidence["requirements"] == 1
    assert evidence["require_hashes"] is True


def test_install_requires_binary_wheels_and_records_installer_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source"
    data = tmp_path / "data"
    source.mkdir()
    data.mkdir()
    envelope = R3Envelope(
        R3Options(source, "HEAD", data, tmp_path / "output", "codex_cli")
    )
    envelope.clone = source
    envelope.logs.mkdir(parents=True)
    envelope.venv.joinpath("bin").mkdir(parents=True)
    python = envelope.venv / "bin/python"
    python.write_bytes(b"python")
    (source / "requirements-lock.txt").write_text(
        "numpy==2.3.5 \\\n" "    --hash=sha256:" + "a" * 64 + "\n",
        encoding="utf-8",
    )
    commands: dict[str, list[str]] = {}

    def checked(name, command, **kwargs):
        del kwargs
        commands[name] = list(command)
        if name == "installer-provenance":
            (envelope.logs / "installer-provenance.stdout.log").write_text(
                "{}\n", encoding="utf-8"
            )
        if name == "installed-inventory":
            (envelope.logs / "installed-inventory.stdout.log").write_text(
                "numpy==2.3.5\n", encoding="utf-8"
            )

    envelope._checked = checked  # type: ignore[method-assign]
    evidence = envelope._install()

    assert "--only-binary=:all:" in commands["dependencies"]
    assert evidence["only_binary"] is True
    assert len(evidence["installer_provenance_sha256"]) == 64
    assert len(evidence["installed_inventory_sha256"]) == 64


def test_r3_options_reject_fixed_paid_without_authorization_and_nested_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    data = tmp_path / "data"
    source.mkdir()
    data.mkdir()
    with pytest.raises(R3EnvelopeError, match="fixed is not live research"):
        R3Options(source, "HEAD", data, tmp_path / "out", "fixed").normalized()
    with pytest.raises(R3EnvelopeError, match="authorize-paid-api"):
        R3Options(source, "HEAD", data, tmp_path / "out", "openai_api").normalized()
    with pytest.raises(R3EnvelopeError, match="outside the source"):
        R3Options(source, "HEAD", data, source / "runs/r3", "codex_cli").normalized()
    for unsafe in ("../escape", "nested/run", "/tmp/escape", "", "space id"):
        with pytest.raises(R3EnvelopeError, match="safe path component"):
            R3Options(
                source,
                "HEAD",
                data,
                tmp_path / "out",
                "codex_cli",
                run_id=unsafe,
            ).normalized()


def _status_database(path: Path, run_id: str) -> None:
    path.parent.mkdir(parents=True)
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE runs (
          run_id TEXT, state TEXT, stop_reason TEXT, hypothesis_count INTEGER,
          official_evaluation_count INTEGER, non_improvement_streak INTEGER,
          best_primary_units INTEGER, search_champion_experiment_id TEXT,
          deadline_epoch_ms INTEGER, updated_at TEXT
        );
        CREATE TABLE experiments (
          run_id TEXT, experiment_id TEXT, iteration_number INTEGER,
          method_card_id TEXT, state TEXT, terminal_reason TEXT
        );
        CREATE TABLE process_sessions (
          run_id TEXT, session_id TEXT, pid INTEGER, last_heartbeat TEXT,
          ended_at TEXT, exit_reason TEXT, started_at TEXT
        );
        CREATE TABLE experiment_repairs (experiment_id TEXT);
        CREATE TABLE search_promotions (run_id TEXT);
        """
    )
    connection.execute(
        "INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,?)",
        (run_id, "SEARCHING", None, 2, 1, 0, 601000000, "candidate-1", 0, "now"),
    )
    connection.execute(
        "INSERT INTO experiments VALUES(?,?,?,?,?,?)",
        (run_id, "candidate-2", 2, "E02", "CHEAP_RUNNING", None),
    )
    connection.execute(
        "INSERT INTO process_sessions VALUES(?,?,?,?,?,?,?)",
        (run_id, "session", 123, "now", None, None, "now"),
    )
    connection.commit()
    connection.close()


def test_compact_status_snapshot_is_read_only_and_validation_only(tmp_path: Path) -> None:
    output = tmp_path / "r3"
    run_id = "r3-test"
    database = output / "runtime/runs" / run_id / "state.sqlite3"
    _status_database(database, run_id)
    envelope = {
        "run_id": run_id,
        "runs_dir": str(output / "runtime/runs"),
        "deadline_epoch": 9_999_999_999,
        "started_epoch": 1,
        "phase": "running",
        "source_commit": "a" * 40,
        "llm": "codex_cli",
        "fault": {"state": "pending"},
    }
    output.mkdir(exist_ok=True)
    (output / "envelope_state.json").write_text(json.dumps(envelope), encoding="utf-8")

    before = database.read_bytes()
    snapshot = compact_status_snapshot(output, reason="hourly")

    assert database.read_bytes() == before
    assert snapshot["run"]["state"] == "SEARCHING"
    assert snapshot["latest_experiment"]["state"] == "CHEAP_RUNNING"
    assert snapshot["validation_only"] is True
    assert snapshot["test_prediction_enabled"] is False
    assert (output / "status/latest.json").is_file()


def test_run_commands_create_then_resume_same_id_and_keep_external_deadline(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    data = tmp_path / "data"
    source.mkdir()
    data.mkdir()
    envelope = R3Envelope(
        R3Options(
            source_root=source,
            source_ref="HEAD",
            data_dir=data,
            output_dir=tmp_path / "outside",
            llm="claude_cli",
            run_id="r3-stable-id",
            wall_clock_seconds=60,
            finalization_reserve_seconds=10,
            skip_dependency_install=True,
        )
    )
    envelope.run_id = "r3-stable-id"
    envelope.clone = source
    initial = envelope._run_command(tmp_path / "runtime.json", resume=False)
    resumed = envelope._run_command(tmp_path / "runtime.json", resume=True)
    assert initial[initial.index("--run-id") + 1] == "r3-stable-id"
    assert "--resume" not in initial
    assert resumed[resumed.index("--resume") + 1] == "r3-stable-id"
    assert "--run-id" not in resumed
    for command in (initial, resumed):
        assert command[command.index("--llm") + 1] == "claude_cli"
        assert command[command.index("--external-deadline-epoch-ms") + 1].isdigit()
        assert "submission" not in " ".join(command).lower()


def test_openai_r3_forwards_explicit_paid_authorization(tmp_path: Path) -> None:
    source = tmp_path / "source"
    data = tmp_path / "data"
    source.mkdir()
    data.mkdir()
    envelope = R3Envelope(
        R3Options(
            source_root=source,
            source_ref="HEAD",
            data_dir=data,
            output_dir=tmp_path / "outside",
            llm="openai_api",
            run_id="r3-openai",
            authorize_paid_api=True,
            wall_clock_seconds=60,
            finalization_reserve_seconds=10,
            skip_dependency_install=True,
        )
    )
    envelope.run_id = "r3-openai"

    command = envelope._run_command(tmp_path / "runtime.json", resume=False)

    assert "--authorize-paid-api" in command


def test_runtime_template_is_json_and_hard_disables_submission() -> None:
    root = Path(__file__).resolve().parents[2]
    template = json.loads((root / "configs/run/rehearsal_r3.yaml").read_text(encoding="utf-8"))
    assert template["execution_mode"] == "production"
    assert template["scientific_execution_enabled"] is True
    assert template["confirmation_enabled"] is False
    assert template["test_prediction_enabled"] is False
    assert template["final_submission_enabled"] is False
    assert "fixed" not in template["llm"]["auto"]["provider_order"]


def test_runtime_configuration_binds_external_raw_data_directory(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    data = tmp_path / "external-data"
    data.mkdir()
    output = tmp_path / "outside"
    envelope = R3Envelope(
        R3Options(
            source_root=root,
            source_ref="HEAD",
            data_dir=data,
            output_dir=output,
            llm="codex_cli",
            run_id="r3-data-binding",
            wall_clock_seconds=60,
            finalization_reserve_seconds=10,
            skip_dependency_install=True,
        )
    )
    envelope.clone = root
    envelope.runtime.mkdir(parents=True)
    envelope.runs.mkdir(parents=True)

    config_path, _budget_path = envelope._write_runtime_files()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["raw_data_dir"] == str(data.resolve())


def test_controlled_failure_kills_only_verified_lease_owner_once(tmp_path: Path) -> None:
    source = tmp_path / "source"
    data = tmp_path / "data"
    source.mkdir()
    data.mkdir()
    output = tmp_path / "outside"
    envelope = R3Envelope(
        R3Options(
            source_root=source,
            source_ref="HEAD",
            data_dir=data,
            output_dir=output,
            llm="codex_cli",
            run_id="r3-kill-test",
            wall_clock_seconds=60,
            finalization_reserve_seconds=10,
            skip_dependency_install=True,
        )
    )
    envelope.run_id = "r3-kill-test"
    envelope.runtime.mkdir(parents=True)
    envelope.runs.mkdir(parents=True)
    envelope._snapshot = lambda *args, **kwargs: None  # type: ignore[method-assign]
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    lease_path = output / "lease.json"
    lease = {
        "state": "active",
        "owner_pid": process.pid,
        "pid": process.pid + 1,
        "pgid": process.pid + 1,
        "request_sha256": "a" * 64,
        "execution_sha256": "b" * 64,
    }
    lease_path.write_text(json.dumps(lease), encoding="utf-8")
    envelope._inject(process, lease_path, lease)
    assert process.poll() is not None
    assert envelope.fault["count"] == 1
    assert envelope.fault["state"] == "injected"
    with pytest.raises(R3EnvelopeError, match="exactly once"):
        envelope._inject(process, lease_path, lease)
    with pytest.raises(ProcessLookupError):
        os.kill(process.pid, 0)


def _artifact(path: Path, kind: str) -> dict[str, object]:
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "artifact_id": f"test-{kind}-{digest[:12]}",
        "kind": kind,
        "path": str(path.resolve()),
        "sha256": digest,
        "size_bytes": path.stat().st_size,
        "schema_version": "1.0",
    }


def _valid_manifest(tmp_path: Path) -> dict[str, object]:
    best = tmp_path / "best.json"
    report = tmp_path / "report.json"
    best.write_text("{}\n", encoding="utf-8")
    report.write_text("{}\n", encoding="utf-8")
    sha = "a" * 64
    return {
        "schema_version": "1.0",
        "level": "R3",
        "rehearsal_id": "rehearsal-r3-test",
        "run_id": "r3-test",
        "state": "COMPLETE",
        "stop_reason": "epsilon_plateau",
        "started_epoch_ms": 1_000,
        "deadline_epoch_ms": 22_000,
        "elapsed_seconds": 20.0,
        "source_commit": "b" * 40,
        "source_tree_sha256": sha,
        "environment_lock_sha256": sha,
        "environment_sha256": sha,
        "data_manifest_sha256": sha,
        "starter_manifest_sha256": sha,
        "evaluator_sha256": sha,
        "provider_requested": "codex_cli",
        "provider_actual": "codex_cli",
        "fault_injected": True,
        "fault_recovered": True,
        "source_unchanged": True,
        "best_valid_manifest": _artifact(best, "best_valid_manifest"),
        "report_artifacts": [_artifact(report, "r3_report_artifact")],
        "test_prediction_created": False,
        "test_scored": False,
        "submission_created": False,
        "started_at": "2026-08-30T07:00:00+00:00",
        "completed_at": "2026-08-30T07:00:20+00:00",
        "wall_clock_ceiling_seconds": 21,
        "within_six_hour_ceiling": True,
        "llm": "codex_cli",
        "provider_calls": [{"provider": "codex_cli", "model": "configured", "calls": 2}],
        "paid_api_authorized": False,
        "dependency": {"require_hashes": True, "only_binary": True},
        "preflight": {"test_rows": 170_588},
        "controlled_failure": {"count": 1},
        "source_audit": {"unchanged": True},
        "clone_audit": {"unchanged": True},
        "validation": {"validation_only": True},
        "winner": {"preserved": True},
        "status": {"run": {"state": "COMPLETE"}},
        "hourly_snapshot_count": 1,
        "evidence": {"report.json": {"sha256": sha, "size_bytes": 3}},
    }


def test_manifest_round_trips_through_fresh_runtime_contract(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    path = tmp_path / "r3_manifest.json"
    manifest = _valid_manifest(tmp_path)
    path.write_text(json.dumps(manifest), encoding="utf-8")

    normalized = validate_r3_manifest_with_runtime(
        path,
        python_executable=sys.executable,
        source_root=root,
        timeout_seconds=10,
    )

    assert normalized == manifest


def test_manifest_seal_fails_closed_when_contract_validation_crosses_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    data = tmp_path / "data"
    source.mkdir()
    data.mkdir()
    clock = [0.0]
    envelope = R3Envelope(
        R3Options(
            source,
            "HEAD",
            data,
            tmp_path / "output",
            "codex_cli",
            run_id="r3-deadline",
            wall_clock_seconds=10,
            finalization_reserve_seconds=1,
            skip_dependency_install=True,
        ),
        monotonic=lambda: clock[0],
        epoch=lambda: 1_000.0,
    )
    path = tmp_path / "r3_manifest.json"

    def cross_deadline(*args, **kwargs):
        del args, kwargs
        clock[0] = 11.0
        return {}

    monkeypatch.setattr(rehearsal_clean, "validate_r3_manifest_with_runtime", cross_deadline)
    with pytest.raises(R3EnvelopeError, match="while validating"):
        envelope._seal_manifest(path, {})
    assert not path.exists()


def _audit_database(path: Path, run_id: str = "r3-audit") -> sqlite3.Connection:
    Database(path).initialize()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute(
        "INSERT INTO runs(run_id,state,created_at,updated_at,deadline_epoch_ms,root_commit,"
        "environment_sha256,data_manifest_sha256,evaluator_sha256,hypothesis_count,"
        "official_evaluation_count,non_improvement_streak,best_primary_units,"
        "search_champion_experiment_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            run_id,
            "SEARCHING",
            "now",
            "now",
            1,
            "commit",
            "a" * 64,
            "b" * 64,
            "c" * 64,
            1,
            0,
            0,
            600_000_000,
            "baseline",
        ),
    )
    connection.execute(
        "INSERT INTO baseline_gates VALUES(?,?,?,?,?,?)",
        (run_id, 600_000_000, 0.6, 0.6, "[]", "now"),
    )
    connection.execute(
        "INSERT INTO experiments(experiment_id,run_id,iteration_number,operator,hypothesis,"
        "proposal_json,state,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
        ("exp-1", run_id, 1, "LOSS", "one controlled change", "{}", "CHEAP_RUNNING", "now", "now"),
    )
    connection.commit()
    return connection


def test_validation_only_audit_rejects_test_metrics_and_predict_attempts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    data = tmp_path / "data"
    source.mkdir()
    data.mkdir()
    envelope = R3Envelope(
        R3Options(source, "HEAD", data, tmp_path / "output", "codex_cli", run_id="r3-audit")
    )
    database = envelope.runs / envelope.run_id / "state.sqlite3"
    connection = _audit_database(database)
    connection.execute(
        "INSERT INTO attempts(attempt_id,experiment_id,rung,status) VALUES(?,?,?,?)",
        ("predict-1", "exp-1", "predict", "success"),
    )
    connection.execute(
        "INSERT INTO metrics(experiment_id,attempt_id,split,evaluator_sha256,gauc,ndcg5,"
        "primary_score,primary_units,rows,users) VALUES(?,?,?,?,?,?,?,?,?,?)",
        ("exp-1", "predict-1", "test", "c" * 64, 0.6, 0.6, 0.6, 600_000_000, 1, 1),
    )
    connection.commit()
    connection.close()

    with pytest.raises(R3EnvelopeError, match="forbidden test/submission"):
        envelope._validation_only_audit()


def test_recovery_audit_requires_the_injected_attempt_exactly_once(tmp_path: Path) -> None:
    source = tmp_path / "source"
    data = tmp_path / "data"
    source.mkdir()
    data.mkdir()
    envelope = R3Envelope(
        R3Options(source, "HEAD", data, tmp_path / "output", "codex_cli", run_id="r3-audit")
    )
    database = envelope.runs / envelope.run_id / "state.sqlite3"
    connection = _audit_database(database)
    connection.executemany(
        "INSERT INTO process_sessions(session_id,run_id,pid,host,started_at,ended_at) "
        "VALUES(?,?,?,?,?,?)",
        [
            ("first", envelope.run_id, 10, "host", "1", "2"),
            ("second", envelope.run_id, 11, "host", "3", "4"),
        ],
    )
    connection.commit()
    connection.close()
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    lease = attempt_dir / "worker_lease.json"
    (attempt_dir / "worker_recovery.json").write_text(
        json.dumps({"events": [{"pid": 99, "outcome": "stale-lease-no-process"}]}),
        encoding="utf-8",
    )
    envelope.fault = {
        "worker_pid": 99,
        "attempt_id": "missing-attempt",
        "pre_fault_status": {"run": {"search_champion_experiment_id": "baseline"}},
        "pre_fault_database_audit": {},
    }

    with pytest.raises(R3EnvelopeError, match="preserved exactly once"):
        envelope._recovery_audit(lease)


def test_winner_audit_rejects_a_corrupt_checkpoint_member(tmp_path: Path) -> None:
    source = tmp_path / "source"
    data = tmp_path / "data"
    source.mkdir()
    data.mkdir()
    envelope = R3Envelope(
        R3Options(source, "HEAD", data, tmp_path / "output", "codex_cli", run_id="r3-winner")
    )
    root = envelope.runs / envelope.run_id / "best-valid"
    model_dir = root / "model"
    model_dir.mkdir(parents=True)
    checkpoint = model_dir / "model.npz"
    checkpoint.write_bytes(b"checkpoint")
    config_sha = "d" * 64
    commit = "e" * 40
    model_manifest = model_dir / "model_bundle.json"
    model_manifest.write_text(
        json.dumps(
            {
                "commit_sha": commit,
                "config_sha256": config_sha,
                "members": [
                    {
                        "name": "model.npz",
                        "sha256": _artifact(checkpoint, "checkpoint")["sha256"],
                        "size_bytes": checkpoint.stat().st_size,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = root / "best_valid_manifest.json"
    artifacts = {
        "model/model.npz": _artifact(checkpoint, "checkpoint"),
        "model/model_bundle.json": _artifact(model_manifest, "model_bundle"),
    }
    manifest_path.write_text(
        json.dumps(
            {
                "kind": "best_valid",
                "test_prediction_created": False,
                "commit_sha": commit,
                "config_sha256": config_sha,
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    checkpoint.write_bytes(b"corrupt")

    with pytest.raises(R3EnvelopeError, match="drifted"):
        envelope._winner()
