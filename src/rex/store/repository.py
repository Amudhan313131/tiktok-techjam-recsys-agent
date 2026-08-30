"""Transactional experiment repository with hash-chained outbox events."""

from __future__ import annotations

import hashlib
import json
import os
import socket
from datetime import datetime, timedelta, timezone
from typing import Any

from rex.contracts import (
    ArtifactRef,
    AttemptStatus,
    ExperimentProposal,
    ExperimentState,
    Metrics,
    RunResult,
    RunState,
)
from rex.control.budget import metric_units, update_metric_trackers
from rex.control.state_machine import require_experiment_transition, require_run_transition
from rex.data.manifest import canonical_json_bytes
from rex.store.db import Database


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RepositoryError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _require_identical(
    row: Any,
    expected: dict[str, Any],
    *,
    entity: str,
) -> None:
    conflicts = {
        key: {"stored": row[key], "incoming": value}
        for key, value in expected.items()
        if row[key] != value
    }
    if conflicts:
        raise RepositoryError(f"conflicting replay for {entity}: {_canonical_json(conflicts)}")


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _same_host_pid_is_missing(host: str | None, pid: int | None) -> bool:
    """Return true only when the recorded local process is conclusively gone."""

    if host != socket.gethostname() or pid is None or int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return False
    return False


class ExperimentRepository:
    def __init__(self, database: Database):
        self.database = database

    def _event(
        self,
        connection,
        *,
        run_id: str,
        event_type: str,
        aggregate_id: str,
        payload: dict[str, Any],
    ) -> None:
        prior = connection.execute(
            "SELECT event_hash FROM event_outbox WHERE run_id=? ORDER BY sequence DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        previous_hash = prior["event_hash"] if prior else None
        body = {
            "run_id": run_id,
            "event_type": event_type,
            "aggregate_id": aggregate_id,
            "payload": payload,
            "previous_hash": previous_hash,
        }
        event_hash = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
        connection.execute(
            "INSERT INTO event_outbox(run_id,event_type,aggregate_id,payload_json,previous_hash,event_hash,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (
                run_id,
                event_type,
                aggregate_id,
                json.dumps(payload, sort_keys=True),
                previous_hash,
                event_hash,
                utc_now(),
            ),
        )

    def create_run(
        self,
        *,
        run_id: str,
        deadline_epoch_ms: int,
        root_commit: str,
        environment_sha256: str,
        data_manifest_sha256: str,
        evaluator_sha256: str,
    ) -> None:
        now = utc_now()
        with self.database.transaction() as connection:
            existing = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if existing is not None:
                _require_identical(
                    existing,
                    {
                        "deadline_epoch_ms": deadline_epoch_ms,
                        "root_commit": root_commit,
                        "environment_sha256": environment_sha256,
                        "data_manifest_sha256": data_manifest_sha256,
                        "evaluator_sha256": evaluator_sha256,
                    },
                    entity=f"run {run_id}",
                )
                return
            connection.execute(
                "INSERT INTO runs(run_id,state,created_at,updated_at,deadline_epoch_ms,root_commit,"
                "environment_sha256,data_manifest_sha256,evaluator_sha256) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    RunState.INITIALIZING,
                    now,
                    now,
                    deadline_epoch_ms,
                    root_commit,
                    environment_sha256,
                    data_manifest_sha256,
                    evaluator_sha256,
                ),
            )
            self._event(
                connection,
                run_id=run_id,
                event_type="run.created",
                aggregate_id=run_id,
                payload={"state": RunState.INITIALIZING},
            )

    def transition_run(self, run_id: str, expected: RunState, next_state: RunState, reason: str | None = None) -> None:
        require_run_transition(expected, next_state)
        with self.database.transaction() as connection:
            row = connection.execute("SELECT state FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None or RunState(row["state"]) != expected:
                raise RepositoryError(f"run {run_id} is not in expected state {expected}")
            connection.execute(
                "UPDATE runs SET state=?, updated_at=?, stop_reason=COALESCE(?,stop_reason) WHERE run_id=?",
                (next_state, utc_now(), reason, run_id),
            )
            self._event(
                connection,
                run_id=run_id,
                event_type="run.transition",
                aggregate_id=run_id,
                payload={"from": expected, "to": next_state, "reason": reason},
            )

    def establish_baseline(
        self,
        *,
        run_id: str,
        metrics: Metrics,
        evidence_artifact_ids: list[str],
    ) -> dict[str, Any]:
        """Atomically establish the validation baseline before search can start."""

        evidence = sorted(set(evidence_artifact_ids))
        evidence_json = _canonical_json(evidence)
        primary_units = metric_units(metrics.primary)
        with self.database.transaction() as connection:
            run = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if run is None:
                raise RepositoryError(f"unknown run: {run_id}")
            prior = connection.execute(
                "SELECT * FROM baseline_gates WHERE run_id=?", (run_id,)
            ).fetchone()
            expected = {
                "primary_units": primary_units,
                "gauc": metrics.GAUC,
                "ndcg5": metrics.ndcg5,
                "evidence_json": evidence_json,
            }
            if prior is not None:
                _require_identical(prior, expected, entity=f"baseline gate {run_id}")
                return {"idempotent": True, "primary_units": primary_units}
            if RunState(run["state"]) != RunState.BASELINE_VERIFYING:
                raise RepositoryError("baseline can only be established while BASELINE_VERIFYING")
            if evidence:
                placeholders = ",".join("?" for _ in evidence)
                known = connection.execute(
                    f"SELECT COUNT(DISTINCT artifact_id) AS n FROM artifacts "
                    f"WHERE artifact_id IN ({placeholders})",
                    evidence,
                ).fetchone()["n"]
                if known != len(evidence):
                    raise RepositoryError("baseline cites missing evidence artifacts")
            connection.execute(
                "INSERT INTO baseline_gates(run_id,primary_units,gauc,ndcg5,evidence_json,created_at) "
                "VALUES(?,?,?,?,?,?)",
                (run_id, primary_units, metrics.GAUC, metrics.ndcg5, evidence_json, utc_now()),
            )
            connection.execute(
                "UPDATE runs SET best_primary_units=?,best_ever_experiment_id='baseline',"
                "search_champion_experiment_id='baseline',updated_at=? WHERE run_id=?",
                (primary_units, utc_now(), run_id),
            )
            self._event(
                connection,
                run_id=run_id,
                event_type="baseline.validated",
                aggregate_id=run_id,
                payload={
                    "primary_units": primary_units,
                    "gauc": metrics.GAUC,
                    "ndcg5": metrics.ndcg5,
                    "evidence_artifact_ids": evidence,
                },
            )
            return {"idempotent": False, "primary_units": primary_units}

    def get_baseline(self, run_id: str) -> dict[str, Any] | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM baseline_gates WHERE run_id=?", (run_id,)
            ).fetchone()
        return None if row is None else dict(row)

    def open_process_session(
        self,
        *,
        session_id: str,
        run_id: str,
        pid: int | None = None,
        host: str | None = None,
        stale_after_seconds: float | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Open the run lease, optionally taking over only sessions proven stale."""
        if stale_after_seconds is not None and stale_after_seconds < 0:
            raise RepositoryError("stale_after_seconds must be non-negative")
        observed_at = now or datetime.now(timezone.utc)
        observed_text = observed_at.isoformat()
        process_id = os.getpid() if pid is None else pid
        hostname = socket.gethostname() if host is None else host
        with self.database.transaction() as connection:
            if connection.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone() is None:
                raise RepositoryError(f"unknown run: {run_id}")
            existing = connection.execute(
                "SELECT * FROM process_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if existing is not None:
                _require_identical(
                    existing,
                    {"run_id": run_id, "pid": process_id, "host": hostname},
                    entity=f"process session {session_id}",
                )
                if existing["ended_at"] is not None:
                    raise RepositoryError(f"process session {session_id} is already closed")
                return {"idempotent": True, "stale_session_ids": []}

            active = connection.execute(
                "SELECT * FROM process_sessions WHERE run_id=? AND ended_at IS NULL ORDER BY started_at",
                (run_id,),
            ).fetchall()
            stale_ids: list[str] = []
            stale_reasons: dict[str, str] = {}
            for item in active:
                latest = item["last_heartbeat"] or item["started_at"]
                timed_out = (
                    stale_after_seconds is not None
                    and observed_at - _parse_timestamp(latest)
                    >= timedelta(seconds=stale_after_seconds)
                )
                dead_local_process = _same_host_pid_is_missing(item["host"], item["pid"])
                stale = timed_out or dead_local_process
                if not stale:
                    raise RepositoryError(
                        f"run {run_id} already has active process session {item['session_id']}"
                    )
                stale_ids.append(item["session_id"])
                stale_reasons[item["session_id"]] = (
                    "dead_process_takeover" if dead_local_process else "stale_takeover"
                )
            for stale_id in stale_ids:
                connection.execute(
                    "UPDATE process_sessions SET ended_at=?,exit_reason=? WHERE session_id=?",
                    (observed_text, stale_reasons[stale_id], stale_id),
                )
                self._event(
                    connection,
                    run_id=run_id,
                    event_type="session.stale",
                    aggregate_id=stale_id,
                    payload={
                        "taken_over_by": session_id,
                        "reason": stale_reasons[stale_id],
                    },
                )
            connection.execute(
                "INSERT INTO process_sessions(session_id,run_id,pid,host,started_at,last_heartbeat) "
                "VALUES(?,?,?,?,?,?)",
                (session_id, run_id, process_id, hostname, observed_text, observed_text),
            )
            self._event(
                connection,
                run_id=run_id,
                event_type="session.opened",
                aggregate_id=session_id,
                payload={"pid": process_id, "host": hostname, "stale_session_ids": stale_ids},
            )
            return {"idempotent": False, "stale_session_ids": stale_ids}

    def heartbeat_process_session(
        self,
        session_id: str,
        monotonic_seconds: float,
        *,
        console_log_sha256: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if monotonic_seconds < 0:
            raise RepositoryError("monotonic_seconds must be non-negative")
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM process_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if row is None:
                raise RepositoryError(f"unknown process session: {session_id}")
            if row["ended_at"] is not None:
                raise RepositoryError(f"process session {session_id} is already closed")
            previous = float(row["monotonic_seconds"])
            if monotonic_seconds < previous:
                raise RepositoryError("session monotonic clock moved backwards")
            if monotonic_seconds == previous and console_log_sha256 == row["console_log_sha256"]:
                return {"idempotent": True}
            connection.execute(
                "UPDATE process_sessions SET monotonic_seconds=?,last_heartbeat=?,"
                "console_log_sha256=COALESCE(?,console_log_sha256) WHERE session_id=?",
                (
                    monotonic_seconds,
                    (now or datetime.now(timezone.utc)).isoformat(),
                    console_log_sha256,
                    session_id,
                ),
            )
            return {"idempotent": False}

    def close_process_session(
        self,
        session_id: str,
        *,
        exit_reason: str,
        monotonic_seconds: float | None = None,
        console_log_sha256: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM process_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if row is None:
                raise RepositoryError(f"unknown process session: {session_id}")
            final_monotonic = (
                float(row["monotonic_seconds"])
                if monotonic_seconds is None
                else monotonic_seconds
            )
            if final_monotonic < float(row["monotonic_seconds"]):
                raise RepositoryError("session monotonic clock moved backwards")
            final_log = console_log_sha256 or row["console_log_sha256"]
            if row["ended_at"] is not None:
                _require_identical(
                    row,
                    {
                        "exit_reason": exit_reason,
                        "monotonic_seconds": final_monotonic,
                        "console_log_sha256": final_log,
                    },
                    entity=f"process session close {session_id}",
                )
                return {"idempotent": True}
            ended_at = (now or datetime.now(timezone.utc)).isoformat()
            connection.execute(
                "UPDATE process_sessions SET ended_at=?,last_heartbeat=?,exit_reason=?,"
                "monotonic_seconds=?,console_log_sha256=? WHERE session_id=?",
                (ended_at, ended_at, exit_reason, final_monotonic, final_log, session_id),
            )
            self._event(
                connection,
                run_id=row["run_id"],
                event_type="session.closed",
                aggregate_id=session_id,
                payload={"exit_reason": exit_reason, "monotonic_seconds": final_monotonic},
            )
            return {"idempotent": False}

    def list_stale_process_sessions(
        self,
        run_id: str,
        *,
        stale_after_seconds: float,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        observed_at = now or datetime.now(timezone.utc)
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM process_sessions WHERE run_id=? AND ended_at IS NULL ORDER BY started_at",
                (run_id,),
            ).fetchall()
        return [
            dict(row)
            for row in rows
            if observed_at - _parse_timestamp(row["last_heartbeat"] or row["started_at"])
            >= timedelta(seconds=stale_after_seconds)
        ]

    def create_experiment(
        self,
        run_id: str,
        proposal: ExperimentProposal,
        parent_commit: str,
        *,
        max_hypotheses: int = 50,
        workspace_path: str | None = None,
        branch_name: str | None = None,
        commit_sha: str | None = None,
        config_sha256: str | None = None,
        method_card_id: str | None = None,
        experiment_kind: str | None = None,
    ) -> int:
        now = utc_now()
        proposal_json = _canonical_json(proposal.model_dump(mode="json"))
        with self.database.transaction() as connection:
            run = connection.execute(
                "SELECT hypothesis_count FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if run is None:
                raise RepositoryError(f"unknown run: {run_id}")
            existing = connection.execute(
                "SELECT * FROM experiments WHERE experiment_id=?", (proposal.experiment_id,)
            ).fetchone()
            if existing is not None:
                _require_identical(
                    existing,
                    {
                        "run_id": run_id,
                        "parent_id": proposal.parent_id,
                        "operator": proposal.operator,
                        "hypothesis": proposal.hypothesis,
                        "parent_commit": parent_commit,
                        "workspace_path": workspace_path,
                        "branch_name": branch_name,
                        "commit_sha": commit_sha,
                        "config_sha256": config_sha256,
                        "method_card_id": method_card_id,
                        "experiment_kind": experiment_kind,
                    },
                    entity=f"experiment {proposal.experiment_id}",
                )
                if json.loads(existing["proposal_json"]) != json.loads(proposal_json):
                    raise RepositoryError(
                        f"conflicting replay for experiment {proposal.experiment_id}: proposal_json"
                    )
                return int(existing["iteration_number"])
            if int(run["hypothesis_count"]) >= max_hypotheses:
                raise RepositoryError(f"hypothesis cap reached: {max_hypotheses}")
            iteration = int(run["hypothesis_count"]) + 1
            connection.execute(
                "INSERT INTO experiments(experiment_id,run_id,iteration_number,parent_id,operator,hypothesis,"
                "proposal_json,state,parent_commit,workspace_path,branch_name,commit_sha,config_sha256,"
                "method_card_id,experiment_kind,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    proposal.experiment_id,
                    run_id,
                    iteration,
                    proposal.parent_id,
                    proposal.operator,
                    proposal.hypothesis,
                    proposal_json,
                    ExperimentState.PROPOSED,
                    parent_commit,
                    workspace_path,
                    branch_name,
                    commit_sha,
                    config_sha256,
                    method_card_id,
                    experiment_kind,
                    now,
                    now,
                ),
            )
            connection.execute(
                "UPDATE runs SET hypothesis_count=?, updated_at=? WHERE run_id=?",
                (iteration, now, run_id),
            )
            self._event(
                connection,
                run_id=run_id,
                event_type="experiment.proposed",
                aggregate_id=proposal.experiment_id,
                payload={"iteration": iteration, "parent": proposal.parent_id, "operator": proposal.operator},
            )
            return iteration

    def record_experiment_workspace(
        self,
        experiment_id: str,
        *,
        workspace_path: str,
        branch_name: str,
        commit_sha: str | None = None,
        config_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Bind immutable workspace provenance, allowing later NULL fields to be filled once."""
        incoming = {
            "workspace_path": workspace_path,
            "branch_name": branch_name,
            "commit_sha": commit_sha,
            "config_sha256": config_sha256,
        }
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,)
            ).fetchone()
            if row is None:
                raise RepositoryError(f"unknown experiment: {experiment_id}")
            updates: dict[str, Any] = {}
            for key, value in incoming.items():
                if value is None:
                    continue
                if row[key] is None:
                    updates[key] = value
                elif row[key] != value:
                    raise RepositoryError(
                        f"conflicting workspace provenance for {experiment_id}: {key}"
                    )
            if not updates:
                return {"idempotent": True}
            assignments = ",".join(f"{key}=?" for key in updates)
            connection.execute(
                f"UPDATE experiments SET {assignments},updated_at=? WHERE experiment_id=?",
                (*updates.values(), utc_now(), experiment_id),
            )
            self._event(
                connection,
                run_id=row["run_id"],
                event_type="experiment.workspace_bound",
                aggregate_id=experiment_id,
                payload=updates,
            )
            return {"idempotent": False}

    def transition_experiment(
        self,
        experiment_id: str,
        expected: ExperimentState,
        next_state: ExperimentState,
        *,
        payload: dict[str, Any] | None = None,
        idempotency_key: str,
    ) -> None:
        require_experiment_transition(expected, next_state)
        payload = payload or {}
        with self.database.transaction() as connection:
            duplicate = connection.execute(
                "SELECT * FROM transitions WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
            if duplicate:
                _require_identical(
                    duplicate,
                    {
                        "experiment_id": experiment_id,
                        "from_state": expected,
                        "to_state": next_state,
                    },
                    entity=f"transition {idempotency_key}",
                )
                if json.loads(duplicate["payload_json"]) != payload:
                    raise RepositoryError(f"conflicting replay for transition {idempotency_key}")
                return
            row = connection.execute(
                "SELECT run_id,state FROM experiments WHERE experiment_id=?", (experiment_id,)
            ).fetchone()
            if row is None or ExperimentState(row["state"]) != expected:
                raise RepositoryError(f"experiment {experiment_id} is not in expected state {expected}")
            now = utc_now()
            connection.execute(
                "UPDATE experiments SET state=?,updated_at=?,terminal_reason=COALESCE(?,terminal_reason) "
                "WHERE experiment_id=?",
                (next_state, now, payload.get("reason"), experiment_id),
            )
            connection.execute(
                "INSERT INTO transitions(experiment_id,from_state,to_state,payload_json,created_at,idempotency_key) "
                "VALUES(?,?,?,?,?,?)",
                (experiment_id, expected, next_state, _canonical_json(payload), now, idempotency_key),
            )
            self._event(
                connection,
                run_id=row["run_id"],
                event_type="experiment.transition",
                aggregate_id=experiment_id,
                payload={"from": expected, "to": next_state, **payload},
            )

    def register_artifact(
        self,
        ref: ArtifactRef,
        *,
        experiment_id: str | None = None,
        attempt_id: str | None = None,
    ) -> str:
        run_id = None
        with self.database.transaction() as connection:
            if attempt_id:
                attempt = connection.execute(
                    "SELECT experiment_id FROM attempts WHERE attempt_id=?", (attempt_id,)
                ).fetchone()
                if attempt is None:
                    raise RepositoryError(f"unknown attempt: {attempt_id}")
                if experiment_id is None:
                    experiment_id = attempt["experiment_id"]
                elif experiment_id != attempt["experiment_id"]:
                    raise RepositoryError("artifact attempt belongs to another experiment")
            if experiment_id:
                row = connection.execute(
                    "SELECT run_id FROM experiments WHERE experiment_id=?", (experiment_id,)
                ).fetchone()
                if row is None:
                    raise RepositoryError(f"unknown experiment: {experiment_id}")
                run_id = row["run_id"]
            existing = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id=?", (ref.artifact_id,)
            ).fetchone()
            expected_content = {
                "kind": ref.kind,
                "sha256": ref.sha256,
                "size_bytes": ref.size_bytes,
                "schema_version": ref.schema_version,
            }
            if existing is None:
                connection.execute(
                    "INSERT INTO artifacts(artifact_id,experiment_id,attempt_id,kind,path,sha256,"
                    "size_bytes,schema_version,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        ref.artifact_id,
                        experiment_id,
                        attempt_id,
                        ref.kind,
                        ref.path,
                        ref.sha256,
                        ref.size_bytes,
                        ref.schema_version,
                        utc_now(),
                    ),
                )
            else:
                _require_identical(
                    existing,
                    expected_content,
                    entity=f"artifact {ref.artifact_id}",
                )
            provenance = (
                f"{ref.artifact_id}\0{experiment_id or ''}\0{attempt_id or ''}\0{ref.path}"
            )
            link_id = "artifact-link-" + hashlib.sha256(provenance.encode("utf-8")).hexdigest()
            link = connection.execute(
                "SELECT * FROM artifact_links WHERE link_id=?", (link_id,)
            ).fetchone()
            if link is not None:
                _require_identical(
                    link,
                    {
                        "artifact_id": ref.artifact_id,
                        "experiment_id": experiment_id,
                        "attempt_id": attempt_id,
                        "artifact_path": ref.path,
                    },
                    entity=f"artifact link {link_id}",
                )
                return link_id
            connection.execute(
                "INSERT INTO artifact_links(link_id,artifact_id,experiment_id,attempt_id,artifact_path,"
                "created_at) VALUES(?,?,?,?,?,?)",
                (link_id, ref.artifact_id, experiment_id, attempt_id, ref.path, utc_now()),
            )
            if run_id:
                self._event(
                    connection,
                    run_id=run_id,
                    event_type="artifact.registered",
                    aggregate_id=link_id,
                    payload={
                        "artifact_id": ref.artifact_id,
                        "experiment_id": experiment_id,
                        "attempt_id": attempt_id,
                        "kind": ref.kind,
                        "sha256": ref.sha256,
                    },
                )
            return link_id

    def reserve_attempt(
        self,
        *,
        attempt_id: str,
        experiment_id: str,
        rung: str,
        repair_number: int = 0,
        command_sha256: str | None = None,
        commit_sha: str | None = None,
    ) -> dict[str, Any]:
        """Durably reserve a worker action before starting its subprocess."""
        expected = {
            "experiment_id": experiment_id,
            "rung": rung,
            "repair_number": repair_number,
            "status": "reserved",
            "command_sha256": command_sha256,
            "commit_sha": commit_sha,
        }
        with self.database.transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM experiments WHERE experiment_id=?", (experiment_id,)
            ).fetchone() is None:
                raise RepositoryError(f"unknown experiment: {experiment_id}")
            row = connection.execute(
                "SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
            if row is not None:
                if row["status"] == "reserved":
                    _require_identical(row, expected, entity=f"attempt reservation {attempt_id}")
                    return {"idempotent": True}
                completed_expected = {
                    "experiment_id": experiment_id,
                    "rung": rung,
                    "repair_number": repair_number,
                }
                if command_sha256 is not None:
                    completed_expected["command_sha256"] = command_sha256
                if commit_sha is not None:
                    completed_expected["commit_sha"] = commit_sha
                _require_identical(
                    row,
                    completed_expected,
                    entity=f"completed attempt reservation {attempt_id}",
                )
                return {"idempotent": True, "completed": True}
            connection.execute(
                "INSERT INTO attempts(attempt_id,experiment_id,rung,repair_number,status,command_sha256,"
                "commit_sha,started_at) VALUES(?,?,?,?,?,?,?,?)",
                (
                    attempt_id,
                    experiment_id,
                    rung,
                    repair_number,
                    "reserved",
                    command_sha256,
                    commit_sha,
                    utc_now(),
                ),
            )
            return {"idempotent": False}

    def reserve_experiment_repair(
        self,
        *,
        experiment_id: str,
        phase: str,
        failure_status: AttemptStatus,
        plan: dict[str, Any],
        maximum: int = 2,
    ) -> dict[str, Any]:
        """Reserve one repair from the experiment-wide allowance.

        A currently incomplete reservation is returned idempotently so a killed
        coordinator resumes the same repair instead of consuming another slot.
        """

        with self.database.transaction() as connection:
            experiment = connection.execute(
                "SELECT run_id FROM experiments WHERE experiment_id=?", (experiment_id,)
            ).fetchone()
            if experiment is None:
                raise RepositoryError(f"unknown experiment: {experiment_id}")
            incomplete = connection.execute(
                "SELECT * FROM experiment_repairs WHERE experiment_id=? AND completed_at IS NULL "
                "ORDER BY repair_number DESC LIMIT 1",
                (experiment_id,),
            ).fetchone()
            plan_json = _canonical_json(plan)
            if incomplete is not None:
                _require_identical(
                    incomplete,
                    {
                        "phase": phase,
                        "failure_status": failure_status,
                        "plan_json": plan_json,
                    },
                    entity=f"incomplete repair {incomplete['repair_id']}",
                )
                return {
                    "idempotent": True,
                    "repair_id": incomplete["repair_id"],
                    "repair_number": int(incomplete["repair_number"]),
                }
            used = int(
                connection.execute(
                    "SELECT COUNT(*) FROM experiment_repairs WHERE experiment_id=?",
                    (experiment_id,),
                ).fetchone()[0]
            )
            if used >= maximum:
                raise RepositoryError(f"experiment repair limit {maximum} reached")
            number = used + 1
            repair_id = f"{experiment_id}:repair:{number}"
            connection.execute(
                "INSERT INTO experiment_repairs(repair_id,experiment_id,repair_number,phase,"
                "failure_status,plan_json,created_at) VALUES(?,?,?,?,?,?,?)",
                (
                    repair_id,
                    experiment_id,
                    number,
                    phase,
                    failure_status,
                    plan_json,
                    utc_now(),
                ),
            )
            self._event(
                connection,
                run_id=experiment["run_id"],
                event_type="repair.reserved",
                aggregate_id=repair_id,
                payload={
                    "experiment_id": experiment_id,
                    "repair_number": number,
                    "phase": phase,
                    "failure_status": failure_status,
                    "plan": plan,
                },
            )
            return {"idempotent": False, "repair_id": repair_id, "repair_number": number}

    def complete_experiment_repair(
        self,
        repair_id: str,
        *,
        evidence_artifact_ids: list[str],
    ) -> dict[str, Any]:
        evidence_json = _canonical_json(sorted(set(evidence_artifact_ids)))
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT repair.*,experiment.run_id FROM experiment_repairs repair "
                "JOIN experiments experiment ON experiment.experiment_id=repair.experiment_id "
                "WHERE repair.repair_id=?",
                (repair_id,),
            ).fetchone()
            if row is None:
                raise RepositoryError(f"unknown repair: {repair_id}")
            if row["completed_at"] is not None:
                _require_identical(
                    row,
                    {"evidence_json": evidence_json},
                    entity=f"repair completion {repair_id}",
                )
                return {"idempotent": True}
            connection.execute(
                "UPDATE experiment_repairs SET evidence_json=?,completed_at=? WHERE repair_id=?",
                (evidence_json, utc_now(), repair_id),
            )
            self._event(
                connection,
                run_id=row["run_id"],
                event_type="repair.completed",
                aggregate_id=repair_id,
                payload={"evidence_artifact_ids": json.loads(evidence_json)},
            )
            return {"idempotent": False}

    def apply_experiment_repair_revision(
        self,
        repair_id: str,
        *,
        repaired_commit_sha: str,
        effective_config_artifact_id: str,
    ) -> dict[str, Any]:
        """Atomically make one immutable repair revision the experiment's effective provenance."""

        if not repaired_commit_sha:
            raise RepositoryError("repaired commit SHA must be non-empty")
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT repair.*,experiment.run_id,experiment.state,experiment.commit_sha,"
                "experiment.config_sha256 FROM experiment_repairs repair JOIN experiments experiment "
                "ON experiment.experiment_id=repair.experiment_id WHERE repair.repair_id=?",
                (repair_id,),
            ).fetchone()
            if row is None:
                raise RepositoryError(f"unknown repair: {repair_id}")
            artifact = connection.execute(
                "SELECT artifact.artifact_id,artifact.kind,artifact.sha256 FROM artifacts artifact "
                "JOIN artifact_links link ON link.artifact_id=artifact.artifact_id "
                "WHERE artifact.artifact_id=? AND link.experiment_id=? LIMIT 1",
                (effective_config_artifact_id, row["experiment_id"]),
            ).fetchone()
            if artifact is None:
                raise RepositoryError("repair config artifact is not linked to the experiment")
            if artifact["kind"] != "repaired_experiment_config":
                raise RepositoryError("repair revision requires a repaired_experiment_config artifact")
            expected = {
                "repaired_commit_sha": repaired_commit_sha,
                "repaired_config_sha256": artifact["sha256"],
                "effective_config_artifact_id": effective_config_artifact_id,
            }
            if row["repaired_commit_sha"] is not None:
                _require_identical(row, expected, entity=f"repair revision {repair_id}")
                if (
                    row["commit_sha"] != repaired_commit_sha
                    or row["config_sha256"] != artifact["sha256"]
                ):
                    raise RepositoryError(
                        f"experiment provenance drifted after repair revision {repair_id}"
                    )
                return {
                    "idempotent": True,
                    "commit_sha": repaired_commit_sha,
                    "config_sha256": artifact["sha256"],
                }
            if row["state"] != ExperimentState.REPAIRING:
                raise RepositoryError(
                    f"repair revision {repair_id} requires experiment state REPAIRING"
                )
            connection.execute(
                "UPDATE experiment_repairs SET previous_commit_sha=?,repaired_commit_sha=?,"
                "previous_config_sha256=?,repaired_config_sha256=?,"
                "effective_config_artifact_id=? WHERE repair_id=?",
                (
                    row["commit_sha"],
                    repaired_commit_sha,
                    row["config_sha256"],
                    artifact["sha256"],
                    effective_config_artifact_id,
                    repair_id,
                ),
            )
            connection.execute(
                "UPDATE experiments SET commit_sha=?,config_sha256=?,updated_at=? "
                "WHERE experiment_id=?",
                (
                    repaired_commit_sha,
                    artifact["sha256"],
                    utc_now(),
                    row["experiment_id"],
                ),
            )
            self._event(
                connection,
                run_id=row["run_id"],
                event_type="repair.revision_applied",
                aggregate_id=repair_id,
                payload={
                    "experiment_id": row["experiment_id"],
                    "previous_commit_sha": row["commit_sha"],
                    "repaired_commit_sha": repaired_commit_sha,
                    "previous_config_sha256": row["config_sha256"],
                    "repaired_config_sha256": artifact["sha256"],
                    "effective_config_artifact_id": effective_config_artifact_id,
                },
            )
            return {
                "idempotent": False,
                "commit_sha": repaired_commit_sha,
                "config_sha256": artifact["sha256"],
            }

    def experiment_repairs_used(self, experiment_id: str) -> int:
        with self.database.connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM experiment_repairs WHERE experiment_id=?",
                    (experiment_id,),
                ).fetchone()[0]
            )

    def record_attempt(self, result: RunResult, *, rung: str, repair_number: int = 0) -> None:
        stdout_id = next((item.artifact_id for item in result.artifacts if item.kind == "stdout"), None)
        stderr_id = next((item.artifact_id for item in result.artifacts if item.kind == "stderr"), None)
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT run_id FROM experiments WHERE experiment_id=?", (result.experiment_id,)
            ).fetchone()
            if row is None:
                raise RepositoryError(f"unknown experiment: {result.experiment_id}")
            expected = {
                "experiment_id": result.experiment_id,
                "rung": rung,
                "repair_number": repair_number,
                "status": result.status,
                "command_sha256": result.command_sha256,
                "commit_sha": result.commit_sha,
                "wall_seconds": result.wall_seconds,
                "exit_code": result.exit_code,
                "signal": result.signal,
                "error_type": result.error_type,
                "error_summary": result.error_summary,
                "stdout_artifact_id": stdout_id,
                "stderr_artifact_id": stderr_id,
            }
            prior = connection.execute(
                "SELECT * FROM attempts WHERE attempt_id=?", (result.attempt_id,)
            ).fetchone()
            if prior is not None and prior["status"] != "reserved":
                _require_identical(prior, expected, entity=f"attempt {result.attempt_id}")
                return
            if prior is not None:
                reservation_expected = {
                    "experiment_id": result.experiment_id,
                    "rung": rung,
                    "repair_number": repair_number,
                }
                if prior["command_sha256"] is not None:
                    reservation_expected["command_sha256"] = result.command_sha256
                if prior["commit_sha"] is not None:
                    reservation_expected["commit_sha"] = result.commit_sha
                _require_identical(
                    prior,
                    reservation_expected,
                    entity=f"attempt reservation {result.attempt_id}",
                )
                connection.execute(
                    "UPDATE attempts SET status=?,command_sha256=?,commit_sha=?,ended_at=?,"
                    "wall_seconds=?,exit_code=?,signal=?,"
                    "error_type=?,error_summary=?,stdout_artifact_id=?,stderr_artifact_id=? "
                    "WHERE attempt_id=?",
                    (
                        result.status,
                        result.command_sha256,
                        result.commit_sha,
                        utc_now(),
                        result.wall_seconds,
                        result.exit_code,
                        result.signal,
                        result.error_type,
                        result.error_summary,
                        stdout_id,
                        stderr_id,
                        result.attempt_id,
                    ),
                )
            else:
                connection.execute(
                    "INSERT INTO attempts(attempt_id,experiment_id,rung,repair_number,status,"
                    "command_sha256,commit_sha,ended_at,wall_seconds,exit_code,signal,error_type,error_summary,"
                    "stdout_artifact_id,stderr_artifact_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        result.attempt_id,
                        result.experiment_id,
                        rung,
                        repair_number,
                        result.status,
                        result.command_sha256,
                        result.commit_sha,
                        utc_now(),
                        result.wall_seconds,
                        result.exit_code,
                        result.signal,
                        result.error_type,
                        result.error_summary,
                        stdout_id,
                        stderr_id,
                    ),
                )
            connection.execute(
                "INSERT INTO resource_usage(run_id,experiment_id,attempt_id,scope,wall_seconds,"
                "cpu_user_seconds,cpu_system_seconds,peak_rss_bytes,gpu_seconds,resource_key) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    row["run_id"],
                    result.experiment_id,
                    result.attempt_id,
                    "worker",
                    result.wall_seconds,
                    result.cpu_user_seconds,
                    result.cpu_system_seconds,
                    result.peak_rss_bytes,
                    result.gpu_seconds,
                    f"attempt:{result.attempt_id}:worker",
                ),
            )
            self._event(
                connection,
                run_id=row["run_id"],
                event_type="attempt.completed",
                aggregate_id=result.attempt_id,
                payload={
                    "experiment_id": result.experiment_id,
                    "rung": rung,
                    "repair_number": repair_number,
                    "status": result.status,
                    "wall_seconds": result.wall_seconds,
                },
            )

    def record_metrics(
        self,
        experiment_id: str,
        metrics: Metrics,
        attempt_id: str | None = None,
        *,
        max_official_evaluations: int = 50,
    ) -> None:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT e.run_id,r.official_evaluation_count FROM experiments e JOIN runs r "
                "ON e.run_id=r.run_id WHERE e.experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if row is None:
                raise RepositoryError(f"unknown experiment: {experiment_id}")
            prior = connection.execute(
                "SELECT * FROM metrics WHERE experiment_id=? AND split=? AND fold IS ? AND seed IS ?",
                (experiment_id, metrics.split, metrics.fold, metrics.seed),
            ).fetchone()
            expected = {
                "attempt_id": attempt_id,
                "evaluator_sha256": metrics.evaluator_sha256,
                "gauc": metrics.GAUC,
                "ndcg5": metrics.ndcg5,
                "primary_score": metrics.primary,
                "primary_units": metric_units(metrics.primary),
                "rows": metrics.rows,
                "users": metrics.users,
            }
            if prior is not None:
                _require_identical(
                    prior,
                    expected,
                    entity=(
                        f"metrics {experiment_id}/{metrics.split}/{metrics.fold}/{metrics.seed}"
                    ),
                )
                return
            if metrics.split == "valid" and int(row["official_evaluation_count"]) >= max_official_evaluations:
                raise RepositoryError(
                    f"official evaluation cap reached: {max_official_evaluations}"
                )
            connection.execute(
                "INSERT INTO metrics(experiment_id,attempt_id,split,fold,seed,evaluator_sha256,gauc,ndcg5,"
                "primary_score,primary_units,rows,users) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    experiment_id,
                    attempt_id,
                    metrics.split,
                    metrics.fold,
                    metrics.seed,
                    metrics.evaluator_sha256,
                    metrics.GAUC,
                    metrics.ndcg5,
                    metrics.primary,
                    expected["primary_units"],
                    metrics.rows,
                    metrics.users,
                ),
            )
            if metrics.split == "valid":
                connection.execute(
                    "UPDATE runs SET official_evaluation_count=official_evaluation_count+1, updated_at=? "
                    "WHERE run_id=?",
                    (utc_now(), row["run_id"]),
                )
            self._event(
                connection,
                run_id=row["run_id"],
                event_type="metrics.recorded",
                aggregate_id=experiment_id,
                payload=metrics.model_dump(mode="json", by_alias=True),
            )

    def reject_non_improving(
        self,
        *,
        run_id: str,
        experiment_id: str,
        expected_state: ExperimentState,
        reason: str,
        patience: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Atomically count a rejected hypothesis toward the convergence streak."""
        require_experiment_transition(expected_state, ExperimentState.REJECTED)
        with self.database.transaction() as connection:
            duplicate = connection.execute(
                "SELECT * FROM transitions WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
            if duplicate:
                if (
                    duplicate["experiment_id"] != experiment_id
                    or duplicate["from_state"] != expected_state
                    or duplicate["to_state"] != ExperimentState.REJECTED
                ):
                    raise RepositoryError(f"conflicting replay for rejection {idempotency_key}")
                if json.loads(duplicate["payload_json"]).get("reason") != reason:
                    raise RepositoryError(f"conflicting replay for rejection {idempotency_key}")
                return {"idempotent": True}
            run = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            experiment = connection.execute(
                "SELECT state FROM experiments WHERE experiment_id=? AND run_id=?",
                (experiment_id, run_id),
            ).fetchone()
            if run is None or experiment is None:
                raise RepositoryError("run or experiment missing for rejection")
            if ExperimentState(experiment["state"]) != expected_state:
                raise RepositoryError(
                    f"experiment {experiment_id} is not in expected state {expected_state}"
                )
            streak = int(run["non_improvement_streak"]) + 1
            converged = streak >= patience
            now = utc_now()
            payload = json.dumps({"reason": reason, "convergence_streak": streak})
            connection.execute(
                "UPDATE experiments SET state=?,terminal_reason=?,updated_at=? WHERE experiment_id=?",
                (ExperimentState.REJECTED, reason, now, experiment_id),
            )
            connection.execute(
                "INSERT INTO convergence_transactions(experiment_id,run_id,outcome,delta_units,created_at) "
                "VALUES(?,?,?,?,?)",
                (experiment_id, run_id, "rejected", None, now),
            )
            connection.execute(
                "INSERT INTO transitions(experiment_id,from_state,to_state,payload_json,created_at,"
                "idempotency_key) VALUES(?,?,?,?,?,?)",
                (
                    experiment_id,
                    expected_state,
                    ExperimentState.REJECTED,
                    payload,
                    now,
                    idempotency_key,
                ),
            )
            connection.execute(
                "UPDATE runs SET non_improvement_streak=?,updated_at=?,stop_reason=CASE WHEN ? "
                "THEN 'epsilon_plateau' ELSE stop_reason END WHERE run_id=?",
                (streak, now, int(converged), run_id),
            )
            self._event(
                connection,
                run_id=run_id,
                event_type="candidate.rejected",
                aggregate_id=experiment_id,
                payload={
                    "reason": reason,
                    "non_improvement_streak": streak,
                    "converged": converged,
                },
            )
            return {
                "idempotent": False,
                "non_improvement_streak": streak,
                "converged": converged,
            }

    def reject_candidate(
        self,
        *,
        run_id: str,
        experiment_id: str,
        expected_state: ExperimentState,
        reason: str,
        patience: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Reject a cheap/full candidate and count the hypothesis exactly once."""

        require_experiment_transition(expected_state, ExperimentState.REJECTED)
        with self.database.transaction() as connection:
            duplicate = connection.execute(
                "SELECT * FROM transitions WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
            if duplicate is not None:
                if (
                    duplicate["experiment_id"] != experiment_id
                    or duplicate["from_state"] != expected_state
                    or duplicate["to_state"] != ExperimentState.REJECTED
                    or json.loads(duplicate["payload_json"]).get("reason") != reason
                ):
                    raise RepositoryError(f"conflicting replay for rejection {idempotency_key}")
                run = connection.execute(
                    "SELECT non_improvement_streak,stop_reason FROM runs WHERE run_id=?",
                    (run_id,),
                ).fetchone()
                return {
                    "idempotent": True,
                    "non_improvement_streak": int(run["non_improvement_streak"]),
                    "converged": run["stop_reason"] == "epsilon_plateau",
                }
            row = connection.execute(
                "SELECT state FROM experiments WHERE run_id=? AND experiment_id=?",
                (run_id, experiment_id),
            ).fetchone()
            if row is None or ExperimentState(row["state"]) != expected_state:
                raise RepositoryError("candidate is not in the expected rejection state")
            now = utc_now()
            run = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            streak = int(run["non_improvement_streak"]) + 1
            converged = streak >= patience
            payload = _canonical_json(
                {
                    "reason": reason,
                    "convergence_counted": True,
                    "convergence_streak": streak,
                }
            )
            connection.execute(
                "UPDATE experiments SET state=?,terminal_reason=?,updated_at=? WHERE experiment_id=?",
                (ExperimentState.REJECTED, reason, now, experiment_id),
            )
            connection.execute(
                "INSERT INTO transitions(experiment_id,from_state,to_state,payload_json,created_at,"
                "idempotency_key) VALUES(?,?,?,?,?,?)",
                (
                    experiment_id,
                    expected_state,
                    ExperimentState.REJECTED,
                    payload,
                    now,
                    idempotency_key,
                ),
            )
            connection.execute(
                "INSERT INTO convergence_transactions(experiment_id,run_id,outcome,delta_units,created_at) "
                "VALUES(?,?,?,?,?)",
                (experiment_id, run_id, "rejected_before_official", None, now),
            )
            connection.execute(
                "UPDATE runs SET non_improvement_streak=?,updated_at=?,stop_reason=CASE WHEN ? "
                "THEN 'epsilon_plateau' ELSE stop_reason END WHERE run_id=?",
                (streak, now, int(converged), run_id),
            )
            self._event(
                connection,
                run_id=run_id,
                event_type="candidate.rejected",
                aggregate_id=experiment_id,
                payload={
                    "reason": reason,
                    "convergence_counted": True,
                    "convergence_streak": streak,
                },
            )
            return {
                "idempotent": False,
                "non_improvement_streak": streak,
                "converged": converged,
            }

    def count_failed_transaction(
        self,
        *,
        run_id: str,
        experiment_id: str,
        reason: str,
        patience: int,
    ) -> dict[str, Any]:
        """Count a terminal failure once without changing the incumbent."""

        with self.database.transaction() as connection:
            experiment = connection.execute(
                "SELECT state FROM experiments WHERE run_id=? AND experiment_id=?",
                (run_id, experiment_id),
            ).fetchone()
            if experiment is None or ExperimentState(experiment["state"]) != ExperimentState.FAILED_FINAL:
                raise RepositoryError("only FAILED_FINAL experiments can count as failed transactions")
            prior = connection.execute(
                "SELECT * FROM convergence_transactions WHERE experiment_id=?", (experiment_id,)
            ).fetchone()
            run = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if prior is not None:
                return {
                    "idempotent": True,
                    "non_improvement_streak": int(run["non_improvement_streak"]),
                    "converged": run["stop_reason"] == "epsilon_plateau",
                }
            streak = int(run["non_improvement_streak"]) + 1
            converged = streak >= patience
            now = utc_now()
            connection.execute(
                "INSERT INTO convergence_transactions(experiment_id,run_id,outcome,delta_units,created_at) "
                "VALUES(?,?,?,?,?)",
                (experiment_id, run_id, "failed_final", None, now),
            )
            connection.execute(
                "UPDATE runs SET non_improvement_streak=?,updated_at=?,stop_reason=CASE WHEN ? "
                "THEN 'epsilon_plateau' ELSE stop_reason END WHERE run_id=?",
                (streak, now, int(converged), run_id),
            )
            self._event(
                connection,
                run_id=run_id,
                event_type="candidate.failed",
                aggregate_id=experiment_id,
                payload={
                    "reason": reason,
                    "convergence_streak": streak,
                    "converged": converged,
                },
            )
            return {
                "idempotent": False,
                "non_improvement_streak": streak,
                "converged": converged,
            }

    def promote_search_candidate(
        self,
        *,
        run_id: str,
        experiment_id: str,
        primary: float,
        evidence_artifact_ids: list[str],
        epsilon: float,
        patience: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Promote a validation-best search candidate without building a test submission."""

        evidence = sorted(set(evidence_artifact_ids))
        evidence_json = _canonical_json(evidence)
        with self.database.transaction() as connection:
            duplicate = connection.execute(
                "SELECT * FROM transitions WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
            if duplicate is not None:
                if (
                    duplicate["experiment_id"] != experiment_id
                    or duplicate["from_state"] != ExperimentState.OFFICIAL_VALID_COMPLETE
                    or duplicate["to_state"]
                    not in {ExperimentState.PROMOTED, ExperimentState.REJECTED}
                    or json.loads(duplicate["payload_json"]).get("primary") != primary
                ):
                    raise RepositoryError(
                        f"conflicting replay for search promotion {idempotency_key}"
                    )
                return {"idempotent": True}
            run = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            experiment = connection.execute(
                "SELECT * FROM experiments WHERE run_id=? AND experiment_id=?",
                (run_id, experiment_id),
            ).fetchone()
            if run is None or experiment is None:
                raise RepositoryError("run or candidate missing for search promotion")
            if ExperimentState(experiment["state"]) != ExperimentState.OFFICIAL_VALID_COMPLETE:
                raise RepositoryError("candidate must complete official validation before promotion")
            if evidence:
                placeholders = ",".join("?" for _ in evidence)
                linked = connection.execute(
                    f"SELECT COUNT(DISTINCT artifact_id) AS n FROM artifact_links "
                    f"WHERE experiment_id=? AND artifact_id IN ({placeholders})",
                    (experiment_id, *evidence),
                ).fetchone()["n"]
                if linked != len(evidence):
                    raise RepositoryError("search promotion cites missing candidate evidence")
            update = update_metric_trackers(
                previous_best_units=run["best_primary_units"],
                candidate_primary=primary,
                previous_streak=int(run["non_improvement_streak"]),
                epsilon_units=metric_units(epsilon),
                patience=patience,
            )
            target = ExperimentState.PROMOTED if update.is_new_best else ExperimentState.REJECTED
            require_experiment_transition(ExperimentState.OFFICIAL_VALID_COMPLETE, target)
            now = utc_now()
            payload_value = {
                "primary": primary,
                "delta_units": update.delta_units,
                "is_new_best": update.is_new_best,
                "convergence_streak": update.non_improvement_streak,
                "evidence_artifact_ids": evidence,
                "test_submission_created": False,
            }
            connection.execute(
                "UPDATE experiments SET state=?,updated_at=?,terminal_reason=? WHERE experiment_id=?",
                (
                    target,
                    now,
                    None if update.is_new_best else "not better than validation incumbent",
                    experiment_id,
                ),
            )
            connection.execute(
                "INSERT INTO transitions(experiment_id,from_state,to_state,payload_json,created_at,"
                "idempotency_key) VALUES(?,?,?,?,?,?)",
                (
                    experiment_id,
                    ExperimentState.OFFICIAL_VALID_COMPLETE,
                    target,
                    _canonical_json(payload_value),
                    now,
                    idempotency_key,
                ),
            )
            if update.is_new_best:
                connection.execute(
                    "INSERT INTO search_promotions(run_id,previous_experiment_id,experiment_id,"
                    "primary_units,evidence_json,created_at) VALUES(?,?,?,?,?,?)",
                    (
                        run_id,
                        run["search_champion_experiment_id"],
                        experiment_id,
                        update.best_primary_units,
                        evidence_json,
                        now,
                    ),
                )
            connection.execute(
                "INSERT INTO convergence_transactions(experiment_id,run_id,outcome,delta_units,created_at) "
                "VALUES(?,?,?,?,?)",
                (
                    experiment_id,
                    run_id,
                    "promoted" if update.is_new_best else "official_rejected",
                    update.delta_units,
                    now,
                ),
            )
            connection.execute(
                "UPDATE runs SET best_primary_units=?,best_ever_experiment_id=CASE WHEN ? THEN ? "
                "ELSE best_ever_experiment_id END,search_champion_experiment_id=CASE WHEN ? THEN ? "
                "ELSE search_champion_experiment_id END,non_improvement_streak=?,updated_at=?,"
                "stop_reason=CASE WHEN ? THEN 'epsilon_plateau' ELSE stop_reason END WHERE run_id=?",
                (
                    update.best_primary_units,
                    int(update.is_new_best),
                    experiment_id,
                    int(update.is_new_best),
                    experiment_id,
                    update.non_improvement_streak,
                    now,
                    int(update.converged),
                    run_id,
                ),
            )
            self._event(
                connection,
                run_id=run_id,
                event_type="search_candidate.validated",
                aggregate_id=experiment_id,
                payload=payload_value,
            )
            return {
                "idempotent": False,
                "is_new_best": update.is_new_best,
                "converged": update.converged,
                "non_improvement_streak": update.non_improvement_streak,
                "best_primary_units": update.best_primary_units,
            }

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise RepositoryError(f"unknown run: {run_id}")
            return dict(row)

    def record_llm_call(
        self,
        *,
        call_id: str,
        run_id: str,
        experiment_id: str | None,
        role: str,
        provider: str,
        model: str,
        request_artifact_id: str | None,
        response_artifact_id: str | None,
        schema_valid: bool,
        input_tokens: int,
        output_tokens: int,
        wall_seconds: float,
        request_id: str | None = None,
        error: str | None = None,
    ) -> None:
        with self.database.transaction() as connection:
            expected = {
                "run_id": run_id,
                "experiment_id": experiment_id,
                "role": role,
                "provider": provider,
                "model": model,
                "request_id": request_id,
                "request_artifact_id": request_artifact_id,
                "response_artifact_id": response_artifact_id,
                "schema_valid": int(schema_valid),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "wall_seconds": wall_seconds,
                "error": error,
            }
            prior = connection.execute(
                "SELECT * FROM llm_calls WHERE call_id=?", (call_id,)
            ).fetchone()
            if prior is not None:
                _require_identical(prior, expected, entity=f"LLM call {call_id}")
                return
            connection.execute(
                "INSERT INTO llm_calls(call_id,run_id,experiment_id,role,provider,model,request_id,"
                "request_artifact_id,response_artifact_id,schema_valid,input_tokens,output_tokens,"
                "wall_seconds,error) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    call_id,
                    run_id,
                    experiment_id,
                    role,
                    provider,
                    model,
                    request_id,
                    request_artifact_id,
                    response_artifact_id,
                    int(schema_valid),
                    input_tokens,
                    output_tokens,
                    wall_seconds,
                    error,
                ),
            )
            connection.execute(
                "INSERT INTO resource_usage(run_id,experiment_id,scope,llm_tokens,wall_seconds,"
                "resource_key) VALUES(?,?,?,?,?,?)",
                (
                    run_id,
                    experiment_id,
                    "llm",
                    input_tokens + output_tokens,
                    wall_seconds,
                    f"llm:{call_id}",
                ),
            )
            self._event(
                connection,
                run_id=run_id,
                event_type="llm.completed",
                aggregate_id=call_id,
                payload={
                    "experiment_id": experiment_id,
                    "role": role,
                    "provider": provider,
                    "model": model,
                    "schema_valid": schema_valid,
                    "tokens": input_tokens + output_tokens,
                    "error": error,
                },
            )

    def record_resource_usage(
        self,
        *,
        resource_key: str,
        run_id: str,
        experiment_id: str | None = None,
        attempt_id: str | None = None,
        scope: str,
        wall_seconds: float = 0,
        cpu_user_seconds: float = 0,
        cpu_system_seconds: float = 0,
        peak_rss_bytes: int = 0,
        gpu_seconds: float = 0,
        llm_tokens: int = 0,
    ) -> dict[str, Any]:
        expected = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "attempt_id": attempt_id,
            "scope": scope,
            "wall_seconds": wall_seconds,
            "cpu_user_seconds": cpu_user_seconds,
            "cpu_system_seconds": cpu_system_seconds,
            "peak_rss_bytes": peak_rss_bytes,
            "gpu_seconds": gpu_seconds,
            "llm_tokens": llm_tokens,
        }
        with self.database.transaction() as connection:
            prior = connection.execute(
                "SELECT * FROM resource_usage WHERE resource_key=?", (resource_key,)
            ).fetchone()
            if prior is not None:
                _require_identical(prior, expected, entity=f"resource usage {resource_key}")
                return {"idempotent": True}
            connection.execute(
                "INSERT INTO resource_usage(run_id,experiment_id,attempt_id,scope,wall_seconds,"
                "cpu_user_seconds,cpu_system_seconds,peak_rss_bytes,gpu_seconds,llm_tokens,resource_key) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    experiment_id,
                    attempt_id,
                    scope,
                    wall_seconds,
                    cpu_user_seconds,
                    cpu_system_seconds,
                    peak_rss_bytes,
                    gpu_seconds,
                    llm_tokens,
                    resource_key,
                ),
            )
            return {"idempotent": False}

    def record_lesson(
        self,
        *,
        lesson_id: str,
        run_id: str,
        experiment_id: str,
        scope: str,
        lesson: str,
        evidence_artifact_ids: list[str],
    ) -> None:
        with self.database.transaction() as connection:
            known = connection.execute(
                "SELECT COUNT(DISTINCT artifact_id) AS n FROM artifact_links WHERE experiment_id=? "
                "AND artifact_id IN "
                f"({','.join('?' for _ in evidence_artifact_ids)})",
                (experiment_id, *evidence_artifact_ids),
            ).fetchone()["n"] if evidence_artifact_ids else 0
            if known != len(evidence_artifact_ids):
                raise RepositoryError("lesson cites missing evidence artifacts")
            evidence_json = _canonical_json(evidence_artifact_ids)
            prior = connection.execute(
                "SELECT * FROM lessons WHERE lesson_id=?", (lesson_id,)
            ).fetchone()
            if prior is not None:
                _require_identical(
                    prior,
                    {
                        "run_id": run_id,
                        "experiment_id": experiment_id,
                        "scope": scope,
                        "lesson": lesson,
                        "evidence_json": evidence_json,
                    },
                    entity=f"lesson {lesson_id}",
                )
                return
            connection.execute(
                "INSERT INTO lessons(lesson_id,run_id,experiment_id,scope,lesson,evidence_json,created_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    lesson_id,
                    run_id,
                    experiment_id,
                    scope,
                    lesson,
                    evidence_json,
                    utc_now(),
                ),
            )
            self._event(
                connection,
                run_id=run_id,
                event_type="lesson.recorded",
                aggregate_id=lesson_id,
                payload={"experiment_id": experiment_id, "scope": scope, "evidence": evidence_artifact_ids},
            )

    def record_intervention(
        self,
        *,
        intervention_id: str,
        run_id: str,
        actor: str,
        action: str,
        reason: str,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        with self.database.transaction() as connection:
            evidence_json = _canonical_json(evidence or {})
            prior = connection.execute(
                "SELECT * FROM interventions WHERE intervention_id=?", (intervention_id,)
            ).fetchone()
            if prior is not None:
                _require_identical(
                    prior,
                    {
                        "run_id": run_id,
                        "actor": actor,
                        "action": action,
                        "reason": reason,
                        "evidence_json": evidence_json,
                    },
                    entity=f"intervention {intervention_id}",
                )
                return
            connection.execute(
                "INSERT INTO interventions(intervention_id,run_id,actor,action,reason,evidence_json,created_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    intervention_id,
                    run_id,
                    actor,
                    action,
                    reason,
                    evidence_json,
                    utc_now(),
                ),
            )
            self._event(
                connection,
                run_id=run_id,
                event_type="intervention.recorded",
                aggregate_id=intervention_id,
                payload={"actor": actor, "action": action, "reason": reason},
            )

    def promote_validated(
        self,
        *,
        run_id: str,
        experiment_id: str,
        primary: float,
        epsilon: float,
        patience: int,
        checkpoint_artifact_id: str,
        prediction_artifact_id: str,
        submission_artifact_id: str,
        validator_artifact_id: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Atomically update convergence and promote only a fully validated, better candidate."""
        with self.database.transaction() as connection:
            duplicate = connection.execute(
                "SELECT * FROM transitions WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
            if duplicate:
                if (
                    duplicate["experiment_id"] != experiment_id
                    or duplicate["from_state"] != ExperimentState.SUBMISSION_VALID
                    or duplicate["to_state"] not in {
                        ExperimentState.PROMOTED,
                        ExperimentState.REJECTED,
                    }
                ):
                    raise RepositoryError(f"conflicting replay for promotion {idempotency_key}")
                if json.loads(duplicate["payload_json"]).get("primary") != primary:
                    raise RepositoryError(f"conflicting replay for promotion {idempotency_key}")
                return {"idempotent": True}
            run = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            experiment = connection.execute(
                "SELECT * FROM experiments WHERE experiment_id=? AND run_id=?",
                (experiment_id, run_id),
            ).fetchone()
            if run is None or experiment is None:
                raise RepositoryError("run or experiment not found for promotion")
            if ExperimentState(experiment["state"]) != ExperimentState.SUBMISSION_VALID:
                raise RepositoryError("candidate must be SUBMISSION_VALID before promotion")
            artifact_ids = (
                checkpoint_artifact_id,
                prediction_artifact_id,
                submission_artifact_id,
                validator_artifact_id,
            )
            placeholders = ",".join("?" for _ in artifact_ids)
            count = connection.execute(
                f"SELECT COUNT(DISTINCT artifact_id) AS n FROM artifact_links "
                f"WHERE artifact_id IN ({placeholders}) AND experiment_id=?",
                (*artifact_ids, experiment_id),
            ).fetchone()["n"]
            if count != len(artifact_ids):
                raise RepositoryError("promotion artifacts are missing or belong to another experiment")
            update = update_metric_trackers(
                previous_best_units=run["best_primary_units"],
                candidate_primary=primary,
                previous_streak=int(run["non_improvement_streak"]),
                epsilon_units=metric_units(epsilon),
                patience=patience,
            )
            now = utc_now()
            target_state = ExperimentState.PROMOTED if update.is_new_best else ExperimentState.REJECTED
            require_experiment_transition(ExperimentState.SUBMISSION_VALID, target_state)
            connection.execute(
                "UPDATE experiments SET state=?,updated_at=?,terminal_reason=? WHERE experiment_id=?",
                (
                    target_state,
                    now,
                    None if update.is_new_best else "not better than validated incumbent",
                    experiment_id,
                ),
            )
            connection.execute(
                "INSERT INTO transitions(experiment_id,from_state,to_state,payload_json,created_at,idempotency_key) "
                "VALUES(?,?,?,?,?,?)",
                (
                    experiment_id,
                    ExperimentState.SUBMISSION_VALID,
                    target_state,
                    json.dumps({"primary": primary, "delta_units": update.delta_units}),
                    now,
                    idempotency_key,
                ),
            )
            if update.is_new_best:
                connection.execute(
                    "INSERT INTO promotions(run_id,previous_experiment_id,experiment_id,primary_units,"
                    "checkpoint_artifact_id,prediction_artifact_id,submission_artifact_id,"
                    "validator_artifact_id,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        run_id,
                        run["best_ever_experiment_id"],
                        experiment_id,
                        update.best_primary_units,
                        checkpoint_artifact_id,
                        prediction_artifact_id,
                        submission_artifact_id,
                        validator_artifact_id,
                        now,
                    ),
                )
            connection.execute(
                "UPDATE runs SET best_primary_units=?,best_ever_experiment_id=CASE WHEN ? THEN ? "
                "ELSE best_ever_experiment_id END,non_improvement_streak=?,updated_at=?,"
                "stop_reason=CASE WHEN ? THEN 'epsilon_plateau' ELSE stop_reason END WHERE run_id=?",
                (
                    update.best_primary_units,
                    int(update.is_new_best),
                    experiment_id,
                    update.non_improvement_streak,
                    now,
                    int(update.converged),
                    run_id,
                ),
            )
            self._event(
                connection,
                run_id=run_id,
                event_type="candidate.validated",
                aggregate_id=experiment_id,
                payload={
                    "primary": primary,
                    "delta_units": update.delta_units,
                    "is_new_best": update.is_new_best,
                    "streak": update.non_improvement_streak,
                    "converged": update.converged,
                },
            )
            return {
                "idempotent": False,
                "is_new_best": update.is_new_best,
                "converged": update.converged,
                "non_improvement_streak": update.non_improvement_streak,
                "best_primary_units": update.best_primary_units,
            }

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,)
            ).fetchone()
            if row is None:
                raise RepositoryError(f"unknown experiment: {experiment_id}")
            return dict(row)
