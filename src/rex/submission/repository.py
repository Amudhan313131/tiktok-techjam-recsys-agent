"""Persistence and immutable-source access for final-submission jobs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterator, Mapping

from rex.data.manifest import canonical_json_bytes, sha256_file


class SubmissionRepositoryError(RuntimeError):
    """A durable submission record is missing, conflicting, or invalid."""


class SubmissionState(StrEnum):
    CREATED = "CREATED"
    SOURCE_VERIFIED = "SOURCE_VERIFIED"
    WORKTREE_READY = "WORKTREE_READY"
    PREDICTING = "PREDICTING"
    PREDICTED = "PREDICTED"
    CSV_BUILT = "CSV_BUILT"
    FIRST_CHECK_VALID = "FIRST_CHECK_VALID"
    STAGING = "STAGING"
    SECOND_CHECK_VALID = "SECOND_CHECK_VALID"
    SEALED = "SEALED"
    READY_FOR_HANDOFF = "READY_FOR_HANDOFF"
    HANDOFF_IN_PROGRESS = "HANDOFF_IN_PROGRESS"
    HANDED_OFF = "HANDED_OFF"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


_NEXT: dict[SubmissionState, frozenset[SubmissionState]] = {
    SubmissionState.CREATED: frozenset({SubmissionState.SOURCE_VERIFIED}),
    SubmissionState.SOURCE_VERIFIED: frozenset({SubmissionState.WORKTREE_READY}),
    SubmissionState.WORKTREE_READY: frozenset({SubmissionState.PREDICTING}),
    SubmissionState.PREDICTING: frozenset({SubmissionState.PREDICTED}),
    SubmissionState.PREDICTED: frozenset({SubmissionState.CSV_BUILT}),
    SubmissionState.CSV_BUILT: frozenset({SubmissionState.FIRST_CHECK_VALID}),
    SubmissionState.FIRST_CHECK_VALID: frozenset({SubmissionState.STAGING}),
    SubmissionState.STAGING: frozenset({SubmissionState.SECOND_CHECK_VALID}),
    SubmissionState.SECOND_CHECK_VALID: frozenset({SubmissionState.SEALED}),
    SubmissionState.SEALED: frozenset({SubmissionState.READY_FOR_HANDOFF}),
    SubmissionState.READY_FOR_HANDOFF: frozenset({SubmissionState.HANDOFF_IN_PROGRESS}),
    SubmissionState.HANDOFF_IN_PROGRESS: frozenset({SubmissionState.HANDED_OFF}),
    SubmissionState.HANDED_OFF: frozenset(),
    SubmissionState.REJECTED: frozenset(),
    SubmissionState.FAILED: frozenset(),
}


@dataclass(frozen=True)
class SourceLocator:
    source_run_id: str
    source_database_path: Path
    source_run: Mapping[str, Any]
    source_run_fingerprint: str
    source_report_path: Path
    source_report_sha256: str
    best_valid_path: Path
    best_valid_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_fingerprint(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(row))).hexdigest()


def report_fingerprint(report: Path) -> str:
    required = {
        "events.jsonl",
        "evidence_index.json",
        "experiment_graph.json",
        "experiments.md",
        "artifact_summary.json",
        "environment_identity.json",
        "interventions.json",
        "iteration_logs.json",
        "manual_interventions.json",
        "manual_interventions.md",
        "recovery_events.json",
        "results.json",
        "resources.json",
    }
    observed: dict[str, dict[str, Any]] = {}
    for path in sorted(report.rglob("*")):
        if path.is_symlink():
            raise SubmissionRepositoryError("production report may not contain symlinks")
        if path.is_file():
            relative = path.relative_to(report).as_posix()
            observed[relative] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    missing = required.difference(observed)
    if missing:
        raise SubmissionRepositoryError(
            f"completed run report is incomplete: missing {sorted(missing)}"
        )
    return hashlib.sha256(canonical_json_bytes(observed)).hexdigest()


def discover_completed_source(
    source_database: str | Path,
    source_run_id: str,
) -> SourceLocator:
    """Read a completed production run without obtaining any write capability."""

    database = Path(source_database).resolve(strict=True)
    # mode=ro preserves least authority while still allowing SQLite to read a
    # completed run's WAL.  immutable=1 would silently ignore an uncheckpointed
    # WAL and can make a valid database appear empty.
    uri = f"file:{database.as_posix()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        try:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id=?", (source_run_id,)
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise SubmissionRepositoryError(f"cannot read production database: {error}") from error
    if row is None:
        raise SubmissionRepositoryError(f"unknown production run: {source_run_id}")
    source_run = dict(row)
    if source_run.get("state") != "COMPLETE":
        raise SubmissionRepositoryError(
            f"production run must be COMPLETE, observed {source_run.get('state')}"
        )

    manifest = database.parent / "best-valid" / "best_valid_manifest.json"
    if not manifest.is_file():
        raise SubmissionRepositoryError(f"completed run has no best-valid manifest: {manifest}")
    report = database.parent / "report"
    if not report.is_dir():
        raise SubmissionRepositoryError(f"completed run has no report directory: {report}")
    return SourceLocator(
        source_run_id=source_run_id,
        source_database_path=database,
        source_run=source_run,
        source_run_fingerprint=_row_fingerprint(source_run),
        source_report_path=report.resolve(),
        source_report_sha256=report_fingerprint(report),
        best_valid_path=manifest.resolve(),
        best_valid_sha256=sha256_file(manifest),
    )


class SubmissionRepository:
    """A standalone state store; it never connects read-write to the production DB."""

    _UPDATE_COLUMNS = frozenset(
        {
            "source_commit",
            "config_sha256",
            "incumbent_experiment_id",
            "worktree_path",
            "prediction_request_json",
            "prediction_path",
            "prediction_sha256",
            "csv_path",
            "csv_sha256",
            "staging_path",
            "sealed_path",
            "seal_sha256",
            "error_code",
            "error_summary",
        }
    )

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
        with self.connect() as connection:
            connection.executescript(schema)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def create_job(self, source: SourceLocator) -> dict[str, Any]:
        identity = f"{source.source_run_id}\0{source.best_valid_sha256}"
        job_id = "submission-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        now = utc_now()
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM submission_jobs WHERE source_run_id=? AND best_valid_sha256=?",
                (source.source_run_id, source.best_valid_sha256),
            ).fetchone()
            if existing is not None:
                expected = {
                    "job_id": job_id,
                    "source_database_path": str(source.source_database_path),
                    "source_run_fingerprint": source.source_run_fingerprint,
                    "source_report_path": str(source.source_report_path),
                    "source_report_sha256": source.source_report_sha256,
                    "best_valid_path": str(source.best_valid_path),
                }
                conflicts = {
                    key: {"stored": existing[key], "incoming": value}
                    for key, value in expected.items()
                    if existing[key] != value
                }
                if conflicts:
                    raise SubmissionRepositoryError(
                        "conflicting submission job replay: "
                        + json.dumps(conflicts, sort_keys=True)
                    )
                return dict(existing)
            connection.execute(
                "INSERT INTO submission_jobs(job_id,source_run_id,source_database_path,"
                "source_run_fingerprint,source_report_path,source_report_sha256,best_valid_path,"
                "best_valid_sha256,state,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    job_id,
                    source.source_run_id,
                    str(source.source_database_path),
                    source.source_run_fingerprint,
                    str(source.source_report_path),
                    source.source_report_sha256,
                    str(source.best_valid_path),
                    source.best_valid_sha256,
                    SubmissionState.CREATED,
                    now,
                    now,
                ),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM submission_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise SubmissionRepositoryError(f"unknown submission job: {job_id}")
        return dict(row)

    def transition(
        self,
        job_id: str,
        expected: SubmissionState,
        target: SubmissionState,
        *,
        evidence: Mapping[str, Any] | None = None,
        updates: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if target not in _NEXT[expected]:
            raise SubmissionRepositoryError(
                f"invalid submission transition: {expected} -> {target}"
            )
        update_values = dict(updates or {})
        unknown = set(update_values).difference(self._UPDATE_COLUMNS)
        if unknown:
            raise SubmissionRepositoryError(
                f"unsupported submission job updates: {sorted(unknown)}"
            )
        now = utc_now()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM submission_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise SubmissionRepositoryError(f"unknown submission job: {job_id}")
            observed = SubmissionState(row["state"])
            if observed == target:
                return dict(row)
            if observed != expected:
                raise SubmissionRepositoryError(
                    f"submission job {job_id} is {observed}, expected {expected}"
                )
            assignments = ["state=?", "updated_at=?", *[f"{name}=?" for name in update_values]]
            values = [target, now, *update_values.values(), job_id]
            connection.execute(
                f"UPDATE submission_jobs SET {','.join(assignments)} WHERE job_id=?", values
            )
            connection.execute(
                "INSERT INTO submission_transitions(job_id,from_state,to_state,evidence_json,created_at) "
                "VALUES(?,?,?,?,?)",
                (
                    job_id,
                    expected,
                    target,
                    json.dumps(dict(evidence or {}), sort_keys=True, default=str),
                    now,
                ),
            )
        return self.get_job(job_id)

    def terminate(
        self,
        job_id: str,
        target: SubmissionState,
        *,
        code: str,
        summary: str,
    ) -> dict[str, Any]:
        if target not in {SubmissionState.REJECTED, SubmissionState.FAILED}:
            raise SubmissionRepositoryError("terminal state must be REJECTED or FAILED")
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT state FROM submission_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise SubmissionRepositoryError(f"unknown submission job: {job_id}")
            observed = SubmissionState(row["state"])
            if observed in {
                SubmissionState.HANDED_OFF,
                SubmissionState.REJECTED,
                SubmissionState.FAILED,
            }:
                raise SubmissionRepositoryError(f"cannot terminate job in {observed}")
            now = utc_now()
            connection.execute(
                "UPDATE submission_jobs SET state=?,error_code=?,error_summary=?,updated_at=? "
                "WHERE job_id=?",
                (target, code, summary[-2000:], now, job_id),
            )
            connection.execute(
                "INSERT INTO submission_transitions(job_id,from_state,to_state,evidence_json,created_at) "
                "VALUES(?,?,?,?,?)",
                (job_id, observed, target, json.dumps({"code": code}), now),
            )
        return self.get_job(job_id)

    def record_check(
        self,
        job_id: str,
        *,
        ordinal: int,
        csv_path: Path,
        command: tuple[str, ...],
        stdout: str,
        stderr: str,
        returncode: int,
        transcript_path: Path,
    ) -> None:
        csv_hash = sha256_file(csv_path)
        transcript_hash = sha256_file(transcript_path)
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM submission_checks WHERE job_id=? AND ordinal=?", (job_id, ordinal)
            ).fetchone()
            expected = {
                "csv_path": str(csv_path.resolve()),
                "csv_sha256": csv_hash,
                "command_json": json.dumps(list(command), separators=(",", ":")),
                "stdout": stdout,
                "stderr": stderr,
                "returncode": returncode,
                "transcript_path": str(transcript_path.resolve()),
                "transcript_sha256": transcript_hash,
            }
            if existing is not None:
                conflicts = {
                    key: {"stored": existing[key], "incoming": value}
                    for key, value in expected.items()
                    if existing[key] != value
                }
                if conflicts:
                    raise SubmissionRepositoryError(
                        "conflicting checker replay: " + json.dumps(conflicts, sort_keys=True)
                    )
                return
            connection.execute(
                "INSERT INTO submission_checks(job_id,ordinal,csv_path,csv_sha256,command_json,"
                "stdout,stderr,returncode,transcript_path,transcript_sha256,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, ordinal, *expected.values(), utc_now()),
            )

    def get_check(self, job_id: str, ordinal: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM submission_checks WHERE job_id=? AND ordinal=?", (job_id, ordinal)
            ).fetchone()
        return None if row is None else dict(row)

    def authorize_handoff(self, job_id: str, seal_sha256: str, target_path: Path) -> dict[str, Any]:
        job = self.get_job(job_id)
        if SubmissionState(job["state"]) != SubmissionState.READY_FOR_HANDOFF:
            if SubmissionState(job["state"]) == SubmissionState.HANDOFF_IN_PROGRESS:
                return self.get_handoff(job_id)
            raise SubmissionRepositoryError("handoff requires READY_FOR_HANDOFF")
        if job["seal_sha256"] != seal_sha256:
            raise SubmissionRepositoryError("handoff authorization does not match sealed manifest")
        target = str(target_path.resolve())
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM submission_handoffs WHERE job_id=?", (job_id,)
            ).fetchone()
            if existing is not None:
                if (
                    existing["authorized_seal_sha256"] != seal_sha256
                    or existing["target_path"] != target
                ):
                    raise SubmissionRepositoryError("one-time handoff was already authorized")
            else:
                try:
                    connection.execute(
                        "INSERT INTO submission_handoffs(job_id,authorized_seal_sha256,target_path,"
                        "status,authorized_at) VALUES(?,?,?,?,?)",
                        (job_id, seal_sha256, target, "AUTHORIZED", utc_now()),
                    )
                except sqlite3.IntegrityError as error:
                    raise SubmissionRepositoryError(
                        "handoff target was already claimed by another job"
                    ) from error
        current = SubmissionState(self.get_job(job_id)["state"])
        if current == SubmissionState.READY_FOR_HANDOFF:
            self.transition(
                job_id,
                SubmissionState.READY_FOR_HANDOFF,
                SubmissionState.HANDOFF_IN_PROGRESS,
                evidence={"seal_sha256": seal_sha256, "target_path": target},
            )
        elif current != SubmissionState.HANDOFF_IN_PROGRESS:
            raise SubmissionRepositoryError(f"handoff authorization job state is {current}")
        return self.get_handoff(job_id)

    def get_handoff(self, job_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM submission_handoffs WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise SubmissionRepositoryError(f"submission job has no handoff: {job_id}")
        return dict(row)

    def complete_handoff(self, job_id: str, target_manifest_sha256: str) -> dict[str, Any]:
        already_complete = False
        with self.transaction() as connection:
            handoff = connection.execute(
                "SELECT * FROM submission_handoffs WHERE job_id=?", (job_id,)
            ).fetchone()
            if handoff is None:
                raise SubmissionRepositoryError("handoff has not been authorized")
            if handoff["status"] == "COMPLETE":
                if handoff["target_manifest_sha256"] != target_manifest_sha256:
                    raise SubmissionRepositoryError("completed handoff manifest drifted")
                already_complete = True
            else:
                connection.execute(
                    "UPDATE submission_handoffs SET status='COMPLETE',target_manifest_sha256=?,"
                    "completed_at=? WHERE job_id=?",
                    (target_manifest_sha256, utc_now(), job_id),
                )
        job = self.get_job(job_id)
        if SubmissionState(job["state"]) == SubmissionState.HANDED_OFF:
            return job
        if SubmissionState(job["state"]) != SubmissionState.HANDOFF_IN_PROGRESS:
            suffix = " after completed copy" if already_complete else ""
            raise SubmissionRepositoryError(f"handoff job state is inconsistent{suffix}")
        return self.transition(
            job_id,
            SubmissionState.HANDOFF_IN_PROGRESS,
            SubmissionState.HANDED_OFF,
            evidence={"target_manifest_sha256": target_manifest_sha256},
        )
