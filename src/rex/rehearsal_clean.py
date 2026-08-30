"""Fresh-clone envelope for the production R3 rehearsal.

This module intentionally uses only the Python standard library.  It is the
outer trust boundary: it starts the six-hour clock before setup, creates the
isolated environment, observes the durable run database, injects one verified
coordinator failure, and resumes the exact same run.  The production control
plane remains responsible for experiments and model artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


MAX_R3_SECONDS = 6 * 60 * 60
DEFAULT_SNAPSHOT_SECONDS = 60 * 60
_FORBIDDEN_OUTPUT_NAMES = {
    "submission.csv",
    "final_submission.csv",
    "test_predictions.npz",
    "test_predictions.json",
}
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class R3EnvelopeError(RuntimeError):
    """Raised when an R3 invariant cannot be proven."""


@dataclass(frozen=True)
class R3Options:
    source_root: Path
    source_ref: str
    data_dir: Path
    output_dir: Path
    llm: str
    run_id: str | None = None
    python_executable: str = sys.executable
    wall_clock_seconds: int = MAX_R3_SECONDS
    finalization_reserve_seconds: int = 20 * 60
    snapshot_interval_seconds: int = DEFAULT_SNAPSHOT_SECONDS
    lease_wait_seconds: int = 3 * 60 * 60
    pre_injection_restart_limit: int = 2
    authorize_paid_api: bool = False
    skip_dependency_install: bool = False

    def normalized(self) -> "R3Options":
        source = self.source_root.resolve()
        data = self.data_dir.resolve()
        output = self.output_dir.resolve()
        if self.llm not in {"codex_cli", "claude_cli", "openai_api", "auto"}:
            raise R3EnvelopeError(
                "R3 requires codex_cli, claude_cli, openai_api, or auto; fixed is not live research"
            )
        if self.llm == "openai_api" and not self.authorize_paid_api:
            raise R3EnvelopeError("openai_api requires --authorize-paid-api")
        if self.wall_clock_seconds <= 0 or self.wall_clock_seconds > MAX_R3_SECONDS:
            raise R3EnvelopeError("R3 wall-clock limit must be between 1 second and six hours")
        if self.finalization_reserve_seconds < 0:
            raise R3EnvelopeError("finalization reserve cannot be negative")
        if self.finalization_reserve_seconds >= self.wall_clock_seconds:
            raise R3EnvelopeError("finalization reserve must be smaller than the wall-clock limit")
        if self.snapshot_interval_seconds <= 0:
            raise R3EnvelopeError("snapshot interval must be positive")
        if self.lease_wait_seconds <= 0:
            raise R3EnvelopeError("worker-lease wait must be positive")
        if not 0 <= self.pre_injection_restart_limit <= 2:
            raise R3EnvelopeError("pre-injection restart limit must be between zero and two")
        if self.run_id is not None and not _SAFE_RUN_ID.fullmatch(self.run_id):
            raise R3EnvelopeError("R3 run ID must be one safe path component")
        if not source.is_dir():
            raise R3EnvelopeError(f"source repository is missing: {source}")
        if not data.is_dir():
            raise R3EnvelopeError(f"KuaiRand-Pure data directory is missing: {data}")
        if _is_relative_to(output, source) or _is_relative_to(source, output):
            raise R3EnvelopeError("R3 output must be outside the source repository")
        return R3Options(
            source_root=source,
            source_ref=self.source_ref,
            data_dir=data,
            output_dir=output,
            llm=self.llm,
            run_id=self.run_id,
            python_executable=self.python_executable,
            wall_clock_seconds=self.wall_clock_seconds,
            finalization_reserve_seconds=self.finalization_reserve_seconds,
            snapshot_interval_seconds=self.snapshot_interval_seconds,
            lease_wait_seconds=self.lease_wait_seconds,
            pre_injection_restart_limit=self.pre_injection_restart_limit,
            authorize_paid_api=self.authorize_paid_api,
            skip_dependency_install=self.skip_dependency_install,
        )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _logical_requirement_lines(text: str) -> list[str]:
    logical: list[str] = []
    pending = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pending = f"{pending} {stripped}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        logical.append(pending)
        pending = ""
    if pending:
        logical.append(pending)
    return logical


def validate_hash_lock(path: str | Path) -> dict[str, Any]:
    """Reject a partial/pinned-only requirements file before pip is invoked."""

    candidate = Path(path).resolve()
    if not candidate.is_file():
        raise R3EnvelopeError(f"dependency lock is missing: {candidate}")
    requirements = []
    errors = []
    hash_pattern = re.compile(r"--hash=sha256:[0-9a-fA-F]{64}(?:\s|$)")
    pin_pattern = re.compile(r"^[A-Za-z0-9_.-]+(?:\[[^]]+\])?==[^\s;]+")
    for line in _logical_requirement_lines(candidate.read_text(encoding="utf-8")):
        if line.startswith("--") and not line.startswith("--editable"):
            continue
        requirements.append(line)
        if not pin_pattern.search(line):
            errors.append(f"not exactly pinned: {line}")
        if not hash_pattern.search(f"{line} "):
            errors.append(f"missing sha256 hash: {line}")
    if not requirements:
        errors.append("lock contains no package requirements")
    if errors:
        raise R3EnvelopeError("dependency lock is not fully hash-locked: " + "; ".join(errors))
    return {
        "path": str(candidate),
        "sha256": _sha256(candidate),
        "requirements": len(requirements),
        "require_hashes": True,
    }


def git_snapshot(root: str | Path) -> dict[str, Any]:
    candidate = Path(root).resolve()

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=candidate,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise R3EnvelopeError(result.stderr.strip() or f"git {' '.join(arguments)} failed")
        return result.stdout.strip()

    return {
        "root": str(candidate),
        "commit": git("rev-parse", "HEAD"),
        "status": git("status", "--porcelain=v1", "--untracked-files=all"),
        "index_sha256": hashlib.sha256(git("ls-files", "-s").encode("utf-8")).hexdigest(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise R3EnvelopeError(f"expected a JSON object: {path}")
    return value


def _read_run_status(database: Path, run_id: str) -> dict[str, Any]:
    if not database.is_file():
        return {"database_ready": False}
    uri = f"file:{database}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=0.25)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        run = connection.execute(
            "SELECT state,stop_reason,hypothesis_count,official_evaluation_count,"
            "non_improvement_streak,best_primary_units,search_champion_experiment_id,"
            "deadline_epoch_ms,updated_at FROM runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if run is None:
            return {"database_ready": True, "run_ready": False}
        latest = connection.execute(
            "SELECT experiment_id,iteration_number,method_card_id,state,terminal_reason "
            "FROM experiments WHERE run_id=? ORDER BY iteration_number DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        session = connection.execute(
            "SELECT session_id,pid,last_heartbeat,ended_at,exit_reason FROM process_sessions "
            "WHERE run_id=? ORDER BY started_at DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        repairs = connection.execute(
            "SELECT COUNT(*) FROM experiment_repairs repair JOIN experiments experiment "
            "ON experiment.experiment_id=repair.experiment_id WHERE experiment.run_id=?",
            (run_id,),
        ).fetchone()[0]
        promotions = connection.execute(
            "SELECT COUNT(*) FROM search_promotions WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        return {
            "database_ready": True,
            "run_ready": True,
            "run": dict(run),
            "latest_experiment": None if latest is None else dict(latest),
            "latest_session": None if session is None else dict(session),
            "repair_count": int(repairs),
            "promotion_count": int(promotions),
        }
    except sqlite3.Error as error:
        return {"database_ready": True, "read_error": str(error)}
    finally:
        if "connection" in locals():
            connection.close()


def _database_audit(database: Path, run_id: str) -> dict[str, Any]:
    """Collect exact-once and validation-only facts from the durable store."""

    if not database.is_file():
        return {"database_ready": False}
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=1)
    connection.row_factory = sqlite3.Row
    try:
        def scalar(sql: str, params: tuple[Any, ...] = (run_id,)) -> int:
            return int(connection.execute(sql, params).fetchone()[0])
        providers = [
            dict(row)
            for row in connection.execute(
                "SELECT provider,model,COUNT(*) AS calls FROM llm_calls WHERE run_id=? "
                "GROUP BY provider,model ORDER BY provider,model",
                (run_id,),
            )
        ]
        duplicate_iterations = [
            dict(row)
            for row in connection.execute(
                "SELECT iteration_number,COUNT(*) AS copies FROM experiments WHERE run_id=? "
                "GROUP BY iteration_number HAVING COUNT(*)>1",
                (run_id,),
            )
        ]
        return {
            "database_ready": True,
            "experiments": scalar("SELECT COUNT(*) FROM experiments WHERE run_id=?"),
            "attempts": scalar(
                "SELECT COUNT(*) FROM attempts attempt JOIN experiments experiment "
                "ON experiment.experiment_id=attempt.experiment_id WHERE experiment.run_id=?"
            ),
            "transitions": scalar(
                "SELECT COUNT(*) FROM transitions transition JOIN experiments experiment "
                "ON experiment.experiment_id=transition.experiment_id WHERE experiment.run_id=?"
            ),
            "promotions": scalar("SELECT COUNT(*) FROM search_promotions WHERE run_id=?"),
            "convergence_transactions": scalar(
                "SELECT COUNT(*) FROM convergence_transactions WHERE run_id=?"
            ),
            "test_metrics": scalar(
                "SELECT COUNT(*) FROM metrics metric JOIN experiments experiment "
                "ON experiment.experiment_id=metric.experiment_id "
                "WHERE experiment.run_id=? AND metric.split='test'"
            ),
            "predict_attempts": scalar(
                "SELECT COUNT(*) FROM attempts attempt JOIN experiments experiment "
                "ON experiment.experiment_id=attempt.experiment_id "
                "WHERE experiment.run_id=? AND attempt.rung='predict'"
            ),
            "duplicate_iterations": duplicate_iterations,
            "providers": providers,
        }
    finally:
        connection.close()


def _artifact_payload(path: Path, kind: str) -> dict[str, Any]:
    digest = _sha256(path)
    return {
        "artifact_id": f"r3-{kind}-{digest[:20]}",
        "kind": kind,
        "path": str(path.resolve()),
        "sha256": digest,
        "size_bytes": path.stat().st_size,
        "schema_version": "1.0",
    }


def validate_r3_manifest_with_runtime(
    path: str | Path,
    *,
    python_executable: str | Path,
    source_root: str | Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Round-trip the emitted manifest through the cloned Pydantic contract.

    The outer launcher stays standard-library-only. Contract validation happens
    in the freshly installed rehearsal interpreter against the exact cloned
    source commit that ran the experiment.
    """

    manifest_path = Path(path).resolve()
    source = Path(source_root).resolve()
    if not manifest_path.is_file():
        raise R3EnvelopeError(f"R3 manifest is missing: {manifest_path}")
    if timeout_seconds <= 0:
        raise R3EnvelopeError("R3 manifest validation has no time remaining")
    helper = (
        "import json,sys; from pathlib import Path; "
        "from rex.contracts import RehearsalR3Manifest; "
        "m=RehearsalR3Manifest.model_validate_json(Path(sys.argv[1]).read_text()); "
        "print(json.dumps(m.model_dump(mode='json'),sort_keys=True,separators=(',',':')))"
    )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(source / "src")
    try:
        result = subprocess.run(
            [str(python_executable), "-c", helper, str(manifest_path)],
            cwd=source,
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise R3EnvelopeError("R3 manifest contract validation exceeded the deadline") from error
    if result.returncode != 0:
        detail = result.stderr.strip()[-2000:] or "contract validator exited unsuccessfully"
        raise R3EnvelopeError(f"R3 manifest failed contract validation: {detail}")
    try:
        normalized = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise R3EnvelopeError("R3 manifest validator returned malformed JSON") from error
    if not isinstance(normalized, dict) or normalized != _read_json(manifest_path):
        raise R3EnvelopeError("R3 manifest did not survive an exact contract round trip")
    return normalized


def compact_status_snapshot(output_dir: str | Path, *, reason: str = "requested") -> dict[str, Any]:
    """Read and persist a compact status sample without importing the agent."""

    output = Path(output_dir).resolve()
    envelope_path = output / "envelope_state.json"
    if not envelope_path.is_file():
        raise R3EnvelopeError(f"R3 envelope state is missing: {envelope_path}")
    envelope = _read_json(envelope_path)
    run_id = str(envelope["run_id"])
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise R3EnvelopeError("R3 envelope contains an unsafe run ID")
    database = Path(str(envelope["runs_dir"])) / run_id / "state.sqlite3"
    deadline_epoch = float(envelope["deadline_epoch"])
    sample = {
        "schema_version": "1.0",
        "sampled_at": _utc_now(),
        "reason": reason,
        "phase": envelope.get("phase"),
        "run_id": run_id,
        "source_commit": envelope.get("source_commit"),
        "llm": envelope.get("llm"),
        "elapsed_seconds": max(0.0, time.time() - float(envelope["started_epoch"])),
        "remaining_seconds": max(0.0, deadline_epoch - time.time()),
        "fault": envelope.get("fault", {"state": "pending"}),
        "validation_only": True,
        "test_prediction_enabled": False,
        "final_submission_enabled": False,
        **_read_run_status(database, run_id),
    }
    _atomic_json(output / "status" / "latest.json", sample)
    return sample


class R3Envelope:
    def __init__(
        self,
        options: R3Options,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        epoch: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.options = options.normalized()
        self.monotonic = monotonic
        self.epoch = epoch
        self.sleep = sleep
        self.started_monotonic = monotonic()
        self.started_epoch = epoch()
        self.deadline_epoch = self.started_epoch + self.options.wall_clock_seconds
        self.deadline_monotonic = self.started_monotonic + self.options.wall_clock_seconds
        self.output = self.options.output_dir
        self.clone = self.output / "source"
        self.venv = self.output / "venv"
        self.runtime = self.output / "runtime"
        self.runs = self.runtime / "runs"
        self.logs = self.output / "logs"
        self.status_dir = self.output / "status"
        self.run_id = self.options.run_id or ""
        self.source_commit = ""
        self.snapshot_number = 0
        self.next_snapshot = self.started_monotonic
        self.fault: dict[str, Any] = {"state": "pending", "count": 0}
        self.processes: list[subprocess.Popen[bytes]] = []
        self.source_snapshot: dict[str, Any] | None = None
        self.clone_snapshot: dict[str, Any] | None = None
        self.clone_protection: dict[str, Any] = {}

    def _remaining(self) -> float:
        return max(0.0, self.deadline_monotonic - self.monotonic())

    def _require_time(self, phase: str) -> float:
        remaining = self._remaining()
        if remaining <= 0:
            raise R3EnvelopeError(f"global six-hour ceiling reached during {phase}")
        return remaining

    def _state(self, phase: str, **extra: Any) -> dict[str, Any]:
        value = {
            "schema_version": "1.0",
            "phase": phase,
            "started_at": datetime.fromtimestamp(self.started_epoch, timezone.utc).isoformat(),
            "started_epoch": self.started_epoch,
            "deadline_epoch": self.deadline_epoch,
            "wall_clock_seconds": self.options.wall_clock_seconds,
            "run_id": self.run_id,
            "source_commit": self.source_commit,
            "source_root": str(self.options.source_root),
            "clone_root": str(self.clone),
            "runs_dir": str(self.runs),
            "llm": self.options.llm,
            "paid_api_authorized": self.options.authorize_paid_api,
            "validation_only": True,
            "test_prediction_enabled": False,
            "final_submission_enabled": False,
            "fault": self.fault,
            **extra,
        }
        _atomic_json(self.output / "envelope_state.json", value)
        return value

    def _checked(
        self,
        name: str,
        command: Sequence[str],
        *,
        cwd: Path,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self._require_time(name)
        self.logs.mkdir(parents=True, exist_ok=True)
        stdout_path = self.logs / f"{name}.stdout.log"
        stderr_path = self.logs / f"{name}.stderr.log"
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                result = subprocess.run(
                    list(command),
                    cwd=cwd,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    check=False,
                    timeout=self._remaining(),
                )
        except subprocess.TimeoutExpired as error:
            raise R3EnvelopeError(f"global deadline expired during {name}") from error
        if result.returncode != 0:
            tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise R3EnvelopeError(f"{name} failed with exit {result.returncode}: {tail}")
        return result

    def _resolve_source(self) -> dict[str, Any]:
        source_before = git_snapshot(self.options.source_root)
        if source_before["status"]:
            raise R3EnvelopeError(
                "source repository must be clean and committed before a clean R3 rehearsal"
            )
        resolved = subprocess.run(
            ["git", "rev-parse", f"{self.options.source_ref}^{{commit}}"],
            cwd=self.options.source_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if resolved.returncode != 0:
            raise R3EnvelopeError(f"cannot resolve source ref {self.options.source_ref!r}")
        self.source_commit = resolved.stdout.strip()
        self.run_id = self.options.run_id or (
            f"r3-{datetime.fromtimestamp(self.started_epoch, timezone.utc):%Y%m%d-%H%M%S}-"
            f"{self.source_commit[:8]}-{uuid.uuid4().hex[:6]}"
        )
        return source_before

    def _prepare_directories(self) -> None:
        if self.output.exists() and any(self.output.iterdir()):
            raise R3EnvelopeError(f"R3 output directory must be new or empty: {self.output}")
        self.output.mkdir(parents=True, exist_ok=True)
        self.runtime.mkdir(parents=True)
        self.runs.mkdir(parents=True)
        self.status_dir.mkdir(parents=True)

    def _clone_source(self) -> dict[str, Any]:
        self._checked(
            "clone",
            [
                "git",
                "clone",
                "--no-local",
                "--no-checkout",
                str(self.options.source_root),
                str(self.clone),
            ],
            cwd=self.output,
        )
        self._checked("checkout", ["git", "checkout", "--detach", self.source_commit], cwd=self.clone)
        snapshot = git_snapshot(self.clone)
        if snapshot["commit"] != self.source_commit or snapshot["status"]:
            raise R3EnvelopeError("fresh clone does not match the selected clean commit")
        tracked = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=self.clone,
            capture_output=True,
            check=True,
        ).stdout.split(b"\0")
        protected = 0
        for raw in tracked:
            if not raw:
                continue
            path = self.clone / os.fsdecode(raw)
            if path.is_file() and not path.is_symlink():
                path.chmod(path.stat().st_mode & ~0o222)
                protected += 1
        writable = [
            os.fsdecode(raw)
            for raw in tracked
            if raw
            and (self.clone / os.fsdecode(raw)).is_file()
            and os.access(self.clone / os.fsdecode(raw), os.W_OK)
        ]
        # Root may report write access despite mode bits, so evidence is based
        # on the actual permission mask as well as this diagnostic list.
        mode_writable = [
            os.fsdecode(raw)
            for raw in tracked
            if raw
            and (self.clone / os.fsdecode(raw)).is_file()
            and ((self.clone / os.fsdecode(raw)).stat().st_mode & 0o222)
        ]
        if mode_writable:
            raise R3EnvelopeError("failed to make the clean controller source read-only")
        self.clone_protection = {
            "tracked_files": protected,
            "mode_writable_files": mode_writable,
            "access_writable_diagnostic_count": len(writable),
        }
        return snapshot

    def _install(self) -> dict[str, Any]:
        lock = self.clone / "requirements-lock.txt"
        lock_evidence = validate_hash_lock(lock)
        if self.options.skip_dependency_install:
            python = Path(self.options.python_executable).resolve()
            return {**lock_evidence, "skipped_for_test": True, "python": str(python)}
        self._checked(
            "venv",
            [self.options.python_executable, "-m", "venv", str(self.venv)],
            cwd=self.output,
        )
        python = self.venv / "bin" / "python"
        self._checked(
            "dependencies",
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--require-hashes",
                "--only-binary=:all:",
                "-r",
                str(lock),
            ],
            cwd=self.clone,
        )
        self._checked("pip-check", [str(python), "-m", "pip", "check"], cwd=self.clone)
        self._checked(
            "installer-provenance",
            [
                str(python),
                "-c",
                (
                    "import importlib.metadata,json,platform,sys,sysconfig;"
                    "print(json.dumps({'executable':sys.executable,'python':sys.version,"
                    "'implementation':platform.python_implementation(),"
                    "'platform':platform.platform(),'machine':platform.machine(),"
                    "'pip':importlib.metadata.version('pip'),"
                    "'soabi':sysconfig.get_config_var('SOABI')},sort_keys=True))"
                ),
            ],
            cwd=self.clone,
        )
        self._checked(
            "installed-inventory",
            [str(python), "-m", "pip", "freeze", "--all"],
            cwd=self.clone,
        )
        inventory = self.logs / "installed-inventory.stdout.log"
        installer = self.logs / "installer-provenance.stdout.log"
        return {
            **lock_evidence,
            "python": str(python),
            "python_sha256": _sha256(python),
            "only_binary": True,
            "installer_provenance_sha256": _sha256(installer),
            "installed_inventory_sha256": _sha256(inventory),
            "skipped_for_test": False,
        }

    def _python(self) -> str:
        if self.options.skip_dependency_install:
            return self.options.python_executable
        return str(self.venv / "bin" / "python")

    def _write_runtime_files(self) -> tuple[Path, Path]:
        template_path = self.clone / "configs" / "run" / "rehearsal_r3.yaml"
        template = _read_json(template_path)
        remaining = self._require_time("runtime configuration")
        # The external deadline is authoritative.  Give the internal budget a
        # tiny rounding margin so ProductionAutopilot can prove that the
        # external deadline only shortens (never extends) its configured cap.
        internal_wall_seconds = min(
            self.options.wall_clock_seconds,
            math.ceil(remaining) + 2,
        )
        reserve = min(
            self.options.finalization_reserve_seconds,
            max(0, internal_wall_seconds - 1),
        )
        budget = {
            "schema_version": "1.0",
            "max_hypotheses": 50,
            "max_official_evaluations": 50,
            "wall_clock_seconds": internal_wall_seconds,
            "finalization_reserve_seconds": reserve,
            "convergence_epsilon": 0.002,
            "convergence_patience": 3,
            "max_repairs_per_experiment": 2,
            "default_attempt_timeout_seconds": 600,
        }
        budget_path = self.runtime / "budget.json"
        _atomic_json(budget_path, budget)
        llm = dict(template["llm"])
        llm["mode"] = self.options.llm
        auto = dict(llm.get("auto", {}))
        auto["allow_paid_api_fallback"] = bool(self.options.authorize_paid_api)
        llm["auto"] = auto
        template.update(
            {
                "project_root": str(self.clone),
                "runs_dir": str(self.runs),
                "budget_config": str(budget_path),
                "protected_paths": str(self.clone / "configs/security/protected_paths.yaml"),
                "data_manifest": str(self.runtime / "data/data_manifest.json"),
                "raw_data_dir": str(self.options.data_dir),
                "evaluator_path": str(self.clone / "kuairand-starter-kit/evaluate.py"),
                "environment_lock": str(self.clone / "requirements-lock.txt"),
                "scientific_execution_enabled": True,
                "confirmation_enabled": False,
                "test_prediction_enabled": False,
                "final_submission_enabled": False,
                "llm": llm,
            }
        )
        config_path = self.runtime / "rehearsal_r3.json"
        _atomic_json(config_path, template)
        return config_path, budget_path

    def _preflight(self, environment: dict[str, str]) -> tuple[Path, dict[str, Any]]:
        python = self._python()
        self._checked(
            "data-bootstrap",
            [
                python,
                "-m",
                "rex.cli",
                "bootstrap",
                "--data-dir",
                str(self.options.data_dir),
                "--output-dir",
                str(self.runtime / "data"),
            ],
            cwd=self.clone,
            environment=environment,
        )
        config_path, budget_path = self._write_runtime_files()
        doctor_command = [
            python,
            "-m",
            "rex.cli",
            "doctor",
            "--config",
            str(config_path),
            "--llm",
            self.options.llm,
            "--live",
            "--tree",
        ]
        self._checked("doctor", doctor_command, cwd=self.clone, environment=environment)
        data_manifest = self.runtime / "data/data_manifest.json"
        manifest = _read_json(data_manifest)
        test = dict(manifest.get("splits", {}).get("test", {}))
        if int(test.get("row_count", -1)) != 170_588 or test.get("target_path") is not None:
            raise R3EnvelopeError("test view must contain exactly 170,588 rows and no target path")
        evidence = {
            "data_manifest": str(data_manifest),
            "data_manifest_sha256": _sha256(data_manifest),
            "raw_dataset_identity_sha256": manifest.get("raw_dataset_identity_sha256"),
            "starter_manifest_sha256": manifest.get("starter_manifest_sha256"),
            "evaluator_sha256": _sha256(self.clone / "kuairand-starter-kit/evaluate.py"),
            "test_rows": 170_588,
            "test_target_present": False,
            "doctor_stdout_sha256": _sha256(self.logs / "doctor.stdout.log"),
            "runtime_config_sha256": _sha256(config_path),
            "runtime_budget_sha256": _sha256(budget_path),
        }
        return config_path, evidence

    def _run_command(self, config_path: Path, *, resume: bool) -> list[str]:
        command = [
            self._python(),
            "-m",
            "rex.cli",
            "run",
            "--config",
            str(config_path),
            "--resume" if resume else "--run-id",
            self.run_id,
            "--external-deadline-epoch-ms",
            str(int(self.deadline_epoch * 1000)),
            "--llm",
            self.options.llm,
        ]
        if self.options.llm == "auto" and self.options.authorize_paid_api:
            command.append("--allow-paid-api-fallback")
        if self.options.llm == "openai_api" and self.options.authorize_paid_api:
            command.append("--authorize-paid-api")
        return command

    def _launch(self, command: Sequence[str], environment: dict[str, str], name: str) -> subprocess.Popen[bytes]:
        self._require_time(name)
        stdout = (self.logs / f"{name}.stdout.log").open("ab")
        stderr = (self.logs / f"{name}.stderr.log").open("ab")
        try:
            process = subprocess.Popen(
                list(command),
                cwd=self.clone,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
        finally:
            stdout.close()
            stderr.close()
        self.processes.append(process)
        return process

    def _find_active_lease(self, coordinator_pid: int) -> tuple[Path, dict[str, Any]] | None:
        run_dir = self.runs / self.run_id
        for path in sorted(run_dir.glob("**/worker_lease.json")) if run_dir.is_dir() else ():
            try:
                lease = _read_json(path)
            except (OSError, UnicodeError, json.JSONDecodeError, R3EnvelopeError):
                continue
            if lease.get("state") == "active" and int(lease.get("owner_pid", -1)) == coordinator_pid:
                return path, lease
        return None

    def _snapshot(self, reason: str, *, force: bool = False) -> dict[str, Any] | None:
        now = self.monotonic()
        scheduled = now >= self.next_snapshot
        if not force and not scheduled:
            return None
        if self.clone_snapshot is not None and git_snapshot(self.clone) != self.clone_snapshot:
            raise R3EnvelopeError("clean rehearsal source changed between hourly audits")
        if self.source_snapshot is not None and git_snapshot(self.options.source_root) != self.source_snapshot:
            raise R3EnvelopeError("operator source changed between hourly audits")
        snapshot = compact_status_snapshot(self.output, reason=reason)
        label = "hourly" if scheduled else "checkpoint"
        path = self.status_dir / f"{label}-{self.snapshot_number:03d}.json"
        _atomic_json(path, snapshot)
        self.snapshot_number += 1
        if scheduled:
            while self.next_snapshot <= now:
                self.next_snapshot += self.options.snapshot_interval_seconds
        return snapshot

    def _wait_for_injection_point(
        self, process: subprocess.Popen[bytes]
    ) -> tuple[Path, dict[str, Any]] | None:
        wait_deadline = min(
            self.deadline_monotonic - self.options.finalization_reserve_seconds,
            self.monotonic() + self.options.lease_wait_seconds,
        )
        while self.monotonic() < wait_deadline:
            self._snapshot("scheduled-hourly")
            observed = self._find_active_lease(process.pid)
            if observed is not None:
                return observed
            if process.poll() is not None:
                return None
            self.sleep(min(1.0, max(0.01, wait_deadline - self.monotonic())))
        raise R3EnvelopeError("no active durable worker lease appeared before the injection cutoff")

    def _pre_injection_recovery(
        self,
        process: subprocess.Popen[bytes],
        *,
        restart_number: int,
        previous_progress_token: str | None,
    ) -> dict[str, Any]:
        database = self.runs / self.run_id / "state.sqlite3"
        if not database.is_file():
            raise R3EnvelopeError("pre-injection coordinator exit left no durable run database")
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            run = connection.execute("SELECT * FROM runs WHERE run_id=?", (self.run_id,)).fetchone()
            session = connection.execute(
                "SELECT pid,ended_at,exit_reason,last_heartbeat FROM process_sessions "
                "WHERE run_id=? AND pid=? ORDER BY started_at DESC LIMIT 1",
                (self.run_id, process.pid),
            ).fetchone()
            main_counts = {
                "experiments": int(
                    connection.execute(
                        "SELECT COUNT(*) FROM experiments WHERE run_id=?", (self.run_id,)
                    ).fetchone()[0]
                ),
                "transitions": int(
                    connection.execute(
                        "SELECT COUNT(*) FROM transitions transition JOIN experiments experiment "
                        "ON experiment.experiment_id=transition.experiment_id WHERE experiment.run_id=?",
                        (self.run_id,),
                    ).fetchone()[0]
                ),
                "artifacts": int(
                    connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
                ),
                "llm_calls": int(
                    connection.execute(
                        "SELECT COUNT(*) FROM llm_calls WHERE run_id=?", (self.run_id,)
                    ).fetchone()[0]
                ),
                "repairs": int(
                    connection.execute(
                        "SELECT COUNT(*) FROM experiment_repairs repair JOIN experiments experiment "
                        "ON experiment.experiment_id=repair.experiment_id WHERE experiment.run_id=?",
                        (self.run_id,),
                    ).fetchone()[0]
                ),
                "completed_repairs": int(
                    connection.execute(
                        "SELECT COUNT(*) FROM experiment_repairs repair JOIN experiments experiment "
                        "ON experiment.experiment_id=repair.experiment_id WHERE experiment.run_id=? "
                        "AND repair.completed_at IS NOT NULL",
                        (self.run_id,),
                    ).fetchone()[0]
                ),
                "event_high_water": int(
                    connection.execute(
                        "SELECT COALESCE(MAX(sequence),0) FROM event_outbox WHERE run_id=?",
                        (self.run_id,),
                    ).fetchone()[0]
                ),
            }
        finally:
            connection.close()
        if run is None or str(run["state"]) != "SEARCHING":
            raise R3EnvelopeError("pre-injection coordinator exit is not a resumable search run")
        if str(run["root_commit"]) != self.source_commit:
            raise R3EnvelopeError("pre-injection recovery detected source-commit drift")
        if int(run["deadline_epoch_ms"]) != int(self.deadline_epoch * 1000):
            raise R3EnvelopeError("pre-injection recovery detected deadline drift")
        if session is None:
            raise R3EnvelopeError("pre-injection exit has no durable process session")

        transactions: list[dict[str, Any]] = []
        active_states: list[str] = []
        allowed = {
            "PROPOSED",
            "WORKTREE_READY",
            "PATCHED",
            "STATIC_VALID",
            "FIXTURE_VALID",
        }
        run_dir = self.runs / self.run_id
        for path in sorted(run_dir.glob("transactions/*/state.sqlite3")):
            transaction = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            transaction.row_factory = sqlite3.Row
            try:
                rows = transaction.execute(
                    "SELECT experiment_id,state,terminal_reason,commit_sha,config_sha256,"
                    "workspace_path FROM experiments ORDER BY iteration_number"
                ).fetchall()
                transitions = int(
                    transaction.execute("SELECT COUNT(*) FROM transitions").fetchone()[0]
                )
                transition_high_water = int(
                    transaction.execute(
                        "SELECT COALESCE(MAX(transition_id),0) FROM transitions"
                    ).fetchone()[0]
                )
                repairs = int(
                    transaction.execute("SELECT COUNT(*) FROM experiment_repairs").fetchone()[0]
                )
                completed_repairs = int(
                    transaction.execute(
                        "SELECT COUNT(*) FROM experiment_repairs WHERE completed_at IS NOT NULL"
                    ).fetchone()[0]
                )
                llm_calls = int(transaction.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0])
                artifacts = int(transaction.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0])
                event_high_water = int(
                    transaction.execute(
                        "SELECT COALESCE(MAX(sequence),0) FROM event_outbox"
                    ).fetchone()[0]
                )
            finally:
                transaction.close()
            item = {
                "path": str(path.relative_to(run_dir)),
                "experiments": [
                    {
                        "experiment_id": str(row["experiment_id"]),
                        "state": str(row["state"]),
                        "terminal_reason": str(row["terminal_reason"] or "")[-500:],
                        "commit_sha": str(row["commit_sha"] or ""),
                        "config_sha256": str(row["config_sha256"] or ""),
                        "workspace_path": str(row["workspace_path"] or ""),
                    }
                    for row in rows
                ],
                "transitions": transitions,
                "transition_high_water": transition_high_water,
                "repairs": repairs,
                "completed_repairs": completed_repairs,
                "llm_calls": llm_calls,
                "artifacts": artifacts,
                "event_high_water": event_high_water,
            }
            transactions.append(item)
            active_states.extend(
                str(row["state"])
                for row in rows
                if str(row["state"])
                not in {"PROMOTED", "REJECTED", "ABANDONED", "FAILED_FINAL"}
            )
        if any(state not in allowed for state in active_states):
            raise R3EnvelopeError(
                "pre-injection coordinator exit is not a typed preparation interruption"
            )
        progress = {
            "run_state": str(run["state"]),
            "hypothesis_count": int(run["hypothesis_count"]),
            "main_counts": main_counts,
            "transactions": transactions,
        }
        progress_token = hashlib.sha256(
            json.dumps(progress, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if previous_progress_token == progress_token:
            raise R3EnvelopeError("pre-injection recovery made no durable progress")
        return {
            "schema_version": "1.0",
            "restart_number": restart_number,
            "exited_pid": process.pid,
            "return_code": int(process.returncode or 0),
            "process_session": dict(session),
            "progress": progress,
            "progress_token": progress_token,
            "decision": "resume-same-run",
            "recorded_at": _utc_now(),
        }

    def _inject(self, process: subprocess.Popen[bytes], lease_path: Path, lease: dict[str, Any]) -> None:
        if self.fault["count"] != 0:
            raise R3EnvelopeError("controlled failure may be injected exactly once")
        if process.poll() is not None:
            raise R3EnvelopeError("coordinator exited before controlled failure injection")
        if int(lease.get("owner_pid", -1)) != process.pid:
            raise R3EnvelopeError("worker lease is not owned by the selected coordinator")
        database = self.runs / self.run_id / "state.sqlite3"
        request_path = lease_path.parent / "input" / "request.json"
        request = _read_json(request_path) if request_path.is_file() else {}
        intent = {
            "schema_version": "1.0",
            "state": "intent-recorded",
            "run_id": self.run_id,
            "coordinator_pid": process.pid,
            "lease_path": str(lease_path),
            "lease_sha256": _sha256(lease_path),
            "worker_pid": lease.get("pid"),
            "worker_pgid": lease.get("pgid"),
            "request_sha256": lease.get("request_sha256"),
            "execution_sha256": lease.get("execution_sha256"),
            "attempt_id": request.get("attempt_id"),
            "experiment_id": request.get("experiment_id"),
            "pre_fault_status": _read_run_status(database, self.run_id),
            "pre_fault_database_audit": _database_audit(database, self.run_id),
            "recorded_at": _utc_now(),
            "pre_injection_recoveries": list(
                self.fault.get("pre_injection_recoveries", [])
            ),
        }
        injection_path = self.runtime / "fault_injection.json"
        _atomic_json(injection_path, intent)
        os.kill(process.pid, signal.SIGKILL)
        return_code = process.wait(timeout=30)
        if return_code not in {-signal.SIGKILL, 128 + signal.SIGKILL}:
            raise R3EnvelopeError(f"coordinator did not exit through SIGKILL: {return_code}")
        self.fault = {
            **intent,
            "state": "injected",
            "count": 1,
            "return_code": return_code,
            "injected_at": _utc_now(),
        }
        _atomic_json(injection_path, self.fault)
        self._state("fault-injected")
        self._snapshot("post-injection", force=True)

    def _wait_for_completion(self, process: subprocess.Popen[bytes]) -> int:
        while process.poll() is None:
            self._snapshot("scheduled-hourly")
            if self._remaining() <= 15:
                raise R3EnvelopeError(
                    "production run reached the external cleanup guard before the global ceiling"
                )
            self.sleep(min(1.0, self._remaining()))
        return int(process.returncode or 0)

    def _terminate_live_processes(self) -> None:
        for process in self.processes:
            if process.poll() is not None:
                continue
            try:
                os.kill(process.pid, signal.SIGKILL)
                process.wait(timeout=min(5.0, max(0.1, self._remaining())))
            except (ProcessLookupError, subprocess.TimeoutExpired):
                continue

    def _recover_active_workers(self) -> list[dict[str, Any]]:
        """Use the production identity verifier to reap workers after envelope failure."""

        recovered: list[dict[str, Any]] = []
        if not self.clone.is_dir():
            return recovered
        run_dir = self.runs / self.run_id
        if not run_dir.is_dir():
            return recovered
        helper = (
            "import json,sys; from pathlib import Path; "
            "from rex.execution.lease import recover_orphan_worker; "
            "p=Path(sys.argv[1]); m=json.loads(p.read_text()); "
            "r=recover_orphan_worker(p,p.with_name('worker_recovery.json'),"
            "request_sha256=m['request_sha256'],execution_sha256=m['execution_sha256']); "
            "print(json.dumps(r))"
        )
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = str(self.clone / "src")
        for lease_path in sorted(run_dir.glob("**/worker_lease.json")):
            try:
                marker = _read_json(lease_path)
            except (OSError, UnicodeError, json.JSONDecodeError, R3EnvelopeError):
                continue
            if marker.get("state") != "active":
                continue
            remaining = self._remaining()
            if remaining <= 0:
                recovered.append({"path": str(lease_path), "error": "cleanup deadline exhausted"})
                continue
            result = subprocess.run(
                [self._python(), "-c", helper, str(lease_path)],
                cwd=self.clone,
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
                timeout=min(5.0, remaining),
            )
            recovered.append(
                {
                    "path": str(lease_path),
                    "return_code": result.returncode,
                    "result": result.stdout.strip()[-2000:],
                    "error": result.stderr.strip()[-2000:],
                }
            )
        if recovered:
            _atomic_json(self.runtime / "envelope_worker_cleanup.json", {"workers": recovered})
        return recovered

    def _recovery_audit(self, injected_lease: Path) -> dict[str, Any]:
        recovery_path = injected_lease.with_name("worker_recovery.json")
        if not recovery_path.is_file():
            raise R3EnvelopeError("resume did not emit recovery evidence for the injected worker")
        recovery = _read_json(recovery_path)
        events = recovery.get("events")
        if not isinstance(events, list) or not events:
            raise R3EnvelopeError("worker recovery evidence has no events")
        worker_pid = int(self.fault["worker_pid"])
        matches = [
            event
            for event in events
            if isinstance(event, dict)
            and int(event.get("pid", -1)) == worker_pid
            and event.get("outcome")
            in {"orphan-process-group-terminated", "stale-lease-no-process"}
        ]
        if not matches:
            raise R3EnvelopeError("worker recovery evidence does not match the injected worker")
        database = self.runs / self.run_id / "state.sqlite3"
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            sessions = [
                dict(row)
                for row in connection.execute(
                    "SELECT session_id,pid,ended_at,exit_reason FROM process_sessions "
                    "WHERE run_id=? ORDER BY started_at",
                    (self.run_id,),
                )
            ]
            attempt_id = self.fault.get("attempt_id")
            attempt_copies = (
                int(
                    connection.execute(
                        "SELECT COUNT(*) FROM attempts WHERE attempt_id=?", (attempt_id,)
                    ).fetchone()[0]
                )
                if attempt_id
                else 0
            )
            pre_champion = (
                self.fault.get("pre_fault_status", {}).get("run", {}).get(
                    "search_champion_experiment_id"
                )
            )
            champion_evidence = 0
            if pre_champion == "baseline":
                champion_evidence = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM baseline_gates WHERE run_id=?", (self.run_id,)
                    ).fetchone()[0]
                )
            elif pre_champion:
                champion_evidence = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM search_promotions WHERE run_id=? AND "
                        "(experiment_id=? OR previous_experiment_id=?)",
                        (self.run_id, pre_champion, pre_champion),
                    ).fetchone()[0]
                )
        finally:
            connection.close()
        if len(sessions) < 2:
            raise R3EnvelopeError("same-run resume did not create a second durable process session")
        if self.fault.get("attempt_id") and attempt_copies != 1:
            raise R3EnvelopeError("injected attempt was not preserved exactly once after resume")
        if pre_champion and champion_evidence < 1:
            raise R3EnvelopeError("pre-fault champion is absent from durable promotion evidence")
        before = dict(self.fault.get("pre_fault_database_audit", {}))
        after = _database_audit(database, self.run_id)
        for field in ("experiments", "attempts", "transitions", "promotions", "convergence_transactions"):
            if int(after.get(field, 0)) < int(before.get(field, 0)):
                raise R3EnvelopeError(f"durable {field} count moved backwards during recovery")
        if after.get("duplicate_iterations"):
            raise R3EnvelopeError("resume produced duplicate hypothesis iteration numbers")
        return {
            "recovery_path": str(recovery_path),
            "recovery_sha256": _sha256(recovery_path),
            "matched_event": matches[-1],
            "process_sessions": sessions,
            "same_run_resumed": True,
            "attempt_id": self.fault.get("attempt_id"),
            "attempt_exactly_once": not self.fault.get("attempt_id") or attempt_copies == 1,
            "pre_fault_champion": pre_champion,
            "pre_fault_champion_preserved": not pre_champion or champion_evidence >= 1,
            "pre_fault_database_audit": before,
            "post_recovery_database_audit": after,
            "fault_count": 1,
        }

    def _validation_only_audit(self) -> dict[str, Any]:
        run_dir = self.runs / self.run_id
        forbidden = sorted(
            str(path)
            for path in run_dir.glob("**/*")
            if path.is_file() and path.name.lower() in _FORBIDDEN_OUTPUT_NAMES
        )
        database = run_dir / "state.sqlite3"
        suspicious_artifacts: list[dict[str, str]] = []
        database_audit: dict[str, Any] = {"database_ready": False}
        if database.is_file():
            connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            try:
                for row in connection.execute("SELECT kind,path FROM artifacts"):
                    kind = str(row["kind"]).lower()
                    path = str(row["path"]).lower()
                    if "submission" in kind or "test_prediction" in kind or "test-prediction" in path:
                        suspicious_artifacts.append(dict(row))
            finally:
                connection.close()
            database_audit = _database_audit(database, self.run_id)
        forbidden_commands: list[str] = []
        for path in self.logs.glob("*.log"):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "--score" in text or "--make" in text:
                forbidden_commands.append(str(path))
        if (
            forbidden
            or suspicious_artifacts
            or forbidden_commands
            or int(database_audit.get("test_metrics", 0))
            or int(database_audit.get("predict_attempts", 0))
        ):
            raise R3EnvelopeError("R3 produced forbidden test/submission artifacts")
        return {
            "validation_only": True,
            "test_prediction_artifacts": 0,
            "submission_artifacts": 0,
            "filesystem_forbidden": forbidden,
            "database_forbidden": suspicious_artifacts,
            "test_metric_rows": database_audit.get("test_metrics", 0),
            "predict_attempt_rows": database_audit.get("predict_attempts", 0),
            "forbidden_command_logs": forbidden_commands,
            "database_audit": database_audit,
        }

    def _winner(self) -> dict[str, Any]:
        manifest = self.runs / self.run_id / "best-valid" / "best_valid_manifest.json"
        if not manifest.is_file():
            raise R3EnvelopeError("completed R3 run did not preserve a best-valid manifest")
        payload = _read_json(manifest)
        if payload.get("kind") != "best_valid" or payload.get("test_prediction_created") is not False:
            raise R3EnvelopeError("best-valid manifest has an invalid identity or test policy")
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise R3EnvelopeError("best-valid manifest has no immutable artifact index")
        root = manifest.parent.resolve()
        verified: dict[str, dict[str, Any]] = {}
        for relative_name, raw_ref in artifacts.items():
            if not isinstance(relative_name, str) or not isinstance(raw_ref, dict):
                raise R3EnvelopeError("best-valid artifact index is malformed")
            relative = Path(relative_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise R3EnvelopeError("best-valid artifact has an unsafe relative path")
            path = (root / relative).resolve()
            if not _is_relative_to(path, root) or not path.is_file() or path.is_symlink():
                raise R3EnvelopeError(f"best-valid artifact is missing: {relative_name}")
            digest = _sha256(path)
            if raw_ref.get("sha256") != digest or raw_ref.get("size_bytes") != path.stat().st_size:
                raise R3EnvelopeError(f"best-valid artifact drifted: {relative_name}")
            if Path(str(raw_ref.get("path", ""))).resolve() != path:
                raise R3EnvelopeError(f"best-valid artifact path disagrees: {relative_name}")
            verified[relative_name] = {"sha256": digest, "size_bytes": path.stat().st_size}
        model_manifest = root / "model/model_bundle.json"
        model = _read_json(model_manifest)
        if (
            model.get("commit_sha") != payload.get("commit_sha")
            or model.get("config_sha256") != payload.get("config_sha256")
        ):
            raise R3EnvelopeError("best-valid model provenance disagrees with its manifest")
        members = model.get("members")
        if not isinstance(members, list) or not members:
            raise R3EnvelopeError("best-valid model bundle has no members")
        for member in members:
            if not isinstance(member, dict):
                raise R3EnvelopeError("best-valid model member is malformed")
            member_path = (model_manifest.parent / str(member.get("name", ""))).resolve()
            if (
                not _is_relative_to(member_path, model_manifest.parent)
                or not member_path.is_file()
                or member.get("sha256") != _sha256(member_path)
                or member.get("size_bytes") != member_path.stat().st_size
            ):
                raise R3EnvelopeError("best-valid model member is missing or corrupt")
        return {
            "path": str(manifest),
            "sha256": _sha256(manifest),
            "preserved": True,
            "model_bundle_sha256": _sha256(model_manifest),
            "artifacts": verified,
        }

    def _seal_manifest(
        self,
        path: Path,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        """Atomically write, contract-validate, and deadline-check the final evidence root."""

        self._require_time("R3 manifest seal")
        value = {
            **manifest,
            "completed_at": _utc_now(),
            "elapsed_seconds": self.monotonic() - self.started_monotonic,
        }
        _atomic_json(path, value)
        try:
            normalized = validate_r3_manifest_with_runtime(
                path,
                python_executable=self._python(),
                source_root=self.clone,
                timeout_seconds=min(30.0, self._remaining()),
            )
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        sealed_elapsed = self.monotonic() - self.started_monotonic
        if sealed_elapsed > self.options.wall_clock_seconds:
            path.unlink(missing_ok=True)
            raise R3EnvelopeError(
                "R3 exceeded its global wall-clock ceiling while validating the evidence seal"
            )
        return normalized

    def _run(self) -> dict[str, Any]:
        source_before = self._resolve_source()
        self.source_snapshot = source_before
        self._prepare_directories()
        self._state("initializing")
        clone_before = self._clone_source()
        self.clone_snapshot = clone_before
        dependency = self._install()
        environment = dict(os.environ)
        environment["REX_SOURCE_COMMIT"] = self.source_commit
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = str(self.clone / "src")
        config_path, preflight = self._preflight(environment)
        self._state("preflight-complete", dependency=dependency, preflight=preflight)
        initial_command = self._run_command(config_path, resume=False)
        resume_command = self._run_command(config_path, resume=True)
        _atomic_json(
            self.runtime / "launch.json",
            {
                "schema_version": "1.0",
                "initial_command": list(initial_command),
                "resume_command": list(resume_command),
                "run_id": self.run_id,
                "external_deadline_epoch_ms": int(self.deadline_epoch * 1000),
                "pre_injection_restart_limit": self.options.pre_injection_restart_limit,
                "stdin": "DEVNULL",
                "validation_only": True,
            },
        )
        first = self._launch(initial_command, environment, "run-initial")
        self._state("running")
        self._snapshot("initial", force=True)
        coordinator = first
        recoveries: list[dict[str, Any]] = []
        previous_progress_token: str | None = None
        while True:
            observed = self._wait_for_injection_point(coordinator)
            if observed is not None:
                lease_path, lease = observed
                break
            if len(recoveries) >= self.options.pre_injection_restart_limit:
                raise R3EnvelopeError(
                    "production run ended before an active durable worker lease was available"
                )
            recovery = self._pre_injection_recovery(
                coordinator,
                restart_number=len(recoveries) + 1,
                previous_progress_token=previous_progress_token,
            )
            previous_progress_token = str(recovery["progress_token"])
            recoveries.append(recovery)
            _atomic_json(
                self.runtime / "pre_injection_recoveries.json",
                {"schema_version": "1.0", "recoveries": recoveries},
            )
            self.fault = {
                **self.fault,
                "state": "recovering-preparation",
                "pre_injection_recoveries": recoveries,
            }
            self._state("recovering-preparation")
            coordinator = self._launch(
                resume_command,
                environment,
                f"run-prelease-resume-{len(recoveries):02d}",
            )
            self._state("running")
        self._inject(coordinator, lease_path, lease)
        resumed = self._launch(resume_command, environment, "run-resumed")
        self.fault = {**self.fault, "state": "resume-started", "resume_pid": resumed.pid}
        self._state("running-resumed")
        return_code = self._wait_for_completion(resumed)
        if return_code != 0:
            raise R3EnvelopeError(f"resumed production run failed with exit {return_code}")
        recovery = self._recovery_audit(lease_path)
        self.fault = {**self.fault, "state": "recovered-and-complete", "recovery": recovery}
        self._state("auditing")
        final_status = self._snapshot("final", force=True)
        if not final_status or final_status.get("run", {}).get("state") != "COMPLETE":
            raise R3EnvelopeError("resumed production run did not reach COMPLETE")
        validation = self._validation_only_audit()
        winner = self._winner()
        clone_after = git_snapshot(self.clone)
        source_after = git_snapshot(self.options.source_root)
        if clone_after != clone_before:
            raise R3EnvelopeError("clean rehearsal source changed during R3")
        if source_after != source_before:
            raise R3EnvelopeError("operator source repository changed during R3")
        if self._remaining() <= 0:
            raise R3EnvelopeError("R3 reached its global ceiling before evidence sealing")
        run_root = self.runs / self.run_id
        evidence_files = [
            self.runtime / "launch.json",
            self.runtime / "fault_injection.json",
            self.runtime / "rehearsal_r3.json",
            self.runtime / "budget.json",
            self.runtime / "data/data_manifest.json",
            self.status_dir / "latest.json",
            run_root / "state.sqlite3",
        ]
        prelease_evidence = self.runtime / "pre_injection_recoveries.json"
        if prelease_evidence.is_file():
            evidence_files.append(prelease_evidence)
        evidence_files.extend(path for path in self.logs.rglob("*") if path.is_file())
        evidence_files.extend(path for path in self.status_dir.rglob("*") if path.is_file())
        evidence_files.extend(
            path
            for path in (
                self.runtime / "envelope_worker_cleanup.json",
                lease_path.with_name("worker_recovery.json"),
                run_root / "state.sqlite3-wal",
                run_root / "state.sqlite3-shm",
            )
            if path.is_file()
        )
        report_dir = run_root / "report"
        report_files: list[Path] = []
        if report_dir.is_dir():
            report_files = [path for path in report_dir.rglob("*") if path.is_file()]
            evidence_files.extend(report_files)
        winner_root = Path(winner["path"]).parent
        evidence_files.extend(path for path in winner_root.rglob("*") if path.is_file())
        evidence_files = sorted(set(evidence_files))
        evidence = {
            str(path.relative_to(self.output)): {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in evidence_files
        }
        database_audit = validation["database_audit"]
        providers = list(database_audit.get("providers", []))
        actual_provider_names = sorted({str(item["provider"]) for item in providers})
        if not actual_provider_names or "fixed" in actual_provider_names:
            raise R3EnvelopeError("R3 did not record an authorized live researcher provider")
        elapsed = self.monotonic() - self.started_monotonic
        if elapsed > self.options.wall_clock_seconds:
            raise R3EnvelopeError("R3 exceeded its global wall-clock ceiling while hashing evidence")
        best_valid_ref = _artifact_payload(Path(winner["path"]), "best_valid_manifest")
        report_refs = [_artifact_payload(path, "r3_report_artifact") for path in report_files]
        environment_inventory = self.logs / "installed-inventory.stdout.log"
        if not environment_inventory.is_file():
            raise R3EnvelopeError("R3 environment inventory evidence is missing")
        manifest = {
            "schema_version": "1.0",
            "level": "R3",
            "rehearsal_id": f"rehearsal-{self.run_id}",
            "state": "COMPLETE",
            "run_id": self.run_id,
            "stop_reason": final_status.get("run", {}).get("stop_reason"),
            "source_commit": self.source_commit,
            "source_tree_sha256": source_before["index_sha256"],
            "environment_lock_sha256": dependency["sha256"],
            "environment_sha256": _sha256(environment_inventory),
            "data_manifest_sha256": preflight["data_manifest_sha256"],
            "starter_manifest_sha256": preflight["starter_manifest_sha256"],
            "evaluator_sha256": preflight["evaluator_sha256"],
            "started_at": datetime.fromtimestamp(self.started_epoch, timezone.utc).isoformat(),
            "started_epoch_ms": int(self.started_epoch * 1000),
            "deadline_epoch_ms": int(self.deadline_epoch * 1000),
            "completed_at": _utc_now(),
            "elapsed_seconds": elapsed,
            "wall_clock_ceiling_seconds": self.options.wall_clock_seconds,
            "within_six_hour_ceiling": elapsed <= MAX_R3_SECONDS,
            "llm": self.options.llm,
            "provider_requested": self.options.llm,
            "provider_actual": ",".join(actual_provider_names),
            "provider_calls": providers,
            "paid_api_authorized": self.options.authorize_paid_api,
            "fault_injected": self.fault.get("count") == 1,
            "fault_recovered": self.fault.get("state") == "recovered-and-complete",
            "source_unchanged": source_after == source_before and clone_after == clone_before,
            "best_valid_manifest": best_valid_ref,
            "report_artifacts": report_refs,
            "test_prediction_created": False,
            "test_scored": False,
            "submission_created": False,
            "dependency": dependency,
            "preflight": preflight,
            "controlled_failure": self.fault,
            "source_audit": {"before": source_before, "after": source_after},
            "clone_audit": {
                "before": clone_before,
                "after": clone_after,
                "protection": self.clone_protection,
            },
            "validation": validation,
            "winner": winner,
            "status": final_status,
            "hourly_snapshot_count": self.snapshot_number,
            "evidence": evidence,
        }
        manifest_path = self.output / "r3_manifest.json"
        manifest = self._seal_manifest(manifest_path, manifest)
        self._state("complete", manifest=str(manifest_path), manifest_sha256=_sha256(manifest_path))
        if self.monotonic() - self.started_monotonic > self.options.wall_clock_seconds:
            manifest_path.unlink(missing_ok=True)
            raise R3EnvelopeError("R3 exceeded its global wall-clock ceiling during final state seal")
        return manifest

    def run(self) -> dict[str, Any]:
        try:
            return self._run()
        except BaseException as error:
            self._terminate_live_processes()
            cleanup: list[dict[str, Any]] = []
            try:
                cleanup = self._recover_active_workers()
            except BaseException as cleanup_error:
                cleanup = [{"error": f"{type(cleanup_error).__name__}: {cleanup_error}"}]
            if self.output.is_dir():
                failure = {
                    "schema_version": "1.0",
                    "state": "FAILED",
                    "failed_at": _utc_now(),
                    "phase": (
                        _read_json(self.output / "envelope_state.json").get("phase")
                        if (self.output / "envelope_state.json").is_file()
                        else "initializing"
                    ),
                    "run_id": self.run_id,
                    "source_commit": self.source_commit,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "worker_cleanup": cleanup,
                    "fault": self.fault,
                }
                _atomic_json(self.output / "r3_failure.json", failure)
                self._state("failed", failure=str(self.output / "r3_failure.json"))
                try:
                    self._snapshot("failure", force=True)
                except BaseException:
                    pass
            raise


def artifact_hashes(paths: Iterable[Path]) -> dict[str, str]:
    """Small public helper used by acceptance tests and external audit tooling."""

    return {str(path): _sha256(path) for path in paths}
