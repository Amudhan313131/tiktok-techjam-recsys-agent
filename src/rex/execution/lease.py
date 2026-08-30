"""Same-host worker process-group leases for crash-safe attempt resumption."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Sequence

import psutil

from rex.data.manifest import canonical_json_bytes
from rex.execution.artifacts import atomic_write_json


class WorkerLeaseError(RuntimeError):
    """A live or ambiguous worker lease made execution unsafe."""


@dataclass
class AttemptLock:
    """An OS-released exclusive lock preventing concurrent same-attempt runners."""

    path: Path
    handle: IO[bytes]

    @classmethod
    def acquire(cls, path: str | Path) -> "AttemptLock":
        candidate = Path(path).resolve()
        candidate.parent.mkdir(parents=True, exist_ok=True)
        handle = candidate.open("a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            raise WorkerLeaseError("another coordinator currently owns this attempt") from error
        return cls(path=candidate, handle=handle)

    def release(self) -> None:
        if self.handle.closed:
            return
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()


def command_sha256(command: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(command))).hexdigest()


def _host_identity() -> dict[str, str | float]:
    return {
        "hostname": socket.gethostname(),
        "boot_time": psutil.boot_time(),
    }


def _process_identity(process: psutil.Process) -> dict[str, Any]:
    command = process.cmdline()
    if not command:
        raise WorkerLeaseError(f"could not observe command identity for worker {process.pid}")
    return {
        "pid": process.pid,
        "pgid": os.getpgid(process.pid),
        "create_time": process.create_time(),
        "command_sha256": command_sha256(command),
        "executable": process.exe(),
    }


def begin_worker_lease(
    path: str | Path,
    *,
    pid: int,
    request_sha256: str,
    execution_sha256: str,
    planned_command_sha256: str,
    identity_token: str | None = None,
) -> dict[str, Any]:
    """Persist the worker identity after spawn and before entering the wait loop."""

    try:
        process = psutil.Process(pid)
    except psutil.Error as error:
        raise WorkerLeaseError(f"could not inspect launched worker {pid}: {error}") from error
    last_error: BaseException | None = None
    identity: dict[str, Any] | None = None
    prior_stable_key: tuple[object, ...] | None = None
    for _ in range(50):
        try:
            candidate = _process_identity(process)
            if identity_token is not None and identity_token not in process.cmdline():
                raise WorkerLeaseError("worker wrapper has not completed exec")
            stable_key = (
                candidate["pid"],
                candidate["pgid"],
                candidate["create_time"],
                candidate["executable"],
            )
            if identity_token is not None and stable_key != prior_stable_key:
                prior_stable_key = stable_key
                time.sleep(0.01)
                continue
            identity = candidate
            break
        except (OSError, psutil.Error, WorkerLeaseError) as error:
            last_error = error
            if not process.is_running():
                break
            time.sleep(0.01)
    if identity is None:
        raise WorkerLeaseError(f"could not establish worker process identity: {last_error}")
    if identity["pgid"] != identity["pid"]:
        raise WorkerLeaseError("worker is not the leader of its isolated process group")
    try:
        owner = psutil.Process(os.getpid())
        owner_create_time = owner.create_time()
    except psutil.Error as error:  # pragma: no cover - the current process must exist
        raise WorkerLeaseError(f"could not inspect coordinator identity: {error}") from error
    marker: dict[str, Any] = {
        "schema_version": "1.0",
        "state": "active",
        **_host_identity(),
        **identity,
        "request_sha256": request_sha256,
        "execution_sha256": execution_sha256,
        "planned_command_sha256": planned_command_sha256,
        "identity_token_sha256": (
            None
            if identity_token is None
            else hashlib.sha256(identity_token.encode("utf-8")).hexdigest()
        ),
        "owner_pid": owner.pid,
        "owner_create_time": owner_create_time,
        "started_at_epoch_ms": int(time.time() * 1000),
    }
    atomic_write_json(path, marker)
    return marker


def close_worker_lease(
    path: str | Path,
    marker: dict[str, Any],
    *,
    reason: str,
    return_code: int | None,
) -> None:
    closed = {
        **marker,
        "state": "closed",
        "closed_reason": reason,
        "return_code": return_code,
        "closed_at_epoch_ms": int(time.time() * 1000),
    }
    atomic_write_json(path, closed)


def _read_marker(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WorkerLeaseError(f"worker lease is unreadable or corrupt: {error}") from error
    required = {
        "state",
        "hostname",
        "boot_time",
        "pid",
        "pgid",
        "create_time",
        "command_sha256",
        "request_sha256",
        "execution_sha256",
    }
    missing = sorted(required - set(value)) if isinstance(value, dict) else sorted(required)
    if missing:
        raise WorkerLeaseError("worker lease is missing fields: " + ", ".join(missing))
    if value["state"] not in {"active", "closed", "recovered"}:
        raise WorkerLeaseError(f"worker lease has unknown state: {value['state']!r}")
    return value


def _group_members(pgid: int) -> list[psutil.Process]:
    members: list[psutil.Process] = []
    for process in psutil.process_iter():
        try:
            if os.getpgid(process.pid) == pgid and process.status() != psutil.STATUS_ZOMBIE:
                members.append(process)
        except (OSError, psutil.Error, ProcessLookupError):
            continue
    return members


def _terminate_group(pgid: int) -> tuple[list[int], str]:
    members = _group_members(pgid)
    member_pids = sorted(process.pid for process in members)
    if not members:
        return member_pids, "already-exited"
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return member_pids, "already-exited"
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if not _group_members(pgid):
            return member_pids, "sigterm"
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return member_pids, "sigterm"
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if not _group_members(pgid):
            return member_pids, "sigkill"
        time.sleep(0.05)
    raise WorkerLeaseError(f"orphan worker process group {pgid} survived SIGKILL")


def _record_recovery(path: Path, event: dict[str, Any]) -> None:
    history: list[dict[str, Any]] = []
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and isinstance(existing.get("events"), list):
                history = existing["events"]
        except (OSError, UnicodeError, json.JSONDecodeError):
            history = []
    atomic_write_json(path, {"schema_version": "1.0", "events": [*history, event]})


def recover_orphan_worker(
    lease_path: str | Path,
    recovery_path: str | Path,
    *,
    request_sha256: str,
    execution_sha256: str,
) -> dict[str, Any] | None:
    """Safely reap a verified orphan, or fail without signaling an uncertain PID."""

    marker_path = Path(lease_path).resolve()
    if not marker_path.is_file():
        return None
    evidence_path = Path(recovery_path).resolve()
    try:
        marker = _read_marker(marker_path)
        if marker["request_sha256"] != request_sha256:
            raise WorkerLeaseError("attempt directory lease belongs to a different request")
        if marker["execution_sha256"] != execution_sha256:
            raise WorkerLeaseError("attempt directory lease belongs to a different execution")
        if marker["state"] != "active":
            return None
        host = _host_identity()
        if marker["hostname"] != host["hostname"] or abs(
            float(marker["boot_time"]) - float(host["boot_time"])
        ) > 0.001:
            raise WorkerLeaseError("active worker lease belongs to a foreign host or boot")
        pid = int(marker["pid"])
        pgid = int(marker["pgid"])
        if pid <= 1 or pgid != pid:
            raise WorkerLeaseError("active worker lease has an unsafe process-group identity")
        try:
            process = psutil.Process(pid)
            observed = _process_identity(process)
        except psutil.NoSuchProcess:
            members = _group_members(pgid)
            if members:
                raise WorkerLeaseError(
                    "worker leader exited while an unverifiable process group remains active"
                )
            event = {
                "recovered_at_epoch_ms": int(time.time() * 1000),
                "outcome": "stale-lease-no-process",
                "pid": pid,
                "pgid": pgid,
                "request_sha256": request_sha256,
            }
            _record_recovery(evidence_path, event)
            atomic_write_json(
                marker_path,
                {
                    **marker,
                    "state": "recovered",
                    "closed_reason": "stale-lease-no-process",
                    "closed_at_epoch_ms": int(time.time() * 1000),
                },
            )
            return event
        except (OSError, psutil.AccessDenied) as error:
            raise WorkerLeaseError(f"could not inspect active worker identity: {error}") from error
        if abs(float(observed["create_time"]) - float(marker["create_time"])) > 0.001:
            raise WorkerLeaseError("worker PID was reused; refusing to signal it")
        if int(observed["pgid"]) != pgid:
            raise WorkerLeaseError("worker process-group identity changed; refusing to signal it")
        token_hash = marker.get("identity_token_sha256")
        if token_hash:
            try:
                observed_tokens = {
                    hashlib.sha256(argument.encode("utf-8")).hexdigest()
                    for argument in process.cmdline()
                }
            except (OSError, psutil.Error) as error:
                raise WorkerLeaseError(
                    f"could not verify worker command token: {error}"
                ) from error
            if token_hash not in observed_tokens:
                raise WorkerLeaseError(
                    "worker command identity token changed; refusing to signal it"
                )
        else:
            if observed["command_sha256"] != marker["command_sha256"]:
                raise WorkerLeaseError("worker command identity changed; refusing to signal it")
            if observed["executable"] != marker.get("executable"):
                raise WorkerLeaseError("worker executable identity changed; refusing to signal it")
        member_pids, termination = _terminate_group(pgid)
        event = {
            "recovered_at_epoch_ms": int(time.time() * 1000),
            "outcome": "orphan-process-group-terminated",
            "termination": termination,
            "pid": pid,
            "pgid": pgid,
            "member_pids": member_pids,
            "request_sha256": request_sha256,
            "command_sha256": marker["command_sha256"],
            "create_time": marker["create_time"],
        }
        _record_recovery(evidence_path, event)
        atomic_write_json(
            marker_path,
            {
                **marker,
                "state": "recovered",
                "closed_reason": "orphan-process-group-terminated",
                "closed_at_epoch_ms": int(time.time() * 1000),
            },
        )
        return event
    except WorkerLeaseError as error:
        _record_recovery(
            evidence_path,
            {
                "recovered_at_epoch_ms": int(time.time() * 1000),
                "outcome": "recovery-refused",
                "reason": str(error),
                "request_sha256": request_sha256,
            },
        )
        raise
