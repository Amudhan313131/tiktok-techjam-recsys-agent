"""Durable, exact-identity leases for Docker workers."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Mapping

from rex.data.manifest import sha256_file
from rex.execution.artifacts import atomic_write_json
from rex.execution.runtime import (
    ExecutionLease,
    ExecutionRuntimeError,
    ExecutionSpec,
    RuntimeHandle,
    RuntimeKind,
)


class DockerLeaseError(ExecutionRuntimeError):
    """A Docker lease is corrupt, mismatched, or unsafe to recover."""


_CONTAINER_ID = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ATTEMPT_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STATES = {"created", "active", "closed", "recovered"}


def _validate_lease(lease: ExecutionLease) -> None:
    if lease.schema_version != "1.0":
        raise DockerLeaseError(f"unsupported Docker lease schema: {lease.schema_version!r}")
    if lease.runtime_kind != RuntimeKind.DOCKER:
        raise DockerLeaseError("active native lease cannot resume through Docker")
    if lease.state not in _STATES:
        raise DockerLeaseError(f"Docker lease has unknown state: {lease.state!r}")
    if not _CONTAINER_ID.fullmatch(lease.container_id):
        raise DockerLeaseError("Docker lease has an invalid full container ID")
    if not _IMAGE_ID.fullmatch(lease.worker_image_id):
        raise DockerLeaseError("Docker lease has an invalid image ID")
    if not _IMAGE_ID.fullmatch(lease.worker_image_digest):
        raise DockerLeaseError("Docker lease has an invalid immutable image digest")
    if not lease.container_name or not lease.daemon_identity:
        raise DockerLeaseError("Docker lease is missing its daemon/container identity")
    if not _IMAGE_ID.fullmatch(lease.daemon_identity):
        raise DockerLeaseError("Docker lease has an invalid daemon identity")
    for value, label in ((lease.run_id, "run_id"), (lease.experiment_id, "experiment_id")):
        if not _IDENTITY.fullmatch(value):
            raise DockerLeaseError(f"Docker lease has an invalid {label}")
    if not _ATTEMPT_IDENTITY.fullmatch(lease.attempt_id):
        raise DockerLeaseError("Docker lease has an invalid attempt_id")
    for value, label in (
        (lease.request_sha256, "request_sha256"),
        (lease.execution_sha256, "execution_sha256"),
    ):
        if not _SHA256.fullmatch(value):
            raise DockerLeaseError(f"Docker lease has an invalid {label}")
    if lease.created_at_epoch_ms <= 0:
        raise DockerLeaseError("Docker lease has an invalid creation time")
    if lease.state == "active" and lease.started_at_epoch_ms is None:
        raise DockerLeaseError("active Docker lease has no start time")
    if lease.state in {"closed", "recovered"} and (
        lease.closed_at_epoch_ms is None or lease.closed_reason is None
    ):
        raise DockerLeaseError("closed Docker lease has no closure evidence")


def create_docker_lease(specification: ExecutionSpec, handle: RuntimeHandle) -> ExecutionLease:
    lease = ExecutionLease(
        schema_version="1.0",
        runtime_kind=RuntimeKind.DOCKER,
        state="created",
        container_id=handle.container_id,
        container_name=handle.container_name,
        worker_image_digest=handle.worker_image_digest,
        worker_image_id=handle.worker_image_id,
        daemon_identity=handle.daemon_identity,
        run_id=specification.run_id,
        experiment_id=specification.experiment_id,
        attempt_id=specification.attempt_id,
        request_sha256=specification.request_sha256,
        execution_sha256=specification.execution_sha256,
        created_at_epoch_ms=int(time.time() * 1000),
        lease_path=Path(specification.lease_path).resolve(),
    )
    _validate_lease(lease)
    return lease


def persist_docker_lease(path: str | Path, lease: ExecutionLease) -> None:
    _validate_lease(lease)
    atomic_write_json(path, lease.to_dict())


def read_docker_lease(path: str | Path) -> ExecutionLease:
    candidate = Path(path)
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DockerLeaseError(f"Docker lease is unreadable or corrupt: {error}") from error
    if not isinstance(payload, Mapping):
        raise DockerLeaseError("Docker lease must be a JSON object")
    required = {
        "schema_version",
        "runtime_kind",
        "state",
        "container_id",
        "container_name",
        "worker_image_digest",
        "worker_image_id",
        "daemon_identity",
        "run_id",
        "experiment_id",
        "attempt_id",
        "request_sha256",
        "execution_sha256",
        "created_at_epoch_ms",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise DockerLeaseError("Docker lease is missing fields: " + ", ".join(missing))
    allowed = required | {
        "started_at_epoch_ms",
        "closed_at_epoch_ms",
        "closed_reason",
        "exit_code",
    }
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise DockerLeaseError("Docker lease has unexpected fields: " + ", ".join(unexpected))
    try:
        lease = ExecutionLease(
            schema_version=str(payload["schema_version"]),
            runtime_kind=RuntimeKind(str(payload["runtime_kind"])),
            state=str(payload["state"]),
            container_id=str(payload["container_id"]),
            container_name=str(payload["container_name"]),
            worker_image_digest=str(payload["worker_image_digest"]),
            worker_image_id=str(payload["worker_image_id"]),
            daemon_identity=str(payload["daemon_identity"]),
            run_id=str(payload["run_id"]),
            experiment_id=str(payload["experiment_id"]),
            attempt_id=str(payload["attempt_id"]),
            request_sha256=str(payload["request_sha256"]),
            execution_sha256=str(payload["execution_sha256"]),
            created_at_epoch_ms=int(payload["created_at_epoch_ms"]),
            started_at_epoch_ms=_optional_int(payload.get("started_at_epoch_ms")),
            closed_at_epoch_ms=_optional_int(payload.get("closed_at_epoch_ms")),
            closed_reason=_optional_str(payload.get("closed_reason")),
            exit_code=_optional_int(payload.get("exit_code")),
            lease_path=candidate.resolve(),
        )
    except (TypeError, ValueError) as error:
        raise DockerLeaseError(f"Docker lease fields are invalid: {error}") from error
    _validate_lease(lease)
    return lease


def archive_closed_docker_lease(
    path: str | Path,
    *,
    related_evidence_paths: tuple[str | Path, ...] = (),
) -> dict[str, object]:
    """Preserve a closed lease before an exact-identity worker relaunch.

    A closed/recovered lease normally proves that collection completed.  If the
    exact container has already been removed but the caller's durable result is
    absent, the only safe retry is to archive that evidence and free the
    canonical lease path for a new disposable worker.  The archive name is
    content addressed, making repeated recovery idempotent.
    """

    raw_candidate = Path(path)
    if raw_candidate.is_symlink():
        raise DockerLeaseError("closed Docker lease archive source is not a regular file")
    candidate = raw_candidate.resolve()
    if not candidate.is_file():
        raise DockerLeaseError("closed Docker lease archive source is not a regular file")
    lease = read_docker_lease(candidate)
    if lease.state not in {"closed", "recovered"}:
        raise DockerLeaseError("only a closed Docker lease may be archived for relaunch")
    lease_sha256 = sha256_file(candidate)
    archive_root = candidate.parent / "lease-archive"
    if archive_root.is_symlink():
        raise DockerLeaseError("closed Docker lease archive directory may not be a symlink")
    archive_root.mkdir(parents=True, exist_ok=True)
    if not archive_root.is_dir():
        raise DockerLeaseError("closed Docker lease archive path is not a directory")
    archived_lease = archive_root / (
        f"{candidate.stem}-{lease.container_id[:12]}-{lease_sha256[:16]}.json"
    )
    lease_payload = json.loads(candidate.read_text(encoding="utf-8"))
    if archived_lease.is_symlink():
        raise DockerLeaseError("closed Docker lease archive may not be a symlink")
    if archived_lease.is_file():
        if sha256_file(archived_lease) != lease_sha256:
            raise DockerLeaseError("closed Docker lease archive collision")
    else:
        atomic_write_json(archived_lease, lease_payload)
        if sha256_file(archived_lease) != lease_sha256:
            raise DockerLeaseError("closed Docker lease archive verification failed")

    archived_related: list[dict[str, str]] = []
    for raw_related in related_evidence_paths:
        raw_related_path = Path(raw_related)
        if raw_related_path.is_symlink():
            raise DockerLeaseError("Docker recovery evidence is not a regular file")
        related = raw_related_path.resolve()
        if not related.exists():
            continue
        if not related.is_file():
            raise DockerLeaseError("Docker recovery evidence is not a regular file")
        related_sha256 = sha256_file(related)
        destination = archive_root / (
            f"{related.stem}-{lease.container_id[:12]}-{related_sha256[:16]}{related.suffix}"
        )
        if destination.is_symlink():
            raise DockerLeaseError("Docker recovery evidence archive may not be a symlink")
        if destination.is_file():
            if sha256_file(destination) != related_sha256:
                raise DockerLeaseError("Docker recovery evidence archive collision")
        else:
            try:
                payload = json.loads(related.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise DockerLeaseError(
                    f"Docker recovery evidence is unreadable or corrupt: {error}"
                ) from error
            atomic_write_json(destination, payload)
            if sha256_file(destination) != related_sha256:
                raise DockerLeaseError("Docker recovery evidence archive verification failed")
        archived_related.append({"path": str(destination), "sha256": related_sha256})

    reservation = candidate.with_name(candidate.name + ".output-reservation")
    if reservation.is_symlink():
        raise DockerLeaseError("Docker output reservation may not be a symlink")
    if reservation.exists() and not reservation.is_file():
        raise DockerLeaseError("Docker output reservation is not a regular file")
    reservation.unlink(missing_ok=True)
    candidate.unlink()
    if candidate.exists():  # pragma: no cover - defensive filesystem invariant
        raise DockerLeaseError("closed Docker lease could not be released for relaunch")
    return {
        "lease_path": str(archived_lease),
        "lease_sha256": lease_sha256,
        "container_id": lease.container_id,
        "state": lease.state,
        "related_evidence": archived_related,
    }


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def mark_docker_lease_started(path: str | Path, lease: ExecutionLease) -> ExecutionLease:
    if lease.state != "created":
        raise DockerLeaseError("only a created Docker lease can be started")
    updated = ExecutionLease(
        **{
            **lease.to_dict(),
            "runtime_kind": RuntimeKind.DOCKER,
            "state": "active",
            "started_at_epoch_ms": int(time.time() * 1000),
            "lease_path": lease.lease_path,
        }
    )
    persist_docker_lease(path, updated)
    return updated


def close_docker_lease(
    path: str | Path,
    lease: ExecutionLease,
    *,
    reason: str,
    exit_code: int | None,
    recovered: bool = False,
) -> ExecutionLease:
    if lease.state not in {"created", "active"}:
        raise DockerLeaseError("only an open Docker lease can be closed")
    if not reason or any(character in reason for character in "\x00\n\r"):
        raise DockerLeaseError("Docker lease closure reason is invalid")
    updated = ExecutionLease(
        **{
            **lease.to_dict(),
            "runtime_kind": RuntimeKind.DOCKER,
            "state": "recovered" if recovered else "closed",
            "closed_at_epoch_ms": int(time.time() * 1000),
            "closed_reason": reason,
            "exit_code": exit_code,
            "lease_path": lease.lease_path,
        }
    )
    persist_docker_lease(path, updated)
    return updated


def verify_lease_handle(lease: ExecutionLease, handle: RuntimeHandle) -> None:
    _validate_lease(lease)
    expected = {
        "runtime_kind": handle.runtime_kind,
        "container_id": handle.container_id,
        "container_name": handle.container_name,
        "worker_image_digest": handle.worker_image_digest,
        "worker_image_id": handle.worker_image_id,
        "daemon_identity": handle.daemon_identity,
        "run_id": handle.run_id,
        "experiment_id": handle.experiment_id,
        "attempt_id": handle.attempt_id,
        "request_sha256": handle.request_sha256,
        "execution_sha256": handle.execution_sha256,
    }
    for field, value in expected.items():
        if getattr(lease, field) != value:
            raise DockerLeaseError(f"Docker lease {field} does not match the runtime handle")


def runtime_handle_from_docker_lease(
    lease: ExecutionLease,
    *,
    timeout_seconds: float,
) -> RuntimeHandle:
    """Reconstruct the exact handle needed to collect/clean a leased container.

    The timeout is supplied by the trusted caller because a lease intentionally
    stores identity, not a renewable wall-clock budget.  It remains bounded by
    the production six-hour ceiling.
    """

    _validate_lease(lease)
    if lease.lease_path is None or not lease.lease_path.is_absolute():
        raise DockerLeaseError("Docker lease has no validated source path")
    if not 0 < timeout_seconds <= 21_600:
        raise DockerLeaseError("Docker lease handle timeout is outside the production bound")
    return RuntimeHandle(
        runtime_kind=RuntimeKind.DOCKER,
        container_id=lease.container_id,
        container_name=lease.container_name,
        worker_image_digest=lease.worker_image_digest,
        worker_image_id=lease.worker_image_id,
        daemon_identity=lease.daemon_identity,
        run_id=lease.run_id,
        experiment_id=lease.experiment_id,
        attempt_id=lease.attempt_id,
        request_sha256=lease.request_sha256,
        execution_sha256=lease.execution_sha256,
        lease_path=lease.lease_path,
        timeout_seconds=float(timeout_seconds),
        started_at_epoch_ms=lease.started_at_epoch_ms or lease.created_at_epoch_ms,
    )
