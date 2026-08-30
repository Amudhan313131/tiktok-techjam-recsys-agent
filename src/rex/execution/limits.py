"""POSIX resource limits applied before an isolated worker is executed."""

from __future__ import annotations

import math
import os
import resource
import sys
from dataclasses import asdict, dataclass
from typing import Callable


@dataclass(frozen=True)
class ResourceLimits:
    """Hard ceilings independent of the runner's wall-clock/process-tree monitor."""

    cpu_seconds: int
    address_space_bytes: int | None
    open_files: int = 256
    processes: int = 64
    file_size_bytes: int = 4 * 1024 * 1024 * 1024
    core_size_bytes: int = 0

    def to_dict(self) -> dict[str, int | None]:
        return asdict(self)


def limits_for_request(timeout_seconds: int, max_memory_mb: int | None) -> ResourceLimits:
    """Derive child-process limits while leaving room for orderly timeout handling."""

    # RLIMIT_NPROC is per real user, not per process tree. Anchor its absolute
    # ceiling to the coordinator's observed user-process count so a busy desktop
    # can still launch the worker while the candidate receives only bounded headroom.
    user_processes = 0
    try:
        import psutil

        uid = os.getuid()
        for process in psutil.process_iter():
            try:
                user_processes += int(process.uids().real == uid)
            except (psutil.Error, ProcessLookupError):
                continue
    except (ImportError, OSError):  # pragma: no cover - psutil is a runtime dependency
        user_processes = 256
    return ResourceLimits(
        cpu_seconds=max(1, math.ceil(timeout_seconds) + 2),
        # Darwin exposes RLIMIT_AS but rejects every finite value with
        # ``EINVAL`` (surfaced by Python as ``ValueError: current limit exceeds
        # maximum limit``).  Passing that unusable limit through ``preexec_fn``
        # prevents the sandboxed worker from launching at all.  The runner's
        # process-tree RSS sampler still enforces ``max_memory_mb`` on macOS;
        # supported POSIX hosts retain the kernel address-space ceiling.
        address_space_bytes=(
            None
            if max_memory_mb is None or sys.platform == "darwin"
            else int(max_memory_mb) * 1024 * 1024
        ),
        processes=max(64, user_processes + 64),
    )


def _bounded_limit(resource_name: int, requested: int) -> tuple[int, int]:
    _soft, hard = resource.getrlimit(resource_name)
    if hard == resource.RLIM_INFINITY:
        return requested, hard
    return min(requested, hard), hard


def apply_resource_limits(limits: ResourceLimits) -> None:
    """Apply limits in the child immediately before exec.

    The hard value inherited from the coordinator is retained. This avoids a child
    accidentally trying to raise a limit, which would make ``Popen`` fail on hosts
    where the coordinator lacks that privilege.
    """

    bounded = (
        (resource.RLIMIT_CPU, limits.cpu_seconds),
        (resource.RLIMIT_NOFILE, limits.open_files),
        (resource.RLIMIT_FSIZE, limits.file_size_bytes),
        (resource.RLIMIT_CORE, limits.core_size_bytes),
    )
    if hasattr(resource, "RLIMIT_NPROC"):
        bounded += ((resource.RLIMIT_NPROC, limits.processes),)
    if limits.address_space_bytes is not None and hasattr(resource, "RLIMIT_AS"):
        bounded += ((resource.RLIMIT_AS, limits.address_space_bytes),)
    for resource_name, requested in bounded:
        resource.setrlimit(resource_name, _bounded_limit(resource_name, requested))


def resource_limit_preexec(limits: ResourceLimits) -> Callable[[], None]:
    """Return the small pre-exec hook used by ``subprocess.Popen``."""

    def apply() -> None:
        apply_resource_limits(limits)

    return apply
