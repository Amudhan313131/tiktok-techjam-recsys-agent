"""Fail-closed execution sandbox abstraction and capability policy."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable, Mapping, Sequence

from rex.data.manifest import canonical_json_bytes
from rex.execution.limits import ResourceLimits


class SandboxError(RuntimeError):
    """The requested isolation policy could not be prepared safely."""


class SandboxMode(StrEnum):
    """Fixture execution is trusted; production execution must be OS-sandboxed."""

    FIXTURE = "fixture"
    PRODUCTION = "production"


@dataclass(frozen=True)
class SandboxDoctorResult:
    backend: str
    available: bool
    safe_for_production: bool
    detail: str

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


@dataclass(frozen=True)
class SandboxPolicy:
    """The complete filesystem/network authority granted to one worker."""

    workspace: Path
    read_paths: tuple[Path, ...]
    write_paths: tuple[Path, ...]
    network_allowed: bool
    resource_limits: ResourceLimits

    def __post_init__(self) -> None:
        if self.network_allowed:
            raise SandboxError("production worker network access is not supported")
        if not self.workspace.is_absolute():
            raise SandboxError("sandbox workspace must be absolute")
        if not self.read_paths:
            raise SandboxError("sandbox requires at least one readable capability")
        if not self.write_paths:
            raise SandboxError("sandbox requires at least one writable capability")
        for path in (*self.read_paths, *self.write_paths):
            if not path.is_absolute():
                raise SandboxError(f"sandbox capability must be absolute: {path}")

    def to_dict(self) -> dict[str, object]:
        return {
            "workspace": str(self.workspace),
            "read_paths": [str(path) for path in self.read_paths],
            "write_paths": [str(path) for path in self.write_paths],
            "network_allowed": self.network_allowed,
            "resource_limits": self.resource_limits.to_dict(),
        }


@dataclass(frozen=True)
class PreparedSandbox:
    command: tuple[str, ...]
    environment: dict[str, str]
    preexec_fn: Callable[[], None] | None
    evidence: dict[str, object]


class SandboxBackend:
    name = "abstract"

    def doctor(self) -> SandboxDoctorResult:
        raise NotImplementedError

    def prepare(
        self,
        policy: SandboxPolicy,
        command: Sequence[str],
        environment: Mapping[str, str],
        policy_path: Path,
    ) -> PreparedSandbox:
        raise NotImplementedError


SAFE_ENVIRONMENT_KEYS = frozenset(
    {
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "PYTHONHASHSEED",
        "SYSTEM_VERSION_COMPAT",
        "TZ",
    }
)


def sanitized_environment(
    source: Mapping[str, str] | None = None,
    *,
    workspace: Path,
    temp_dir: Path,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a worker environment without credentials or user-home capabilities."""

    source = os.environ if source is None else source
    environment = {key: source[key] for key in SAFE_ENVIRONMENT_KEYS if key in source}
    environment.update(
        {
            "PYTHONPATH": str(workspace / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": str(temp_dir),
            "HOME": str(temp_dir),
            "MPLCONFIGDIR": str(temp_dir / "matplotlib"),
            "XDG_CACHE_HOME": str(temp_dir / "cache"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    if overrides:
        forbidden = sorted(set(overrides) - SAFE_ENVIRONMENT_KEYS)
        if forbidden:
            raise SandboxError(
                "unsafe production environment override(s): " + ", ".join(forbidden)
            )
        environment.update(overrides)
    return environment


def fixture_environment(workspace: Path) -> dict[str, str]:
    """Preserve the historic trusted-fixture environment explicitly."""

    environment = os.environ.copy()
    source_root = str(workspace / "src")
    environment["PYTHONPATH"] = (
        source_root
        if not environment.get("PYTHONPATH")
        else source_root + os.pathsep + environment["PYTHONPATH"]
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def policy_sha256(policy: SandboxPolicy) -> str:
    import hashlib

    return hashlib.sha256(canonical_json_bytes(policy.to_dict())).hexdigest()


def production_backend() -> SandboxBackend:
    """Select a supported production backend, failing closed on other hosts."""

    if platform.system() == "Darwin":
        from rex.execution.sandbox_macos import MacOSSandboxBackend

        return MacOSSandboxBackend()
    raise SandboxError(
        f"no production sandbox backend is available for {platform.system() or sys.platform}"
    )
