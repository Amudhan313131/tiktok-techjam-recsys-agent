"""Lifecycle contract shared by production execution runtimes.

The contract is intentionally larger than ``subprocess.Popen``: a production
runtime must durably identify a stopped worker before it starts, prove the
worker still has that identity during recovery, and only then collect/remove
it.  Runtime-specific code stays behind :class:`ExecutionRuntime`.
"""

from __future__ import annotations

import os
import platform
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Protocol, runtime_checkable


class ExecutionRuntimeError(RuntimeError):
    """A runtime request could not be executed without weakening isolation."""


class RuntimeKind(StrEnum):
    DOCKER = "docker"
    NATIVE_MACOS = "native_macos"


class RuntimeLifecycleState(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    EXITED = "exited"
    REMOVED = "removed"
    UNKNOWN = "unknown"


class ExecutionOutcome(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    OOM = "oom"
    INTERRUPTED = "interrupted"


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_ATTEMPT_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_SECRET_KEY = re.compile(
    r"(?:^|_)(?:API_?KEY|AUTH|CREDENTIAL|PASSWORD|SECRET|TOKEN)(?:$|_)", re.IGNORECASE
)


def _validate_sha256(value: str, *, label: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ExecutionRuntimeError(f"{label} must be a lowercase SHA-256 hex digest")


def _validate_identity(value: str, *, label: str) -> None:
    if not _SAFE_IDENTITY.fullmatch(value):
        raise ExecutionRuntimeError(f"{label} is not a safe runtime identity")


def _validate_container_path(value: str, *, label: str) -> PurePosixPath:
    if not value.startswith("/") or "\x00" in value or "\n" in value or "\r" in value:
        raise ExecutionRuntimeError(f"{label} must be an absolute container path")
    raw_parts = value.split("/")
    if ".." in raw_parts or "." in raw_parts:
        raise ExecutionRuntimeError(f"{label} contains traversal components")
    path = PurePosixPath(value)
    if path == PurePosixPath("/"):
        raise ExecutionRuntimeError(f"{label} may not be the container root")
    if path.name in {"docker.sock", "containerd.sock"}:
        raise ExecutionRuntimeError(f"{label} may not expose a container runtime socket")
    return path


@dataclass(frozen=True)
class RuntimeMount:
    """A controller-visible path mounted at one exact worker path."""

    source: Path
    target: str
    read_only: bool

    def __post_init__(self) -> None:
        source = Path(self.source)
        if not source.is_absolute():
            raise ExecutionRuntimeError("runtime mount source must be absolute")
        _validate_container_path(self.target, label="runtime mount target")


@dataclass(frozen=True)
class ExecutionSpec:
    """Complete least-authority request for one disposable worker."""

    command: tuple[str, ...]
    working_directory: str
    mounts: tuple[RuntimeMount, ...]
    environment: Mapping[str, str]
    timeout_seconds: float
    memory_bytes: int
    nano_cpus: int
    pids_limit: int
    run_id: str
    experiment_id: str
    attempt_id: str
    request_sha256: str
    execution_sha256: str
    lease_path: Path
    user: str = "10001:10001"
    tmpfs_size_bytes: int = 256 * 1024 * 1024
    # Six workers may run concurrently (three folds, candidate and control).
    # Reserve enough for the model artifacts we actually emit without consuming
    # 3 GiB in sparse reservations before useful work can start.
    output_bytes_limit: int = 128 * 1024 * 1024
    file_size_limit_bytes: int = 64 * 1024 * 1024
    log_bytes_limit: int = 8 * 1024 * 1024
    minimum_free_space_bytes: int = 2 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        if (
            not self.command
            or not self.command[0]
            or any(not isinstance(item, str) or "\x00" in item for item in self.command)
        ):
            raise ExecutionRuntimeError("runtime command must contain non-empty NUL-free arguments")
        _validate_container_path(self.working_directory, label="working directory")
        if not self.mounts:
            raise ExecutionRuntimeError("production worker requires explicit mounts")
        targets = [mount.target for mount in self.mounts]
        if len(targets) != len(set(targets)):
            raise ExecutionRuntimeError("runtime mount targets must be unique")
        writable_mounts = [mount for mount in self.mounts if not mount.read_only]
        if len(writable_mounts) != 1 or writable_mounts[0].target != "/output":
            raise ExecutionRuntimeError(
                "production worker requires exactly one writable mount at /output"
            )
        if self.timeout_seconds <= 0:
            raise ExecutionRuntimeError("runtime timeout must be positive")
        if self.memory_bytes < 64 * 1024 * 1024:
            raise ExecutionRuntimeError("runtime memory limit must be at least 64 MiB")
        if not 1 <= self.nano_cpus <= 64_000_000_000:
            raise ExecutionRuntimeError("runtime CPU quota is outside the supported range")
        if not 1 <= self.pids_limit <= 512:
            raise ExecutionRuntimeError("runtime process limit is outside the supported range")
        if not 1 <= self.tmpfs_size_bytes <= self.memory_bytes:
            raise ExecutionRuntimeError("runtime tmpfs limit must fit within the memory limit")
        if not 1 <= self.output_bytes_limit <= 512 * 1024 * 1024:
            raise ExecutionRuntimeError("runtime output budget is outside the supported range")
        if not 1 <= self.file_size_limit_bytes <= self.output_bytes_limit:
            raise ExecutionRuntimeError(
                "runtime per-file size limit must fit within the output budget"
            )
        if not 1024 <= self.log_bytes_limit <= 64 * 1024 * 1024:
            raise ExecutionRuntimeError("runtime log budget is outside the supported range")
        if not 0 <= self.minimum_free_space_bytes <= 16 * 1024 * 1024 * 1024:
            raise ExecutionRuntimeError(
                "runtime minimum free-space reserve is outside the supported range"
            )
        for value, label in ((self.run_id, "run_id"), (self.experiment_id, "experiment_id")):
            _validate_identity(value, label=label)
        if not _SAFE_ATTEMPT_IDENTITY.fullmatch(self.attempt_id):
            raise ExecutionRuntimeError("attempt_id is not a safe runtime identity")
        _validate_sha256(self.request_sha256, label="request_sha256")
        _validate_sha256(self.execution_sha256, label="execution_sha256")
        if not Path(self.lease_path).is_absolute():
            raise ExecutionRuntimeError("runtime lease path must be absolute")
        if not re.fullmatch(r"[1-9][0-9]{0,9}:[1-9][0-9]{0,9}", self.user):
            raise ExecutionRuntimeError(
                "production worker user must be an explicit non-root UID:GID"
            )
        for key, value in self.environment.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ExecutionRuntimeError("runtime environment must contain only string pairs")
            if not key or "=" in key or "\x00" in key or "\x00" in value:
                raise ExecutionRuntimeError("runtime environment contains an invalid entry")
            if _SECRET_KEY.search(key):
                raise ExecutionRuntimeError(
                    f"worker environment may not contain secret key {key!r}"
                )

    @property
    def identity_labels(self) -> dict[str, str]:
        return {
            "rex.managed": "true",
            "rex.run_id": self.run_id,
            "rex.experiment_id": self.experiment_id,
            "rex.attempt_id": self.attempt_id,
            "rex.request_sha256": self.request_sha256,
            "rex.execution_sha256": self.execution_sha256,
        }


def assert_no_controller_secret_leakage(
    specification: ExecutionSpec,
    controller_environment: Mapping[str, str] | None = None,
) -> None:
    """Reject a worker request containing a controller credential value or name."""

    source = os.environ if controller_environment is None else controller_environment
    serialized = "\x00".join(
        (
            *specification.command,
            *(f"{key}={value}" for key, value in specification.environment.items()),
        )
    )
    forbidden_names = {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "CLAUDE_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "GITHUB_TOKEN",
        "SSH_AUTH_SOCK",
    }
    if any(name in serialized for name in forbidden_names):
        raise ExecutionRuntimeError(
            "worker command or environment references a controller credential"
        )
    for key, value in source.items():
        if (
            (_SECRET_KEY.search(key) or key in forbidden_names)
            and len(value) >= 8
            and value in serialized
        ):
            raise ExecutionRuntimeError(f"worker request contains the controller credential {key}")


@dataclass(frozen=True)
class RuntimeHandle:
    runtime_kind: RuntimeKind
    container_id: str
    container_name: str
    worker_image_digest: str
    worker_image_id: str
    daemon_identity: str
    run_id: str
    experiment_id: str
    attempt_id: str
    request_sha256: str
    execution_sha256: str
    lease_path: Path
    timeout_seconds: float
    started_at_epoch_ms: int


@dataclass(frozen=True)
class RuntimeStatus:
    state: RuntimeLifecycleState
    exit_code: int | None = None
    oom_killed: bool = False
    detail: str = ""


@dataclass(frozen=True)
class ExecutionResult:
    outcome: ExecutionOutcome
    exit_code: int | None
    stdout: str
    stderr: str
    wall_seconds: float
    oom_killed: bool = False
    timed_out: bool = False


@dataclass(frozen=True)
class ExecutionLease:
    schema_version: str
    runtime_kind: RuntimeKind
    state: str
    container_id: str
    container_name: str
    worker_image_digest: str
    worker_image_id: str
    daemon_identity: str
    run_id: str
    experiment_id: str
    attempt_id: str
    request_sha256: str
    execution_sha256: str
    created_at_epoch_ms: int
    started_at_epoch_ms: int | None = None
    closed_at_epoch_ms: int | None = None
    closed_reason: str | None = None
    exit_code: int | None = None
    lease_path: Path | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("lease_path", None)
        payload["runtime_kind"] = self.runtime_kind.value
        return payload


@dataclass(frozen=True)
class RecoveryResult:
    outcome: str
    container_id: str
    terminated: bool
    detail: str = ""


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class DoctorResult:
    runtime_kind: RuntimeKind
    available: bool
    safe_for_production: bool
    checks: tuple[DoctorCheck, ...] = field(default_factory=tuple)
    detail: str = ""
    environment_identity: Mapping[str, str] = field(default_factory=dict)


@runtime_checkable
class ExecutionRuntime(Protocol):
    def doctor(self) -> DoctorResult: ...

    def launch(self, specification: ExecutionSpec) -> RuntimeHandle: ...

    def inspect(self, handle: RuntimeHandle) -> RuntimeStatus: ...

    def terminate(self, handle: RuntimeHandle) -> None: ...

    def collect(self, handle: RuntimeHandle) -> ExecutionResult: ...

    def recover(self, lease: ExecutionLease) -> RecoveryResult: ...

    def cleanup(self, handle: RuntimeHandle) -> None: ...


def production_runtime(
    *,
    docker_client: object | None = None,
    environ: Mapping[str, str] | None = None,
    native_macos_factory: Callable[[], ExecutionRuntime] | None = None,
) -> ExecutionRuntime:
    """Select the production runtime; Docker is the mandatory default.

    Native macOS execution is an explicitly gated, temporary rollback path.  A
    caller must provide its adapter rather than the selector silently falling
    back to an unsandboxed subprocess implementation.
    """

    values = os.environ if environ is None else environ
    selected = values.get("REX_PRODUCTION_RUNTIME", RuntimeKind.DOCKER.value).strip().lower()
    if selected == RuntimeKind.DOCKER.value:
        from rex.execution.runtime_docker import DockerCLIClient, DockerExecutionRuntime

        image_reference = values.get("REX_WORKER_IMAGE", "").strip()
        if not image_reference:
            raise ExecutionRuntimeError("REX_WORKER_IMAGE must pin the production worker image")
        controller_id = values.get(
            "REX_CONTROLLER_ID",
            values.get("REX_CONTROLLER_CONTAINER_ID", values.get("HOSTNAME", "")),
        ).strip()
        if not controller_id:
            raise ExecutionRuntimeError("REX_CONTROLLER_ID must identify the controller container")
        client = DockerCLIClient() if docker_client is None else docker_client
        return DockerExecutionRuntime(
            client=client,
            image_reference=image_reference,
            controller_id=controller_id,
        )
    if selected == RuntimeKind.NATIVE_MACOS.value:
        if values.get("REX_ALLOW_NATIVE_MACOS_ROLLBACK") != "1":
            raise ExecutionRuntimeError("native macOS rollback requires explicit authorization")
        if platform.system() != "Darwin":
            raise ExecutionRuntimeError("native macOS rollback is unavailable on this platform")
        if native_macos_factory is None:
            raise ExecutionRuntimeError(
                "native macOS rollback requires an explicit runtime adapter"
            )
        runtime = native_macos_factory()
        if not isinstance(runtime, ExecutionRuntime):
            raise ExecutionRuntimeError("native macOS adapter does not implement ExecutionRuntime")
        return runtime
    raise ExecutionRuntimeError(f"unsupported production runtime: {selected!r}")
