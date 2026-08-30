"""Fail-closed Docker implementation of the production execution runtime."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol, Sequence

from rex.data.manifest import canonical_json_bytes
from rex.execution.docker_lease import (
    DockerLeaseError,
    close_docker_lease,
    create_docker_lease,
    mark_docker_lease_started,
    persist_docker_lease,
    read_docker_lease,
    runtime_handle_from_docker_lease,
    verify_lease_handle,
)
from rex.execution.docker_mounts import DockerMountResolver, ResolvedDockerMount
from rex.execution.runtime import (
    DoctorCheck,
    DoctorResult,
    ExecutionLease,
    ExecutionOutcome,
    ExecutionResult,
    ExecutionRuntimeError,
    ExecutionSpec,
    RecoveryResult,
    RuntimeHandle,
    RuntimeKind,
    RuntimeLifecycleState,
    RuntimeMount,
    RuntimeStatus,
    assert_no_controller_secret_leakage,
)


class DockerRuntimeError(ExecutionRuntimeError):
    """The Docker daemon or worker violated the production contract."""


class DockerWaitTimeout(DockerRuntimeError):
    """A Docker worker exceeded its wall-clock limit."""


class DockerContainerNotFound(DockerRuntimeError):
    """The daemon proved that one exact full container ID no longer exists."""

    def __init__(self, container_id: str) -> None:
        super().__init__(f"No such container: {container_id}")
        self.container_id = container_id


class DockerOutputLimitExceeded(DockerRuntimeError):
    """A worker exceeded its aggregate controller-side output budget."""


@dataclass(frozen=True)
class DockerCreateRequest:
    name: str
    image_reference: str
    command: tuple[str, ...]
    working_directory: str
    user: str
    environment: Mapping[str, str]
    labels: Mapping[str, str]
    mounts: tuple[ResolvedDockerMount, ...]
    memory_bytes: int
    nano_cpus: int
    pids_limit: int
    tmpfs_size_bytes: int
    output_bytes_limit: int
    file_size_limit_bytes: int
    log_bytes_limit: int
    minimum_free_space_bytes: int


class DockerClient(Protocol):
    def daemon_info(self) -> Mapping[str, Any]: ...

    def inspect_image(self, image_reference: str) -> Mapping[str, Any]: ...

    def inspect_container(self, container_id: str) -> Mapping[str, Any]: ...

    def create_container(self, request: DockerCreateRequest) -> str: ...

    def start_container(self, container_id: str) -> None: ...

    def wait_container(self, container_id: str, timeout_seconds: float) -> int: ...

    def container_logs(self, container_id: str, max_bytes: int) -> tuple[bytes, bytes]: ...

    def stop_container(self, container_id: str, timeout_seconds: int) -> None: ...

    def kill_container(self, container_id: str) -> None: ...

    def remove_container(self, container_id: str, *, force: bool) -> None: ...


_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONTAINER_ID = re.compile(r"^[0-9a-f]{64}$")
_NO_SUCH_CONTAINER = re.compile(
    r"^(?:Error response from daemon: )?No such container: (?P<id>[0-9a-f]{64})$"
)
_SECRET_ENV = re.compile(
    r"(?:^|_)(?:API_?KEY|AUTH_TOKEN|CREDENTIAL|PASSWORD|PRIVATE_KEY|SECRET|TOKEN)(?:$|_)",
    re.IGNORECASE,
)
_IMAGE_LABELS = {
    "source_git_commit": "org.opencontainers.image.revision",
    "dependency_lock_sha256": "org.rex.dependency-lock-sha256",
    "pyproject_sha256": "org.rex.pyproject-sha256",
    "starter_kit_manifest_sha256": "org.rex.starter-kit-sha256",
    "base_image_digest": "org.rex.base-image-digest",
}
_DOCKER_ENVIRONMENT_KEYS = frozenset(
    {
        "DOCKER_CONFIG",
        "DOCKER_CONTEXT",
        "DOCKER_HOST",
        "DOCKER_TLS",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
        "HOME",
        "PATH",
        "SYSTEMROOT",
        "TMPDIR",
        "XDG_RUNTIME_DIR",
    }
)
_SECURITY_POLICY = {
    "schema_version": "1.0",
    "network_mode": "none",
    "read_only_rootfs": True,
    "user": "explicit-non-root",
    "capabilities": "drop-all",
    "no_new_privileges": True,
    "privileged": False,
    "docker_socket": False,
    "single_writable_bind": True,
    "writable_bind_target": "/output",
    "bounded_log_driver": "local:max-file=1",
    "controller_output_budget": True,
    "tmpfs": "/tmp:rw,noexec,nosuid,nodev",
    "entrypoint": "/usr/bin/tini --",
    "resource_limits": (
        "memory",
        "nano_cpus",
        "pids",
        "nofile",
        "core",
        "fsize",
        "logs",
        "aggregate_output",
    ),
}

_LOG_TRUNCATION_MARKER = b"\n[rex: docker log truncated to configured byte limit]\n"


def docker_security_policy_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(_SECURITY_POLICY)).hexdigest()


def _pinned_digest(image_reference: str) -> str:
    if _DIGEST.fullmatch(image_reference):
        return image_reference
    name, separator, digest = image_reference.rpartition("@")
    if not separator or not name or not _DIGEST.fullmatch(digest):
        raise DockerRuntimeError(
            "production worker image must be an immutable name@sha256:<64-hex> reference"
        )
    if (
        any(character.isspace() for character in name)
        or any(character in name for character in "\x00\n\r,@\\")
        or "://" in name
    ):
        raise DockerRuntimeError("production worker image name is invalid")
    return digest


def _daemon_identity(info: Mapping[str, Any]) -> str:
    identity = {
        key: info.get(key)
        for key in ("ID", "Name", "ServerVersion", "OperatingSystem", "Architecture", "OSType")
    }
    if identity["OSType"] != "linux" or not identity["ServerVersion"]:
        raise DockerRuntimeError("production requires a reachable Linux Docker Engine")
    if not identity["ID"] or not identity["Name"] or not identity["Architecture"]:
        raise DockerRuntimeError("Docker Engine did not expose a stable daemon identity")
    security_options = info.get("SecurityOptions")
    if not isinstance(security_options, list) or not any(
        "seccomp" in str(value).lower() for value in security_options
    ):
        raise DockerRuntimeError("Docker Engine did not advertise seccomp isolation")
    return "sha256:" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def _normalize_digest_label(value: object, *, label: str, allow_prefixed: bool = False) -> str:
    if not isinstance(value, str):
        raise DockerRuntimeError(f"worker image is missing required label {label}")
    candidate = value[7:] if allow_prefixed and value.startswith("sha256:") else value
    if not re.fullmatch(r"[0-9a-f]{64}", candidate):
        raise DockerRuntimeError(f"worker image label {label} is not a SHA-256 digest")
    return ("sha256:" if allow_prefixed else "") + candidate


def _decode(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def _bounded_bytes(value: bytes, max_bytes: int) -> bytes:
    """Return a diagnostic tail without ever retaining more than the declared cap."""

    if max_bytes <= 0:
        raise DockerRuntimeError("Docker log byte limit must be positive")
    if len(value) <= max_bytes:
        return value
    if max_bytes <= len(_LOG_TRUNCATION_MARKER):
        return _LOG_TRUNCATION_MARKER[-max_bytes:]
    return _LOG_TRUNCATION_MARKER + value[-(max_bytes - len(_LOG_TRUNCATION_MARKER)) :]


def _read_bounded_file(stream: Any, max_bytes: int) -> bytes:
    """Read only a bounded tail from a seekable temporary log spool."""

    stream.flush()
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    if size <= max_bytes:
        stream.seek(0)
        return stream.read(max_bytes)
    retained = max(0, max_bytes - len(_LOG_TRUNCATION_MARKER))
    stream.seek(max(0, size - retained))
    return _LOG_TRUNCATION_MARKER + stream.read(retained)


def _reserve_disk_bytes(path: Path, size: int) -> None:
    """Atomically reserve real blocks; sparse truncation is intentionally forbidden."""

    fallocate = getattr(os, "posix_fallocate", None)
    if fallocate is None:
        raise DockerRuntimeError("production controller cannot prove disk reservation support")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        fallocate(descriptor, 0, size)
    except Exception:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    os.close(descriptor)


def _normalized_architecture(value: object) -> str:
    architecture = str(value or "").lower()
    return {"x86_64": "amd64", "aarch64": "arm64"}.get(architecture, architecture)


class DockerCLIClient:
    """Small Docker Engine client implemented through the cross-platform CLI."""

    def __init__(self, executable: str = "docker", *, environment: Mapping[str, str] | None = None):
        if not executable or any(character in executable for character in "\x00\n\r"):
            raise DockerRuntimeError("Docker executable is invalid")
        source = os.environ if environment is None else environment
        self.executable = executable
        self.environment = {key: source[key] for key in _DOCKER_ENVIRONMENT_KEYS if key in source}

    def _run(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float = 30,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        try:
            completed = subprocess.run(
                [self.executable, *arguments],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.environment,
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise DockerRuntimeError(f"Docker command could not complete: {error}") from error
        if check and completed.returncode != 0:
            detail = _decode(completed.stderr)[-2000:].strip()
            raise DockerRuntimeError(f"Docker command failed: {detail or 'no diagnostic'}")
        return completed

    def _json(self, arguments: Sequence[str]) -> Mapping[str, Any]:
        completed = self._run(arguments)
        try:
            value = json.loads(_decode(completed.stdout))
        except json.JSONDecodeError as error:
            raise DockerRuntimeError(f"Docker returned invalid JSON: {error}") from error
        if not isinstance(value, Mapping):
            raise DockerRuntimeError("Docker JSON response was not an object")
        return value

    def daemon_info(self) -> Mapping[str, Any]:
        return self._json(("info", "--format", "{{json .}}"))

    def inspect_image(self, image_reference: str) -> Mapping[str, Any]:
        completed = self._run(("image", "inspect", image_reference))
        try:
            value = json.loads(_decode(completed.stdout))
        except json.JSONDecodeError as error:
            raise DockerRuntimeError(f"Docker image inspection was invalid: {error}") from error
        if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], Mapping):
            raise DockerRuntimeError("Docker image inspection was ambiguous")
        return value[0]

    def inspect_container(self, container_id: str) -> Mapping[str, Any]:
        if not _CONTAINER_ID.fullmatch(container_id):
            raise DockerRuntimeError("Docker container inspection requires a full container ID")
        completed = self._run(("container", "inspect", container_id), check=False)
        if completed.returncode != 0:
            detail = _decode(completed.stderr)[-2000:].strip()
            absent = _NO_SUCH_CONTAINER.fullmatch(detail)
            if absent is not None and absent.group("id") == container_id:
                raise DockerContainerNotFound(container_id)
            raise DockerRuntimeError(
                f"Docker container inspection failed: {detail or 'no diagnostic'}"
            )
        try:
            value = json.loads(_decode(completed.stdout))
        except json.JSONDecodeError as error:
            raise DockerRuntimeError(f"Docker container inspection was invalid: {error}") from error
        if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], Mapping):
            raise DockerRuntimeError("Docker container inspection was ambiguous")
        return value[0]

    def create_container(self, request: DockerCreateRequest) -> str:
        arguments = [
            "container",
            "create",
            "--name",
            request.name,
            "--network",
            "none",
            "--read-only",
            "--user",
            request.user,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges=true",
            "--pids-limit",
            str(request.pids_limit),
            "--memory",
            str(request.memory_bytes),
            "--memory-swap",
            str(request.memory_bytes),
            "--cpu-period",
            "100000",
            "--cpu-quota",
            str(max(1, request.nano_cpus // 10_000)),
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,nodev,size={request.tmpfs_size_bytes}",
            "--ulimit",
            "nofile=256:256",
            "--ulimit",
            "core=0:0",
            "--ulimit",
            f"fsize={request.file_size_limit_bytes}:{request.file_size_limit_bytes}",
            "--log-driver",
            "local",
            "--log-opt",
            f"max-size={request.log_bytes_limit}",
            "--log-opt",
            "max-file=1",
            "--log-opt",
            "compress=true",
            "--workdir",
            request.working_directory,
            "--entrypoint",
            "/usr/bin/tini",
        ]
        for key, value in sorted(request.labels.items()):
            arguments.extend(("--label", f"{key}={value}"))
        for key, value in sorted(request.environment.items()):
            arguments.extend(("--env", f"{key}={value}"))
        for mount in request.mounts:
            option = f"type=bind,src={mount.daemon_source},dst={mount.target}" + (
                ",readonly" if mount.read_only else ""
            )
            arguments.extend(("--mount", option))
        arguments.extend((request.image_reference, "--", *request.command))
        completed = self._run(arguments)
        container_id = _decode(completed.stdout).strip()
        if not _CONTAINER_ID.fullmatch(container_id):
            raise DockerRuntimeError("Docker create did not return a full container ID")
        return container_id

    def start_container(self, container_id: str) -> None:
        self._run(("container", "start", container_id))

    def wait_container(self, container_id: str, timeout_seconds: float) -> int:
        try:
            completed = subprocess.run(
                [self.executable, "container", "wait", container_id],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.environment,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise DockerWaitTimeout("Docker worker exceeded its wall-clock timeout") from error
        except OSError as error:
            raise DockerRuntimeError(f"Docker wait failed: {error}") from error
        if completed.returncode != 0:
            raise DockerRuntimeError(
                "Docker wait failed: " + (_decode(completed.stderr)[-2000:].strip() or "unknown")
            )
        try:
            return int(_decode(completed.stdout).strip())
        except ValueError as error:
            raise DockerRuntimeError("Docker wait returned an invalid exit code") from error

    def container_logs(self, container_id: str, max_bytes: int) -> tuple[bytes, bytes]:
        if not _CONTAINER_ID.fullmatch(container_id):
            raise DockerRuntimeError("Docker logs require a full container ID")
        if not 1024 <= max_bytes <= 64 * 1024 * 1024:
            raise DockerRuntimeError("Docker log read limit is outside the supported range")
        try:
            with (
                tempfile.TemporaryFile() as stdout_stream,
                tempfile.TemporaryFile() as stderr_stream,
            ):
                completed = subprocess.run(
                    [self.executable, "container", "logs", container_id],
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    env=self.environment,
                    timeout=30,
                    check=False,
                )
                stdout = _read_bounded_file(stdout_stream, max_bytes)
                stderr = _read_bounded_file(stderr_stream, max_bytes)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise DockerRuntimeError(f"Docker logs could not be collected: {error}") from error
        if completed.returncode != 0:
            raise DockerRuntimeError(
                "Docker logs failed: " + (_decode(stderr).strip() or "no diagnostic")
            )
        return stdout, stderr

    def stop_container(self, container_id: str, timeout_seconds: int) -> None:
        self._run(("container", "stop", "--time", str(timeout_seconds), container_id))

    def kill_container(self, container_id: str) -> None:
        self._run(("container", "kill", container_id))

    def remove_container(self, container_id: str, *, force: bool) -> None:
        arguments = ["container", "rm"]
        if force:
            arguments.append("--force")
        arguments.append(container_id)
        self._run(arguments)


class DockerExecutionRuntime:
    """Create-verify-lease-start lifecycle for one disposable Docker worker."""

    def __init__(
        self,
        *,
        client: DockerClient,
        image_reference: str,
        controller_id: str,
        doctor_input_path: Path = Path("/source/pyproject.toml"),
        doctor_output_root: Path = Path("/runs/control/docker-doctor"),
        output_monitor_interval_seconds: float = 1.0,
        disk_reserver: Callable[[Path, int], None] = _reserve_disk_bytes,
    ) -> None:
        self.client = client
        self.image_reference = image_reference
        self.worker_image_digest = _pinned_digest(image_reference)
        self.controller_id = controller_id
        self.doctor_input_path = doctor_input_path
        self.doctor_output_root = doctor_output_root
        if not 0.01 <= output_monitor_interval_seconds <= 10:
            raise DockerRuntimeError("Docker output monitor interval is outside the safe range")
        self.output_monitor_interval_seconds = output_monitor_interval_seconds
        self.disk_reserver = disk_reserver
        if not controller_id or any(character in controller_id for character in "\x00\n\r"):
            raise DockerRuntimeError("controller identity is invalid")
        self._mount_resolver: DockerMountResolver | None = None

    @property
    def mount_resolver(self) -> DockerMountResolver:
        if self._mount_resolver is None:
            self._mount_resolver = DockerMountResolver(self.client, self.controller_id)
        return self._mount_resolver

    def _image_identity(self) -> tuple[str, Mapping[str, Any], Mapping[str, str]]:
        inspection = self.client.inspect_image(self.image_reference)
        image_id = inspection.get("Id")
        if not isinstance(image_id, str) or not _DIGEST.fullmatch(image_id):
            raise DockerRuntimeError("Docker image has no immutable local image ID")
        if _DIGEST.fullmatch(self.image_reference):
            if image_id != self.image_reference:
                raise DockerRuntimeError(
                    "local worker image ID differs from the configured image ID"
                )
        else:
            repo_digests = inspection.get("RepoDigests")
            if not isinstance(repo_digests, list) or not any(
                isinstance(value, str) and value.endswith("@" + self.worker_image_digest)
                for value in repo_digests
            ):
                raise DockerRuntimeError(
                    "local worker image does not match the configured manifest digest"
                )
        config = inspection.get("Config")
        labels = config.get("Labels") if isinstance(config, Mapping) else None
        if not isinstance(labels, Mapping):
            raise DockerRuntimeError("worker image has no reproducibility labels")
        metadata: dict[str, str] = {}
        for output_key, image_label in _IMAGE_LABELS.items():
            value = labels.get(image_label)
            if output_key == "source_git_commit":
                if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40,64}", value):
                    raise DockerRuntimeError(
                        f"worker image label {image_label} is not a source commit"
                    )
                metadata[output_key] = value
            else:
                metadata[output_key] = _normalize_digest_label(
                    value,
                    label=image_label,
                    allow_prefixed=output_key == "base_image_digest",
                )
        seccomp_value = labels.get("org.rex.seccomp-sha256")
        metadata["seccomp_sha256"] = (
            _normalize_digest_label(seccomp_value, label="org.rex.seccomp-sha256")
            if seccomp_value is not None
            else ""
        )
        architecture = inspection.get("Architecture")
        operating_system = inspection.get("Os")
        if not isinstance(architecture, str) or operating_system != "linux":
            raise DockerRuntimeError("worker image must declare a Linux platform and architecture")
        declared_architecture = labels.get("org.rex.target-architecture")
        if _normalized_architecture(declared_architecture) != _normalized_architecture(
            architecture
        ):
            raise DockerRuntimeError("worker image target-architecture label drifted")
        metadata["target_architecture"] = _normalized_architecture(declared_architecture)
        metadata["container_platform"] = f"linux/{_normalized_architecture(architecture)}"
        return image_id, inspection, metadata

    def environment_identity(self) -> dict[str, str]:
        """Return stable components used by provenance and result cache keys."""

        info = self.client.daemon_info()
        daemon_identity = _daemon_identity(info)
        image_id, _inspection, metadata = self._image_identity()
        engine_platform = f"linux/{_normalized_architecture(info.get('Architecture'))}"
        if metadata["container_platform"] != engine_platform:
            raise DockerRuntimeError(
                "worker image architecture differs from the Docker Engine architecture"
            )
        if not metadata["seccomp_sha256"]:
            metadata["seccomp_sha256"] = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "profile": "docker-default",
                        "server_version": info.get("ServerVersion"),
                        "security_options": info.get("SecurityOptions", []),
                    }
                )
            ).hexdigest()
        return {
            "runtime_kind": RuntimeKind.DOCKER.value,
            "worker_image_digest": self.worker_image_digest,
            "worker_image_id": image_id,
            "docker_daemon_identity": daemon_identity,
            "docker_engine_platform": engine_platform,
            "docker_security_policy_sha256": docker_security_policy_sha256(),
            **metadata,
        }

    def doctor(self) -> DoctorResult:
        checks: list[DoctorCheck] = []
        identity: dict[str, str] = {}
        try:
            identity = self.environment_identity()
            checks.extend(
                (
                    DoctorCheck("docker_daemon", True, identity["docker_daemon_identity"]),
                    DoctorCheck("worker_image_digest", True, self.worker_image_digest),
                    DoctorCheck("worker_image_metadata", True, identity["source_git_commit"]),
                )
            )
        except Exception as error:
            checks.append(DoctorCheck("docker_daemon_and_image", False, str(error)))
        try:
            roots = self.mount_resolver.roots
            checks.append(
                DoctorCheck(
                    "controller_mount_roots",
                    len(roots) == 3,
                    ", ".join(str(root.controller_path) for root in roots),
                )
            )
        except Exception as error:
            checks.append(DoctorCheck("controller_mount_roots", False, str(error)))
        if checks and all(check.passed for check in checks):
            try:
                from rex.execution.docker_doctor import DockerSecurityDoctor

                active = DockerSecurityDoctor(
                    self,
                    input_path=self.doctor_input_path,
                    output_root=self.doctor_output_root,
                ).run()
                checks.extend(active.checks)
            except Exception as error:
                checks.append(DoctorCheck("active_worker_security_probe", False, str(error)))
        safe = bool(checks) and all(check.passed for check in checks)
        return DoctorResult(
            runtime_kind=RuntimeKind.DOCKER,
            available=any(check.name == "docker_daemon" and check.passed for check in checks),
            safe_for_production=safe,
            checks=tuple(checks),
            detail="Docker runtime ready" if safe else "Docker runtime failed closed",
            environment_identity=identity,
        )

    def _container_name(self, specification: ExecutionSpec) -> str:
        raw_parts = (
            specification.run_id[:24],
            specification.experiment_id[:24],
            specification.attempt_id[:16],
            specification.execution_sha256[:12],
        )
        parts = tuple(
            re.sub(r"[^a-z0-9_.-]+", "-", value.lower()).strip("-.") or "unknown"
            for value in raw_parts
        )
        return "rex-" + "-".join(parts)

    def _create_request(
        self,
        specification: ExecutionSpec,
        mounts: tuple[ResolvedDockerMount, ...],
    ) -> DockerCreateRequest:
        working = specification.working_directory.rstrip("/") + "/"
        if not any(
            working.startswith(mount.target.rstrip("/") + "/")
            or specification.working_directory == mount.target
            for mount in mounts
        ):
            raise DockerRuntimeError("worker working directory is outside its approved mounts")
        resource_labels = {
            "rex.output_bytes_limit": str(specification.output_bytes_limit),
            "rex.file_size_limit_bytes": str(specification.file_size_limit_bytes),
            "rex.log_bytes_limit": str(specification.log_bytes_limit),
        }
        return DockerCreateRequest(
            name=self._container_name(specification),
            image_reference=self.image_reference,
            command=specification.command,
            working_directory=specification.working_directory,
            user=specification.user,
            environment=dict(specification.environment),
            labels={**specification.identity_labels, **resource_labels},
            mounts=mounts,
            memory_bytes=specification.memory_bytes,
            nano_cpus=specification.nano_cpus,
            pids_limit=specification.pids_limit,
            tmpfs_size_bytes=specification.tmpfs_size_bytes,
            output_bytes_limit=specification.output_bytes_limit,
            file_size_limit_bytes=specification.file_size_limit_bytes,
            log_bytes_limit=specification.log_bytes_limit,
            minimum_free_space_bytes=specification.minimum_free_space_bytes,
        )

    def _verify_inspection(
        self,
        inspection: Mapping[str, Any],
        request: DockerCreateRequest,
        *,
        container_id: str,
        image_id: str,
    ) -> None:
        if inspection.get("Id") != container_id or inspection.get("Image") != image_id:
            raise DockerRuntimeError("created container identity or image ID drifted")
        if str(inspection.get("Name", "")).lstrip("/") != request.name:
            raise DockerRuntimeError("created container name drifted")
        config = inspection.get("Config")
        host = inspection.get("HostConfig")
        if not isinstance(config, Mapping) or not isinstance(host, Mapping):
            raise DockerRuntimeError("created container inspection is incomplete")
        observed_labels = config.get("Labels")
        if not isinstance(observed_labels, Mapping):
            raise DockerRuntimeError("created container labels are missing")
        for key, value in request.labels.items():
            if observed_labels.get(key) != value:
                raise DockerRuntimeError(f"created container label {key} drifted")
        unexpected_rex = sorted(
            key
            for key in observed_labels
            if str(key).startswith("rex.") and key not in request.labels
        )
        if unexpected_rex:
            raise DockerRuntimeError(
                "created container has unexpected REX labels: " + ", ".join(unexpected_rex)
            )
        if config.get("User") != request.user or request.user.startswith("0:"):
            raise DockerRuntimeError("created container is not the expected non-root user")
        if config.get("WorkingDir") != request.working_directory:
            raise DockerRuntimeError("created container working directory drifted")
        if tuple(config.get("Cmd") or ()) != ("--", *request.command):
            raise DockerRuntimeError("created container command drifted")
        if tuple(config.get("Entrypoint") or ()) != ("/usr/bin/tini",):
            raise DockerRuntimeError("created container did not use the explicit worker entrypoint")
        observed_environment = config.get("Env") or []
        if not isinstance(observed_environment, list):
            raise DockerRuntimeError("created container environment is malformed")
        parsed_environment: dict[str, str] = {}
        for entry in observed_environment:
            if not isinstance(entry, str) or "=" not in entry:
                raise DockerRuntimeError("created container environment is malformed")
            key, value = entry.split("=", 1)
            if key in parsed_environment:
                raise DockerRuntimeError(f"created container environment duplicates {key}")
            if _SECRET_ENV.search(key):
                raise DockerRuntimeError(f"created container leaked forbidden secret {key}")
            parsed_environment[key] = value
        for key, value in request.environment.items():
            if parsed_environment.get(key) != value:
                raise DockerRuntimeError(f"created container environment {key} drifted")
        if host.get("NetworkMode") != "none":
            raise DockerRuntimeError("created container network is not disabled")
        if not bool(host.get("ReadonlyRootfs")):
            raise DockerRuntimeError("created container root filesystem is writable")
        if bool(host.get("Privileged")):
            raise DockerRuntimeError("created container is privileged")
        cap_drop = {str(value).upper() for value in (host.get("CapDrop") or [])}
        if "ALL" not in cap_drop:
            raise DockerRuntimeError("created container did not drop all Linux capabilities")
        security_options = {str(value) for value in (host.get("SecurityOpt") or [])}
        if not any(value.startswith("no-new-privileges") for value in security_options):
            raise DockerRuntimeError("created container permits privilege escalation")
        if int(host.get("PidsLimit") or 0) != request.pids_limit:
            raise DockerRuntimeError("created container process limit drifted")
        if int(host.get("Memory") or 0) != request.memory_bytes:
            raise DockerRuntimeError("created container memory limit drifted")
        if int(host.get("MemorySwap") or 0) != request.memory_bytes:
            raise DockerRuntimeError("created container swap limit drifted")
        ulimits = host.get("Ulimits")
        if not isinstance(ulimits, list):
            raise DockerRuntimeError("created container ulimits are missing")
        observed_ulimits = {
            str(item.get("Name")): (int(item.get("Soft", -1)), int(item.get("Hard", -1)))
            for item in ulimits
            if isinstance(item, Mapping)
        }
        if (
            observed_ulimits.get("nofile") != (256, 256)
            or observed_ulimits.get("core") != (0, 0)
            or observed_ulimits.get("fsize")
            != (request.file_size_limit_bytes, request.file_size_limit_bytes)
        ):
            raise DockerRuntimeError("created container file/core/size ulimits drifted")
        log_config = host.get("LogConfig")
        if not isinstance(log_config, Mapping) or log_config.get("Type") != "local":
            raise DockerRuntimeError("created container log driver is not bounded local storage")
        log_options = log_config.get("Config")
        if not isinstance(log_options, Mapping) or dict(log_options) != {
            "compress": "true",
            "max-file": "1",
            "max-size": str(request.log_bytes_limit),
        }:
            raise DockerRuntimeError("created container log-driver bounds drifted")
        expected_quota = max(1, request.nano_cpus // 10_000)
        if (
            int(host.get("CpuPeriod") or 0) != 100000
            or int(host.get("CpuQuota") or 0) != expected_quota
        ):
            raise DockerRuntimeError("created container CPU quota drifted")
        tmpfs = host.get("Tmpfs")
        if not isinstance(tmpfs, Mapping) or set(tmpfs) != {"/tmp"}:
            raise DockerRuntimeError("created container tmpfs policy drifted")
        tmp_options = {value for value in str(tmpfs["/tmp"]).split(",") if value}
        required_options = {
            "rw",
            "noexec",
            "nosuid",
            "nodev",
            f"size={request.tmpfs_size_bytes}",
        }
        if not required_options.issubset(tmp_options):
            raise DockerRuntimeError(
                "created container tmpfs is not noexec/nosuid/nodev and bounded"
            )
        inspected_mounts = inspection.get("Mounts")
        if not isinstance(inspected_mounts, list):
            raise DockerRuntimeError("created container mount inventory is missing")
        actual: set[tuple[str, str, bool]] = set()
        for mount in inspected_mounts:
            if not isinstance(mount, Mapping):
                raise DockerRuntimeError("created container mount inventory is malformed")
            if mount.get("Type") == "tmpfs" and mount.get("Destination") == "/tmp":
                continue
            source = str(mount.get("Source", ""))
            destination = str(mount.get("Destination", ""))
            if source.replace("\\", "/").lower().endswith("/docker.sock"):
                raise DockerRuntimeError("created container received the Docker socket")
            actual.add((source, destination, not bool(mount.get("RW"))))
        expected = {
            (mount.daemon_source, mount.target, mount.read_only) for mount in request.mounts
        }
        if actual != expected:
            raise DockerRuntimeError("created container mounts differ from the approved exact set")

    def _verify_handle_identity(
        self, inspection: Mapping[str, Any], handle: RuntimeHandle | ExecutionLease
    ) -> None:
        if (
            inspection.get("Id") != handle.container_id
            or inspection.get("Image") != handle.worker_image_id
        ):
            raise DockerRuntimeError("container ID or image changed; refusing lifecycle operation")
        if str(inspection.get("Name", "")).lstrip("/") != handle.container_name:
            raise DockerRuntimeError("container name changed; refusing lifecycle operation")
        config = inspection.get("Config")
        labels = config.get("Labels") if isinstance(config, Mapping) else None
        expected = {
            "rex.managed": "true",
            "rex.run_id": handle.run_id,
            "rex.experiment_id": handle.experiment_id,
            "rex.attempt_id": handle.attempt_id,
            "rex.request_sha256": handle.request_sha256,
            "rex.execution_sha256": handle.execution_sha256,
        }
        if not isinstance(labels, Mapping) or any(
            labels.get(key) != value for key, value in expected.items()
        ):
            raise DockerRuntimeError("container labels changed; refusing lifecycle operation")

    @staticmethod
    def _reservation_path(lease_path: Path) -> Path:
        return lease_path.with_name(lease_path.name + ".output-reservation")

    @staticmethod
    def _output_usage(directory: Path, limit: int) -> int:
        """Measure aggregate regular-file bytes without following worker-created links."""

        total = 0
        entries_seen = 0
        pending = [directory]
        while pending:
            current = pending.pop()
            try:
                entries = os.scandir(current)
            except OSError as error:
                raise DockerOutputLimitExceeded(
                    f"could not inspect Docker output budget: {error}"
                ) from error
            with entries:
                for entry in entries:
                    entries_seen += 1
                    if entries_seen > 1_000_000:
                        raise DockerOutputLimitExceeded(
                            "Docker output exceeded the safe filesystem-entry bound"
                        )
                    try:
                        if entry.is_symlink():
                            raise DockerOutputLimitExceeded(
                                f"Docker output contains a forbidden symlink: {entry.name}"
                            )
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            raise DockerOutputLimitExceeded(
                                f"Docker output contains a forbidden special file: {entry.name}"
                            )
                        total += entry.stat(follow_symlinks=False).st_size
                    except OSError as error:
                        raise DockerOutputLimitExceeded(
                            f"could not inspect Docker output entry: {error}"
                        ) from error
                    if total > limit:
                        raise DockerOutputLimitExceeded(
                            f"Docker output exceeded its aggregate {limit}-byte budget"
                        )
        return total

    def _controller_output_directory(self, inspection: Mapping[str, Any]) -> tuple[Path, int, int]:
        config = inspection.get("Config")
        labels = config.get("Labels") if isinstance(config, Mapping) else None
        if not isinstance(labels, Mapping):
            raise DockerRuntimeError("container resource labels are missing")
        try:
            output_limit = int(str(labels["rex.output_bytes_limit"]))
            log_limit = int(str(labels["rex.log_bytes_limit"]))
            file_limit = int(str(labels["rex.file_size_limit_bytes"]))
        except (KeyError, TypeError, ValueError) as error:
            raise DockerRuntimeError("container resource labels are malformed") from error
        if not 1 <= output_limit <= 512 * 1024 * 1024:
            raise DockerRuntimeError("container output budget label is outside the safe bound")
        if not 1 <= file_limit <= output_limit:
            raise DockerRuntimeError("container per-file limit label is outside the safe bound")
        if not 1024 <= log_limit <= 64 * 1024 * 1024:
            raise DockerRuntimeError("container log budget label is outside the safe bound")

        mounts = inspection.get("Mounts")
        if not isinstance(mounts, list):
            raise DockerRuntimeError("container mount inventory is missing")
        writable = [
            mount
            for mount in mounts
            if isinstance(mount, Mapping) and mount.get("Type") == "bind" and bool(mount.get("RW"))
        ]
        if len(writable) != 1 or writable[0].get("Destination") != "/output":
            raise DockerRuntimeError("container no longer has exactly one writable bind at /output")
        daemon_source = str(writable[0].get("Source") or "")
        normalized_source = PurePosixPath(daemon_source.replace("\\", "/"))
        matches: list[tuple[int, Any, PurePosixPath]] = []
        for root in self.mount_resolver.roots:
            normalized_root = PurePosixPath(root.daemon_source.replace("\\", "/"))
            if normalized_source == normalized_root or normalized_root in normalized_source.parents:
                matches.append((len(normalized_root.parts), root, normalized_root))
        if not matches:
            raise DockerRuntimeError("container output mount is outside controller-approved roots")
        _length, root, normalized_root = max(matches, key=lambda value: value[0])
        if not root.writable:
            raise DockerRuntimeError("container output mount resolves beneath a read-only root")
        relative = normalized_source.relative_to(normalized_root)
        controller_output = Path(str(root.controller_path)).joinpath(*relative.parts)
        resolved = self.mount_resolver.resolve(
            RuntimeMount(controller_output, "/output", read_only=False)
        )
        if resolved.daemon_source.replace("\\", "/") != daemon_source.replace("\\", "/"):
            raise DockerRuntimeError("container output mount reverse translation drifted")
        return controller_output, output_limit, log_limit

    def _reserve_output_budget(
        self,
        specification: ExecutionSpec,
        output_directory: Path,
    ) -> Path:
        usage = self._output_usage(output_directory, specification.output_bytes_limit)
        remaining = specification.output_bytes_limit - usage
        lease_parent = Path(specification.lease_path).parent
        if output_directory.stat().st_dev != lease_parent.stat().st_dev:
            raise DockerRuntimeError(
                "Docker output reservation must share a filesystem with /output"
            )
        free = shutil.disk_usage(output_directory).free
        required = remaining + specification.minimum_free_space_bytes
        if free < required:
            raise DockerRuntimeError(
                "insufficient free space for Docker output budget and safety reserve "
                f"(required={required}, available={free})"
            )
        reservation = self._reservation_path(Path(specification.lease_path))
        if reservation.exists() or reservation.is_symlink():
            raise DockerRuntimeError("Docker output reservation already exists")
        try:
            self.disk_reserver(reservation, remaining)
        except Exception as error:
            raise DockerRuntimeError(f"could not reserve Docker output budget: {error}") from error
        if not reservation.is_file() or reservation.is_symlink():
            reservation.unlink(missing_ok=True)
            raise DockerRuntimeError("Docker output reservation was not created safely")
        return reservation

    def _monitor_output(self, handle: RuntimeHandle, output: Path, limit: int) -> int:
        usage = self._output_usage(output, limit)
        reservation = self._reservation_path(handle.lease_path)
        if reservation.is_file() and not reservation.is_symlink():
            remaining = max(0, limit - usage)
            if reservation.stat().st_size > remaining:
                with reservation.open("r+b") as stream:
                    stream.truncate(remaining)
        return usage

    def launch(self, specification: ExecutionSpec) -> RuntimeHandle:
        assert_no_controller_secret_leakage(specification)
        lease_path = Path(specification.lease_path)
        reservation_path = self._reservation_path(lease_path)
        if lease_path.is_symlink():
            raise DockerRuntimeError("Docker lease path may not be a symlink")
        if lease_path.is_file():
            prior_lease = read_docker_lease(lease_path)
            if prior_lease.state in {"created", "active"}:
                raise DockerLeaseError("active Docker lease must be recovered before relaunch")
            if prior_lease.request_sha256 != specification.request_sha256:
                raise DockerLeaseError("closed Docker lease belongs to a different request")
            if prior_lease.execution_sha256 != specification.execution_sha256:
                raise DockerLeaseError("closed Docker lease belongs to a different execution")
            if reservation_path.is_symlink():
                raise DockerRuntimeError("Docker output reservation may not be a symlink")
            reservation_path.unlink(missing_ok=True)
        self.mount_resolver.resolve(
            RuntimeMount(
                source=lease_path.parent,
                target="/lease-validation",
                read_only=False,
            )
        )
        mounts = self.mount_resolver.resolve_all(specification.mounts)
        normalized_lease = lease_path.resolve()
        for mount in mounts:
            normalized_mount = mount.controller_source.resolve()
            try:
                normalized_lease.relative_to(normalized_mount)
            except ValueError:
                continue
            raise DockerRuntimeError("Docker lease must remain outside every worker mount")
        request = self._create_request(specification, mounts)
        output_mount = next(mount for mount in mounts if not mount.read_only)
        reservation_path = self._reserve_output_budget(
            specification,
            output_mount.controller_source,
        )
        container_id: str | None = None
        lease: ExecutionLease | None = None
        try:
            info = self.client.daemon_info()
            daemon_identity = _daemon_identity(info)
            image_id, _inspection, _metadata = self._image_identity()
            container_id = self.client.create_container(request)
            if not _CONTAINER_ID.fullmatch(container_id):
                raise DockerRuntimeError("Docker daemon returned a non-canonical container ID")
            inspection = self.client.inspect_container(container_id)
            self._verify_inspection(
                inspection,
                request,
                container_id=container_id,
                image_id=image_id,
            )
            started_at = int(time.time() * 1000)
            handle = RuntimeHandle(
                runtime_kind=RuntimeKind.DOCKER,
                container_id=container_id,
                container_name=request.name,
                worker_image_digest=self.worker_image_digest,
                worker_image_id=image_id,
                daemon_identity=daemon_identity,
                run_id=specification.run_id,
                experiment_id=specification.experiment_id,
                attempt_id=specification.attempt_id,
                request_sha256=specification.request_sha256,
                execution_sha256=specification.execution_sha256,
                lease_path=Path(specification.lease_path).resolve(),
                timeout_seconds=specification.timeout_seconds,
                started_at_epoch_ms=started_at,
            )
            lease = create_docker_lease(specification, handle)
            persist_docker_lease(handle.lease_path, lease)
            self.client.start_container(container_id)
            mark_docker_lease_started(handle.lease_path, lease)
            return handle
        except Exception as error:
            reservation_path.unlink(missing_ok=True)
            if lease is not None:
                try:
                    close_docker_lease(
                        specification.lease_path,
                        lease,
                        reason="launch-failed",
                        exit_code=None,
                    )
                except Exception:
                    pass
            if container_id is not None:
                try:
                    self.client.remove_container(container_id, force=True)
                except Exception:
                    pass
            if isinstance(error, ExecutionRuntimeError):
                raise
            raise DockerRuntimeError(f"Docker worker launch failed: {error}") from error

    def inspect(self, handle: RuntimeHandle) -> RuntimeStatus:
        if handle.runtime_kind != RuntimeKind.DOCKER:
            raise DockerRuntimeError("runtime handle belongs to a different backend")
        if handle.worker_image_digest != self.worker_image_digest:
            raise DockerRuntimeError("runtime handle belongs to a different worker image digest")
        if _daemon_identity(self.client.daemon_info()) != handle.daemon_identity:
            raise DockerRuntimeError("Docker daemon identity changed")
        try:
            inspection = self.client.inspect_container(handle.container_id)
        except DockerContainerNotFound as error:
            if error.container_id != handle.container_id:
                raise DockerRuntimeError(
                    "Docker daemon reported absence for a different container ID"
                ) from error
            return RuntimeStatus(
                state=RuntimeLifecycleState.REMOVED,
                detail="Docker daemon verified the exact full container ID is absent",
            )
        self._verify_handle_identity(inspection, handle)
        state = inspection.get("State")
        if not isinstance(state, Mapping):
            raise DockerRuntimeError("container state inspection is missing")
        if bool(state.get("Running")):
            lifecycle = RuntimeLifecycleState.RUNNING
        elif state.get("Status") == "created":
            lifecycle = RuntimeLifecycleState.CREATED
        elif state.get("Status") in {"exited", "dead"}:
            lifecycle = RuntimeLifecycleState.EXITED
        else:
            lifecycle = RuntimeLifecycleState.UNKNOWN
        exit_code_value = state.get("ExitCode")
        return RuntimeStatus(
            state=lifecycle,
            exit_code=int(exit_code_value) if exit_code_value is not None else None,
            oom_killed=bool(state.get("OOMKilled")),
            detail=str(state.get("Error") or ""),
        )

    def terminate(self, handle: RuntimeHandle) -> None:
        status = self.inspect(handle)
        if status.state != RuntimeLifecycleState.RUNNING:
            return
        try:
            self.client.stop_container(handle.container_id, 2)
        except Exception:
            status = self.inspect(handle)
            if status.state == RuntimeLifecycleState.RUNNING:
                self.client.kill_container(handle.container_id)
        final = self.inspect(handle)
        if final.state == RuntimeLifecycleState.RUNNING:
            raise DockerRuntimeError("Docker worker survived stop and kill")

    def _wait_with_output_monitor(
        self,
        handle: RuntimeHandle,
        output_directory: Path,
        output_limit: int,
    ) -> tuple[int, bool]:
        deadline = time.monotonic() + handle.timeout_seconds
        while True:
            self._monitor_output(handle, output_directory, output_limit)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.terminate(handle)
                final = self.inspect(handle)
                if final.state != RuntimeLifecycleState.EXITED or final.exit_code is None:
                    raise DockerRuntimeError(
                        "timed-out Docker worker did not reach a verifiable exited state"
                    )
                return final.exit_code, True
            try:
                exit_code = self.client.wait_container(
                    handle.container_id,
                    min(self.output_monitor_interval_seconds, remaining),
                )
            except DockerWaitTimeout:
                status = self.inspect(handle)
                if status.state == RuntimeLifecycleState.EXITED and status.exit_code is not None:
                    return status.exit_code, False
                if status.state != RuntimeLifecycleState.RUNNING:
                    raise DockerRuntimeError(
                        "Docker wait timed out but the worker is not verifiably running"
                    )
                continue
            self._monitor_output(handle, output_directory, output_limit)
            return exit_code, False

    def collect(self, handle: RuntimeHandle) -> ExecutionResult:
        lease = read_docker_lease(handle.lease_path)
        verify_lease_handle(lease, handle)
        started = handle.started_at_epoch_ms / 1000
        try:
            inspection = self.client.inspect_container(handle.container_id)
            self._verify_handle_identity(inspection, handle)
            output_directory, output_limit, log_limit = self._controller_output_directory(
                inspection
            )
            exit_code, timed_out = self._wait_with_output_monitor(
                handle,
                output_directory,
                output_limit,
            )
        except DockerOutputLimitExceeded:
            status = self.inspect(handle)
            if status.state == RuntimeLifecycleState.RUNNING:
                self.terminate(handle)
                status = self.inspect(handle)
            if status.state != RuntimeLifecycleState.EXITED:
                raise DockerRuntimeError(
                    "output-limit worker could not be stopped in a verifiable state"
                )
            if lease.state in {"created", "active"}:
                close_docker_lease(
                    handle.lease_path,
                    lease,
                    reason="output-limit-exceeded",
                    exit_code=status.exit_code,
                )
            self._reservation_path(handle.lease_path).unlink(missing_ok=True)
            raise
        status = self.inspect(handle)
        if status.exit_code is not None and status.exit_code != exit_code:
            raise DockerRuntimeError("Docker wait exit code differs from inspected container state")
        stdout_bytes, stderr_bytes = self.client.container_logs(handle.container_id, log_limit)
        stdout_bytes = _bounded_bytes(stdout_bytes, log_limit)
        stderr_bytes = _bounded_bytes(stderr_bytes, log_limit)
        if status.oom_killed:
            outcome = ExecutionOutcome.OOM
        elif timed_out:
            outcome = ExecutionOutcome.TIMEOUT
        elif exit_code == 0:
            outcome = ExecutionOutcome.SUCCESS
        elif exit_code in {137, 143}:
            outcome = ExecutionOutcome.INTERRUPTED
        else:
            outcome = ExecutionOutcome.FAILED
        if lease.state in {"created", "active"}:
            close_docker_lease(
                handle.lease_path,
                lease,
                reason=outcome.value,
                exit_code=exit_code,
            )
        self._reservation_path(handle.lease_path).unlink(missing_ok=True)
        return ExecutionResult(
            outcome=outcome,
            exit_code=exit_code,
            stdout=_decode(stdout_bytes),
            stderr=_decode(stderr_bytes),
            wall_seconds=max(0.0, time.time() - started),
            oom_killed=status.oom_killed,
            timed_out=timed_out,
        )

    def recover(self, lease: ExecutionLease) -> RecoveryResult:
        if lease.runtime_kind != RuntimeKind.DOCKER:
            raise DockerLeaseError("active native lease cannot resume through Docker")
        if lease.worker_image_digest != self.worker_image_digest:
            raise DockerLeaseError("Docker lease belongs to a different worker image digest")
        if lease.state not in {"created", "active"}:
            return RecoveryResult(
                outcome="already-closed",
                container_id=lease.container_id,
                terminated=False,
            )
        observed_daemon = _daemon_identity(self.client.daemon_info())
        if observed_daemon != lease.daemon_identity:
            raise DockerLeaseError("Docker lease belongs to a different daemon identity")
        inspection = self.client.inspect_container(lease.container_id)
        self._verify_handle_identity(inspection, lease)
        state = inspection.get("State")
        running = isinstance(state, Mapping) and bool(state.get("Running"))
        if running:
            handle = runtime_handle_from_docker_lease(lease, timeout_seconds=1)
            self.terminate(handle)
        exit_value = state.get("ExitCode") if isinstance(state, Mapping) else None
        if lease.lease_path is not None:
            close_docker_lease(
                lease.lease_path,
                lease,
                reason="orphan-container-terminated" if running else "stale-lease-container-exited",
                exit_code=int(exit_value) if exit_value is not None else None,
                recovered=True,
            )
        return RecoveryResult(
            outcome="orphan-container-terminated" if running else "stale-lease-container-exited",
            container_id=lease.container_id,
            terminated=running,
        )

    def cleanup(self, handle: RuntimeHandle) -> None:
        status = self.inspect(handle)
        if status.state == RuntimeLifecycleState.RUNNING:
            raise DockerRuntimeError("refusing to remove a running Docker worker")
        lease = read_docker_lease(handle.lease_path)
        verify_lease_handle(lease, handle)
        if lease.state not in {"closed", "recovered"}:
            raise DockerRuntimeError("refusing to remove Docker worker before evidence is closed")
        self.client.remove_container(handle.container_id, force=False)
        self._reservation_path(handle.lease_path).unlink(missing_ok=True)
