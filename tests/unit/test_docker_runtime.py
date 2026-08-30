from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from rex.execution.docker_lease import (
    DockerLeaseError,
    read_docker_lease,
    runtime_handle_from_docker_lease,
)
from rex.execution.docker_mounts import DockerMountError, DockerMountResolver
from rex.execution.runtime import (
    ExecutionOutcome,
    ExecutionRuntimeError,
    ExecutionSpec,
    RuntimeLifecycleState,
    RuntimeMount,
    production_runtime,
)
from rex.execution.runtime_docker import (
    DockerCLIClient,
    DockerContainerNotFound,
    DockerCreateRequest,
    DockerExecutionRuntime,
    DockerOutputLimitExceeded,
    DockerRuntimeError,
    DockerWaitTimeout,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
IMAGE_ID = "sha256:" + "c" * 64
CONTAINER_ID = "d" * 64


class FakeDockerClient:
    def __init__(self, source: Path, data: Path, runs: Path) -> None:
        self.source = source
        self.data = data
        self.runs = runs
        self.requests: dict[str, DockerCreateRequest] = {}
        self.inspections: dict[str, dict[str, Any]] = {}
        self.removed: list[str] = []
        self.wait_timeout_once = False
        self.wait_always_timeout = False
        self.oom = False
        self.probe_failure = False
        self.output_bytes_on_wait = 0
        self.stdout = b"worker output\n"
        self.stderr = b""
        self.absent: set[str] = set()
        self.inspect_failure: Exception | None = None

    def daemon_info(self) -> Mapping[str, Any]:
        return {
            "ID": "daemon-test",
            "Name": "fake-engine",
            "ServerVersion": "28.0.0",
            "OperatingSystem": "Linux",
            "Architecture": "arm64",
            "OSType": "linux",
            "SecurityOptions": ["name=seccomp,profile=builtin"],
        }

    def inspect_image(self, image_reference: str) -> Mapping[str, Any]:
        assert image_reference == IMAGE_ID
        return {
            "Id": IMAGE_ID,
            "RepoDigests": [],
            "Architecture": "arm64",
            "Os": "linux",
            "Config": {
                "Env": ["PATH=/usr/local/bin:/usr/bin:/bin", "GPG_KEY=public-material"],
                "Labels": {
                    "org.opencontainers.image.revision": "1" * 40,
                    "org.rex.dependency-lock-sha256": "2" * 64,
                    "org.rex.pyproject-sha256": "3" * 64,
                    "org.rex.starter-kit-sha256": "4" * 64,
                    "org.rex.base-image-digest": "sha256:" + "5" * 64,
                    "org.rex.target-architecture": "arm64",
                },
            },
        }

    def inspect_container(self, container_id: str) -> Mapping[str, Any]:
        if container_id == "controller":
            return {
                "Mounts": [
                    {
                        "Type": "bind",
                        "Source": "/daemon/source",
                        "Destination": str(self.source),
                        "RW": False,
                    },
                    {
                        "Type": "bind",
                        "Source": "/daemon/data",
                        "Destination": str(self.data),
                        "RW": False,
                    },
                    {
                        "Type": "bind",
                        "Source": "/daemon/runs",
                        "Destination": str(self.runs),
                        "RW": True,
                    },
                ]
            }
        if self.inspect_failure is not None:
            raise self.inspect_failure
        if container_id in self.absent:
            raise DockerContainerNotFound(container_id)
        if container_id not in self.inspections:
            raise DockerRuntimeError("container inspection failed unexpectedly")
        return self.inspections[container_id]

    def create_container(self, request: DockerCreateRequest) -> str:
        identifier = f"{len(self.requests) + 13:064x}"
        self.requests[identifier] = request
        image_labels = dict(self.inspect_image(IMAGE_ID)["Config"]["Labels"])
        self.inspections[identifier] = {
            "Id": identifier,
            "Image": IMAGE_ID,
            "Name": "/" + request.name,
            "Config": {
                "User": request.user,
                "WorkingDir": request.working_directory,
                "Cmd": ["--", *request.command],
                "Entrypoint": ["/usr/bin/tini"],
                "Env": [
                    "PATH=/usr/local/bin:/usr/bin:/bin",
                    "GPG_KEY=public-material",
                    *[f"{key}={value}" for key, value in sorted(request.environment.items())],
                ],
                "Labels": {**image_labels, **request.labels},
            },
            "HostConfig": {
                "NetworkMode": "none",
                "ReadonlyRootfs": True,
                "Privileged": False,
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges:true"],
                "PidsLimit": request.pids_limit,
                "Memory": request.memory_bytes,
                "MemorySwap": request.memory_bytes,
                "Ulimits": [
                    {"Name": "nofile", "Soft": 256, "Hard": 256},
                    {"Name": "core", "Soft": 0, "Hard": 0},
                    {
                        "Name": "fsize",
                        "Soft": request.file_size_limit_bytes,
                        "Hard": request.file_size_limit_bytes,
                    },
                ],
                "LogConfig": {
                    "Type": "local",
                    "Config": {
                        "compress": "true",
                        "max-file": "1",
                        "max-size": str(request.log_bytes_limit),
                    },
                },
                "CpuPeriod": 100000,
                "CpuQuota": max(1, request.nano_cpus // 10_000),
                "Tmpfs": {"/tmp": (f"rw,noexec,nosuid,nodev,size={request.tmpfs_size_bytes}")},
            },
            "Mounts": [
                {
                    "Type": "bind",
                    "Source": mount.daemon_source,
                    "Destination": mount.target,
                    "RW": not mount.read_only,
                }
                for mount in request.mounts
            ],
            "State": {"Status": "created", "Running": False, "ExitCode": 0},
        }
        return identifier

    def start_container(self, container_id: str) -> None:
        self.inspections[container_id]["State"] = {
            "Status": "running",
            "Running": True,
            "ExitCode": 0,
        }

    def wait_container(self, container_id: str, timeout_seconds: float) -> int:
        if self.wait_always_timeout and self.inspections[container_id]["State"].get("Running"):
            raise DockerWaitTimeout("controlled timeout")
        if self.wait_timeout_once:
            self.wait_timeout_once = False
            raise DockerWaitTimeout("controlled timeout")
        self.inspections[container_id]["State"] = {
            "Status": "exited",
            "Running": False,
            "ExitCode": 137 if self.oom else 0,
            "OOMKilled": self.oom,
        }
        request = self.requests[container_id]
        if self.output_bytes_on_wait:
            writable = next(mount for mount in request.mounts if not mount.read_only)
            (writable.controller_source / "oversized.bin").write_bytes(
                b"x" * self.output_bytes_on_wait
            )
        if request.labels["rex.experiment_id"] == "security-probe":
            writable = next(mount for mount in request.mounts if not mount.read_only)
            (writable.controller_source / "worker-write-ok.json").write_text(
                '{"ok":true}\n', encoding="utf-8"
            )
        return 137 if self.oom else 0

    def container_logs(self, container_id: str, max_bytes: int) -> tuple[bytes, bytes]:
        request = self.requests[container_id]
        if request.labels["rex.experiment_id"] == "security-probe":
            checks = {
                "approved_input_read": True,
                "protected_input_read_only": True,
                "exact_output_writable": True,
                "root_read_only": True,
                "worker_non_root": True,
                "privilege_escalation_disabled": True,
                "capabilities_dropped": True,
                "docker_socket_absent": True,
                "unapproved_host_mount_absent": True,
                "credentials_absent": True,
                "dns_denied": not self.probe_failure,
                "outbound_tcp_denied": True,
                "tmp_noexec": True,
                "memory_limit_visible": True,
                "process_limit_visible": True,
                "cpu_limit_visible": True,
            }
            return (json.dumps(checks).encode() + b"\n", b"")
        return self.stdout, self.stderr

    def stop_container(self, container_id: str, timeout_seconds: int) -> None:
        self.inspections[container_id]["State"] = {
            "Status": "exited",
            "Running": False,
            "ExitCode": 143,
        }

    def kill_container(self, container_id: str) -> None:
        self.inspections[container_id]["State"] = {
            "Status": "exited",
            "Running": False,
            "ExitCode": 137,
        }

    def remove_container(self, container_id: str, *, force: bool) -> None:
        self.removed.append(container_id)
        self.inspections.pop(container_id, None)


def _runtime(tmp_path: Path) -> tuple[DockerExecutionRuntime, FakeDockerClient, Path, Path, Path]:
    source = tmp_path / "source"
    data = tmp_path / "data"
    runs = tmp_path / "runs"
    source.mkdir()
    data.mkdir()
    runs.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    client = FakeDockerClient(source, data, runs)
    runtime = DockerExecutionRuntime(
        client=client,
        image_reference=IMAGE_ID,
        controller_id="controller",
        doctor_input_path=source / "pyproject.toml",
        doctor_output_root=runs / "doctor",
        output_monitor_interval_seconds=0.01,
        disk_reserver=lambda path, _size: path.touch(mode=0o600),
    )
    runtime._mount_resolver = DockerMountResolver(
        client,
        "controller",
        expected_roots={
            PurePosixPath(str(source)): False,
            PurePosixPath(str(data)): False,
            PurePosixPath(str(runs)): True,
        },
    )
    return runtime, client, source, data, runs


def _spec(source: Path, runs: Path, *, attempt_id: str = "attempt-1") -> ExecutionSpec:
    attempt_root = runs / attempt_id.replace(":", "-")
    attempt_root.mkdir()
    output = attempt_root / "output"
    output.mkdir()
    return ExecutionSpec(
        command=("python", "-m", "rex.execution.worker"),
        working_directory="/workspace",
        mounts=(
            RuntimeMount(source=source, target="/workspace", read_only=True),
            RuntimeMount(source=output, target="/output", read_only=False),
        ),
        environment={"PYTHONDONTWRITEBYTECODE": "1"},
        timeout_seconds=30,
        memory_bytes=256 * 1024 * 1024,
        nano_cpus=1_000_000_000,
        pids_limit=32,
        run_id="run-1",
        experiment_id="experiment-1",
        attempt_id=attempt_id,
        request_sha256=SHA_A,
        execution_sha256=SHA_B,
        lease_path=attempt_root / "worker.lease.json",
        tmpfs_size_bytes=16 * 1024 * 1024,
    )


def test_docker_runtime_create_verify_lease_collect_cleanup(tmp_path: Path) -> None:
    runtime, client, source, _data, runs = _runtime(tmp_path)
    specification = _spec(source, runs)

    handle = runtime.launch(specification)

    assert runtime.inspect(handle).state == RuntimeLifecycleState.RUNNING
    lease = read_docker_lease(specification.lease_path)
    assert lease.state == "active"
    assert lease.container_id == handle.container_id
    assert client.requests[handle.container_id].environment == {"PYTHONDONTWRITEBYTECODE": "1"}

    result = runtime.collect(handle)

    assert result.outcome == ExecutionOutcome.SUCCESS
    assert result.stdout == "worker output\n"
    assert read_docker_lease(specification.lease_path).state == "closed"
    runtime.cleanup(handle)
    assert handle.container_id in client.removed


def test_real_colon_attempt_id_is_preserved_in_label_and_sanitized_in_name(
    tmp_path: Path,
) -> None:
    runtime, client, source, _data, runs = _runtime(tmp_path)
    attempt_id = "run-e01:cheap-A-candidate-repair-0:fit"

    handle = runtime.launch(_spec(source, runs, attempt_id=attempt_id))

    request = client.requests[handle.container_id]
    assert request.labels["rex.attempt_id"] == attempt_id
    assert ":" not in request.name
    assert read_docker_lease(handle.lease_path).attempt_id == attempt_id


def test_runtime_fails_closed_on_security_drift(tmp_path: Path) -> None:
    runtime, client, source, _data, runs = _runtime(tmp_path)
    handle = runtime.launch(_spec(source, runs))
    client.inspections[handle.container_id]["Config"]["Labels"]["rex.request_sha256"] = "0" * 64

    with pytest.raises(DockerRuntimeError, match="labels changed"):
        runtime.terminate(handle)


def test_duplicate_launch_requires_explicit_recovery(tmp_path: Path) -> None:
    runtime, _client, source, _data, runs = _runtime(tmp_path)
    specification = _spec(source, runs)
    runtime.launch(specification)

    with pytest.raises(DockerLeaseError, match="must be recovered"):
        runtime.launch(specification)


def test_runtime_rejects_lease_outside_runs_and_controller_secret_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, _client, source, _data, runs = _runtime(tmp_path)
    specification = _spec(source, runs)

    with pytest.raises(DockerMountError, match="write access"):
        runtime.launch(replace(specification, lease_path=source / "forbidden-lease.json"))
    with pytest.raises(DockerRuntimeError, match="outside every worker mount"):
        runtime.launch(
            replace(
                specification,
                lease_path=specification.mounts[1].source / "worker-controlled-lease.json",
            )
        )

    secret = "sk-test-never-cross-runtime-boundary"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    with pytest.raises(ExecutionRuntimeError) as caught:
        runtime.launch(replace(specification, command=("python", "-c", secret)))
    assert secret not in str(caught.value)


def test_runtime_classifies_timeout_and_oom(tmp_path: Path) -> None:
    runtime, client, source, _data, runs = _runtime(tmp_path)
    timeout_specification = replace(
        _spec(source, runs, attempt_id="timeout"),
        timeout_seconds=0.02,
    )
    timeout_handle = runtime.launch(timeout_specification)
    client.wait_always_timeout = True
    timeout = runtime.collect(timeout_handle)
    client.wait_always_timeout = False
    assert timeout.outcome == ExecutionOutcome.TIMEOUT
    assert timeout.timed_out

    oom_handle = runtime.launch(_spec(source, runs, attempt_id="oom"))
    client.oom = True
    oom = runtime.collect(oom_handle)
    assert oom.outcome == ExecutionOutcome.OOM
    assert oom.oom_killed


def test_recovery_verifies_exact_daemon_labels_and_container_id(tmp_path: Path) -> None:
    runtime, client, source, _data, runs = _runtime(tmp_path)
    specification = _spec(source, runs)
    handle = runtime.launch(specification)
    lease = read_docker_lease(handle.lease_path)

    recovery = runtime.recover(lease)

    assert recovery.terminated
    recovered_lease = read_docker_lease(handle.lease_path)
    assert recovered_lease.state == "recovered"
    assert runtime.inspect(handle).state == RuntimeLifecycleState.EXITED
    recovered_handle = runtime_handle_from_docker_lease(recovered_lease, timeout_seconds=1)
    runtime.cleanup(recovered_handle)

    invalid = read_docker_lease(handle.lease_path)
    object.__setattr__(invalid, "runtime_kind", "native_macos")
    with pytest.raises(DockerLeaseError, match="native lease"):
        runtime.recover(invalid)


def test_runtime_handle_reconstructs_exact_closed_lease_identity(tmp_path: Path) -> None:
    runtime, _client, source, _data, runs = _runtime(tmp_path)
    specification = _spec(source, runs)
    launched = runtime.launch(specification)
    runtime.collect(launched)
    closed = read_docker_lease(launched.lease_path)

    reconstructed = runtime_handle_from_docker_lease(closed, timeout_seconds=30)

    assert reconstructed.container_id == launched.container_id
    assert reconstructed.worker_image_digest == launched.worker_image_digest
    assert reconstructed.request_sha256 == launched.request_sha256
    assert reconstructed.execution_sha256 == launched.execution_sha256
    assert reconstructed.lease_path == launched.lease_path
    assert runtime.inspect(reconstructed).state == RuntimeLifecycleState.EXITED
    runtime.cleanup(reconstructed)


def test_runtime_handle_rejects_unbounded_timeout_or_detached_lease(tmp_path: Path) -> None:
    runtime, _client, source, _data, runs = _runtime(tmp_path)
    launched = runtime.launch(_spec(source, runs))
    lease = read_docker_lease(launched.lease_path)

    with pytest.raises(DockerLeaseError, match="production bound"):
        runtime_handle_from_docker_lease(lease, timeout_seconds=21_601)
    object.__setattr__(lease, "lease_path", None)
    with pytest.raises(DockerLeaseError, match="source path"):
        runtime_handle_from_docker_lease(lease, timeout_seconds=1)


def test_mount_resolver_rejects_traversal_symlink_escape_and_write_escalation(
    tmp_path: Path,
) -> None:
    runtime, _client, source, _data, runs = _runtime(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (source / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(DockerMountError, match="symlink"):
        runtime.mount_resolver.resolve(
            RuntimeMount(source=source / "escape", target="/workspace", read_only=True)
        )
    with pytest.raises(DockerMountError, match="write access"):
        runtime.mount_resolver.resolve(
            RuntimeMount(source=source, target="/workspace", read_only=False)
        )
    with pytest.raises(DockerMountError, match="outside approved"):
        runtime.mount_resolver.resolve(
            RuntimeMount(source=tmp_path, target="/workspace", read_only=True)
        )
    with pytest.raises(DockerMountError, match="traversal"):
        runtime.mount_resolver.resolve(
            RuntimeMount(
                source=Path(str(runs / "child") + "/../escape"),
                target="/output",
                read_only=False,
            )
        )


def test_worker_environment_rejects_credentials() -> None:
    with pytest.raises(ExecutionRuntimeError, match="secret key"):
        ExecutionSpec(
            command=("python",),
            working_directory="/source",
            mounts=(
                RuntimeMount(source=Path("/source"), target="/source", read_only=True),
                RuntimeMount(source=Path("/runs/output"), target="/output", read_only=False),
            ),
            environment={"OPENAI_API_KEY": "must-not-cross-boundary"},
            timeout_seconds=1,
            memory_bytes=64 * 1024 * 1024,
            nano_cpus=1,
            pids_limit=1,
            run_id="run",
            experiment_id="experiment",
            attempt_id="attempt",
            request_sha256=SHA_A,
            execution_sha256=SHA_B,
            lease_path=Path("/runs/output/lease.json"),
            tmpfs_size_bytes=1,
        )


def test_active_docker_doctor_and_environment_identity(tmp_path: Path) -> None:
    runtime, _client, _source, _data, _runs = _runtime(tmp_path)

    result = runtime.doctor()

    assert result.available
    assert result.safe_for_production
    assert all(check.passed for check in result.checks)
    assert result.environment_identity["worker_image_digest"] == IMAGE_ID
    assert result.environment_identity["container_platform"] == "linux/arm64"
    assert result.environment_identity["dependency_lock_sha256"] == "2" * 64


def test_active_docker_doctor_fails_closed_on_probe_failure(tmp_path: Path) -> None:
    runtime, client, _source, _data, _runs = _runtime(tmp_path)
    client.probe_failure = True

    result = runtime.doctor()

    assert not result.safe_for_production
    assert any(check.name == "dns_denied" and not check.passed for check in result.checks)


def test_production_selector_defaults_to_docker_and_native_is_explicit(tmp_path: Path) -> None:
    runtime, client, _source, _data, _runs = _runtime(tmp_path)
    selected = production_runtime(
        docker_client=client,
        environ={
            "REX_WORKER_IMAGE": IMAGE_ID,
            "REX_CONTROLLER_CONTAINER_ID": "controller",
        },
    )
    assert isinstance(selected, DockerExecutionRuntime)

    with pytest.raises(ExecutionRuntimeError, match="explicit authorization"):
        production_runtime(
            environ={"REX_PRODUCTION_RUNTIME": "native_macos"},
            native_macos_factory=lambda: runtime,
        )


def test_execution_spec_requires_exactly_one_writable_output_mount(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    extra = tmp_path / "extra"
    for path in (source, output, extra):
        path.mkdir()
    base = {
        "command": ("python",),
        "working_directory": "/workspace",
        "environment": {},
        "timeout_seconds": 1,
        "memory_bytes": 64 * 1024 * 1024,
        "nano_cpus": 1,
        "pids_limit": 1,
        "run_id": "run",
        "experiment_id": "experiment",
        "attempt_id": "attempt",
        "request_sha256": SHA_A,
        "execution_sha256": SHA_B,
        "lease_path": tmp_path / "lease.json",
        "tmpfs_size_bytes": 1,
    }

    with pytest.raises(ExecutionRuntimeError, match="exactly one writable mount at /output"):
        ExecutionSpec(
            **base,
            mounts=(
                RuntimeMount(source, "/workspace", True),
                RuntimeMount(output, "/scratch", False),
            ),
        )
    with pytest.raises(ExecutionRuntimeError, match="exactly one writable mount at /output"):
        ExecutionSpec(
            **base,
            mounts=(
                RuntimeMount(source, "/workspace", True),
                RuntimeMount(output, "/output", False),
                RuntimeMount(extra, "/other", False),
            ),
        )


def test_runtime_verifies_bounded_logs_fsize_and_resource_labels(tmp_path: Path) -> None:
    runtime, client, source, _data, runs = _runtime(tmp_path)
    specification = replace(
        _spec(source, runs),
        output_bytes_limit=16 * 1024 * 1024,
        file_size_limit_bytes=4 * 1024 * 1024,
        log_bytes_limit=1024,
        minimum_free_space_bytes=0,
    )

    handle = runtime.launch(specification)
    request = client.requests[handle.container_id]
    inspection = client.inspections[handle.container_id]

    assert request.labels["rex.output_bytes_limit"] == str(16 * 1024 * 1024)
    assert request.labels["rex.log_bytes_limit"] == "1024"
    assert {
        item["Name"]: (item["Soft"], item["Hard"]) for item in inspection["HostConfig"]["Ulimits"]
    }["fsize"] == (4 * 1024 * 1024, 4 * 1024 * 1024)
    assert inspection["HostConfig"]["LogConfig"] == {
        "Type": "local",
        "Config": {"compress": "true", "max-file": "1", "max-size": "1024"},
    }

    inspection["HostConfig"]["LogConfig"]["Config"]["max-file"] = "2"
    with pytest.raises(DockerRuntimeError, match="log-driver bounds drifted"):
        runtime._verify_inspection(
            inspection,
            request,
            container_id=handle.container_id,
            image_id=IMAGE_ID,
        )


def test_runtime_truncates_untrusted_client_logs_to_explicit_cap(tmp_path: Path) -> None:
    runtime, client, source, _data, runs = _runtime(tmp_path)
    client.stdout = b"a" * 10_000
    client.stderr = b"b" * 10_000
    specification = replace(
        _spec(source, runs),
        log_bytes_limit=1024,
        minimum_free_space_bytes=0,
    )

    result = runtime.collect(runtime.launch(specification))

    assert len(result.stdout.encode()) <= 1024
    assert len(result.stderr.encode()) <= 1024
    assert "docker log truncated" in result.stdout
    assert "docker log truncated" in result.stderr


def test_runtime_terminates_and_closes_lease_on_aggregate_output_breach(
    tmp_path: Path,
) -> None:
    runtime, client, source, _data, runs = _runtime(tmp_path)
    specification = replace(
        _spec(source, runs),
        output_bytes_limit=1024,
        file_size_limit_bytes=512,
        minimum_free_space_bytes=0,
    )
    client.output_bytes_on_wait = 1025
    handle = runtime.launch(specification)
    reservation = runtime._reservation_path(handle.lease_path)
    assert reservation.is_file()

    with pytest.raises(DockerOutputLimitExceeded, match="aggregate 1024-byte budget"):
        runtime.collect(handle)

    closed = read_docker_lease(handle.lease_path)
    assert closed.state == "closed"
    assert closed.closed_reason == "output-limit-exceeded"
    assert runtime.inspect(handle).state == RuntimeLifecycleState.EXITED
    assert not reservation.exists()


def test_runtime_fails_before_create_when_disk_reserve_is_insufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, client, source, _data, runs = _runtime(tmp_path)
    specification = replace(
        _spec(source, runs),
        output_bytes_limit=4096,
        file_size_limit_bytes=2048,
        minimum_free_space_bytes=8192,
    )
    monkeypatch.setattr(
        "rex.execution.runtime_docker.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=12_000),
    )

    with pytest.raises(DockerRuntimeError, match="insufficient free space"):
        runtime.launch(specification)

    assert not client.requests
    assert not runtime._reservation_path(specification.lease_path).exists()


def test_inspect_maps_only_typed_exact_absence_to_removed(tmp_path: Path) -> None:
    runtime, client, source, _data, runs = _runtime(tmp_path)
    handle = runtime.launch(_spec(source, runs))
    client.absent.add(handle.container_id)

    assert runtime.inspect(handle).state == RuntimeLifecycleState.REMOVED

    client.absent.clear()
    client.inspect_failure = DockerRuntimeError(f"No such container: {handle.container_id}")
    with pytest.raises(DockerRuntimeError, match="No such container"):
        runtime.inspect(handle)
    client.inspect_failure = DockerContainerNotFound("e" * 64)
    with pytest.raises(DockerRuntimeError, match="different container ID"):
        runtime.inspect(handle)


def test_cli_client_requires_exact_full_id_absence_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = DockerCLIClient(environment={})
    exact = f"Error response from daemon: No such container: {CONTAINER_ID}\n".encode()
    monkeypatch.setattr(
        client,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, b"", exact),
    )
    with pytest.raises(DockerContainerNotFound) as caught:
        client.inspect_container(CONTAINER_ID)
    assert caught.value.container_id == CONTAINER_ID

    wrong = f"No such container: {'e' * 64}\n".encode()
    monkeypatch.setattr(
        client,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, b"", wrong),
    )
    with pytest.raises(DockerRuntimeError, match="inspection failed"):
        client.inspect_container(CONTAINER_ID)
