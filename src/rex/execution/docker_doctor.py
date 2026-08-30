"""Active security and crash-recovery probes for the Docker worker runtime."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rex.data.manifest import canonical_json_bytes, sha256_file
from rex.execution.docker_lease import read_docker_lease
from rex.execution.runtime import (
    DoctorCheck,
    ExecutionOutcome,
    ExecutionSpec,
    RuntimeLifecycleState,
    RuntimeMount,
)

if TYPE_CHECKING:
    from rex.execution.runtime_docker import DockerExecutionRuntime


_PROBE = r"""
import hashlib
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path

source = Path('/doctor/input')
output = Path('/output')
expected = os.environ['REX_DOCTOR_INPUT_SHA256']
checks = {}

try:
    checks['approved_input_read'] = hashlib.sha256(source.read_bytes()).hexdigest() == expected
except Exception:
    checks['approved_input_read'] = False

try:
    with source.open('ab') as handle:
        handle.write(b'forbidden')
    checks['protected_input_read_only'] = False
except OSError:
    checks['protected_input_read_only'] = True

try:
    marker = output / 'worker-write-ok.json'
    marker.write_text('{"ok":true}\n', encoding='utf-8')
    checks['exact_output_writable'] = marker.is_file()
except OSError:
    checks['exact_output_writable'] = False

try:
    Path('/rex-forbidden-write').write_text('forbidden', encoding='utf-8')
    checks['root_read_only'] = False
except OSError:
    checks['root_read_only'] = True

checks['worker_non_root'] = os.getuid() != 0 and os.getgid() != 0
status = Path('/proc/self/status').read_text(encoding='utf-8')
checks['privilege_escalation_disabled'] = any(
    line.split(':', 1)[1].strip() == '1'
    for line in status.splitlines()
    if line.startswith('NoNewPrivs:')
)
checks['capabilities_dropped'] = any(
    line.split(':', 1)[1].strip() == '0000000000000000'
    for line in status.splitlines()
    if line.startswith('CapEff:')
)
checks['docker_socket_absent'] = not any(
    Path(path).exists() for path in ('/var/run/docker.sock', '/run/docker.sock')
)
checks['unapproved_host_mount_absent'] = not any(
    Path(path).exists() for path in ('/host', '/mnt/host', '/run/desktop/mnt/host')
)
for key in (
    'OPENAI' + '_API_KEY', 'ANTHROPIC' + '_API_KEY', 'CLAUDE' + '_API_KEY',
    'AWS_SECRET' + '_ACCESS_KEY', 'GITHUB' + '_TOKEN', 'SSH_AUTH' + '_SOCK',
):
    if key in os.environ:
        checks['credentials_absent'] = False
        break
else:
    checks['credentials_absent'] = True

try:
    socket.getaddrinfo('example.com', 443)
    checks['dns_denied'] = False
except OSError:
    checks['dns_denied'] = True

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.0)
try:
    sock.connect(('1.1.1.1', 443))
    checks['outbound_tcp_denied'] = False
except OSError:
    checks['outbound_tcp_denied'] = True
finally:
    sock.close()

try:
    executable = Path('/tmp/rex-doctor-exec')
    shutil.copy2('/bin/true', executable)
    executable.chmod(0o755)
    subprocess.run([str(executable)], check=False, timeout=2)
    checks['tmp_noexec'] = False
except (OSError, subprocess.SubprocessError):
    checks['tmp_noexec'] = True

checks['memory_limit_visible'] = any(
    Path(path).is_file() and Path(path).read_text(encoding='utf-8').strip() not in {'', 'max'}
    for path in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes')
)
checks['process_limit_visible'] = any(
    Path(path).is_file() and Path(path).read_text(encoding='utf-8').strip() not in {'', 'max'}
    for path in ('/sys/fs/cgroup/pids.max', '/sys/fs/cgroup/pids/pids.max')
)
checks['cpu_limit_visible'] = any(
    Path(path).is_file() and Path(path).read_text(encoding='utf-8').strip() not in {'', 'max', '-1'}
    for path in (
        '/sys/fs/cgroup/cpu.max',
        '/sys/fs/cgroup/cpu/cpu.cfs_quota_us',
    )
)
print(json.dumps(checks, sort_keys=True))
"""


@dataclass(frozen=True)
class DockerDoctorProbeResult:
    checks: tuple[DoctorCheck, ...]
    probe_directory: Path


def _identity_hash(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _parse_probe_output(stdout: str) -> dict[str, bool]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("security probe emitted no result")
    value = json.loads(lines[-1])
    if not isinstance(value, dict) or not value:
        raise ValueError("security probe result is not a non-empty object")
    if any(not isinstance(key, str) or not isinstance(item, bool) for key, item in value.items()):
        raise ValueError("security probe result contains non-boolean checks")
    return value


class DockerSecurityDoctor:
    """Launch real workers to prove the Docker isolation contract end to end."""

    def __init__(
        self,
        runtime: DockerExecutionRuntime,
        *,
        input_path: Path = Path("/source/pyproject.toml"),
        output_root: Path = Path("/runs/control/docker-doctor"),
    ) -> None:
        self.runtime = runtime
        self.input_path = input_path
        self.output_root = output_root

    def run(self) -> DockerDoctorProbeResult:
        if not self.input_path.is_file() or self.input_path.is_symlink():
            raise ValueError(f"Docker doctor input is unavailable or unsafe: {self.input_path}")
        self.output_root.mkdir(parents=True, exist_ok=True)
        if self.output_root.is_symlink():
            raise ValueError("Docker doctor output root may not be a symlink")
        token = _identity_hash(
            {
                "image": self.runtime.worker_image_digest,
                "time_ns": time.time_ns(),
            }
        )[:16]
        probe_directory = self.output_root / token
        probe_directory.mkdir(mode=0o770)
        probe_directory.chmod(0o777)
        input_sha = sha256_file(self.input_path)
        request_sha = _identity_hash(
            {"purpose": "docker-security-doctor", "input_sha256": input_sha, "token": token}
        )
        execution_sha = _identity_hash(
            {
                "request_sha256": request_sha,
                "image_digest": self.runtime.worker_image_digest,
                "security_probe_sha256": hashlib.sha256(_PROBE.encode("utf-8")).hexdigest(),
            }
        )
        specification = ExecutionSpec(
            command=("python", "-I", "-c", _PROBE),
            working_directory="/output",
            mounts=(
                RuntimeMount(source=self.input_path, target="/doctor/input", read_only=True),
                RuntimeMount(source=probe_directory, target="/output", read_only=False),
            ),
            environment={
                "PYTHONDONTWRITEBYTECODE": "1",
                "REX_DOCTOR_INPUT_SHA256": input_sha,
            },
            timeout_seconds=20,
            memory_bytes=256 * 1024 * 1024,
            nano_cpus=1_000_000_000,
            pids_limit=32,
            run_id="docker-doctor",
            experiment_id="security-probe",
            attempt_id=token,
            request_sha256=request_sha,
            execution_sha256=execution_sha,
            lease_path=self.output_root / f"{token}.security-probe.lease.json",
            tmpfs_size_bytes=32 * 1024 * 1024,
        )
        handle = self.runtime.launch(specification)
        try:
            result = self.runtime.collect(handle)
            if result.outcome != ExecutionOutcome.SUCCESS:
                raise ValueError(
                    f"Docker security probe failed with {result.outcome.value}: {result.stderr[-1000:]}"
                )
            values = _parse_probe_output(result.stdout)
            checks = [
                DoctorCheck(name, passed, "active worker probe")
                for name, passed in sorted(values.items())
            ]
            marker = probe_directory / "worker-write-ok.json"
            checks.append(
                DoctorCheck(
                    "controller_observed_output",
                    marker.is_file() and not marker.is_symlink(),
                    str(marker),
                )
            )
        finally:
            status = self.runtime.inspect(handle)
            if status.state == RuntimeLifecycleState.RUNNING:
                self.runtime.terminate(handle)
            self.runtime.cleanup(handle)

        recovery_directory = probe_directory / "recovery"
        recovery_directory.mkdir(mode=0o770)
        recovery_directory.chmod(0o777)
        recovery_request_sha = _identity_hash({"purpose": "docker-recovery-doctor", "token": token})
        recovery_execution_sha = _identity_hash(
            {
                "request_sha256": recovery_request_sha,
                "image_digest": self.runtime.worker_image_digest,
            }
        )
        recovery_specification = ExecutionSpec(
            command=("python", "-I", "-c", "import time; time.sleep(300)"),
            working_directory="/output",
            mounts=(
                RuntimeMount(source=self.input_path, target="/doctor/input", read_only=True),
                RuntimeMount(source=recovery_directory, target="/output", read_only=False),
            ),
            environment={"PYTHONDONTWRITEBYTECODE": "1"},
            timeout_seconds=310,
            memory_bytes=128 * 1024 * 1024,
            nano_cpus=500_000_000,
            pids_limit=16,
            run_id="docker-doctor",
            experiment_id="recovery-probe",
            attempt_id=token,
            request_sha256=recovery_request_sha,
            execution_sha256=recovery_execution_sha,
            lease_path=self.output_root / f"{token}.recovery-probe.lease.json",
            tmpfs_size_bytes=16 * 1024 * 1024,
        )
        recovery_handle = self.runtime.launch(recovery_specification)
        try:
            lease = read_docker_lease(recovery_handle.lease_path)
            recovery = self.runtime.recover(lease)
            recovery_status = self.runtime.inspect(recovery_handle)
            recovered_safely = (
                recovery.terminated
                and recovery.outcome == "orphan-container-terminated"
                and recovery_status.state != RuntimeLifecycleState.RUNNING
            )
            checks.append(
                DoctorCheck(
                    "controller_interrupt_recovery",
                    recovered_safely,
                    recovery.outcome,
                )
            )
        finally:
            status = self.runtime.inspect(recovery_handle)
            if status.state == RuntimeLifecycleState.RUNNING:
                self.runtime.terminate(recovery_handle)
            self.runtime.cleanup(recovery_handle)
        return DockerDoctorProbeResult(checks=tuple(checks), probe_directory=probe_directory)
