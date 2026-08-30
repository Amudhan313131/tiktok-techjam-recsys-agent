"""macOS ``sandbox-exec`` backend for production model workers."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from rex.execution.limits import resource_limit_preexec
from rex.execution.sandbox import (
    PreparedSandbox,
    SandboxBackend,
    SandboxDoctorResult,
    SandboxError,
    SandboxPolicy,
    policy_sha256,
)


SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _path_rule(operation: str, paths: Sequence[Path]) -> str:
    filters: list[str] = []
    for path in paths:
        resolved = path.resolve()
        selector = "subpath" if resolved.is_dir() else "literal"
        filters.append(f"({selector} {_quote(str(resolved))})")
    return f"(allow {operation} {' '.join(filters)})" if filters else ""


def _runtime_read_paths() -> tuple[Path, ...]:
    """Read-only roots required by the interpreter and native ML libraries."""

    import sys

    candidates = {
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path("/System"),
        Path("/usr/lib"),
        Path("/usr/share"),
        Path("/Library/Frameworks"),
        Path("/Library/Preferences"),
        Path("/private/var/db/dyld"),
        Path("/private/var/db/timezone"),
        Path("/dev/null"),
        Path("/dev/random"),
        Path("/dev/urandom"),
        # Homebrew's ``opt`` entries are symlinks whose dynamic-library opens
        # are authorized against their resolved Cellar paths by sandbox-exec.
        # Both namespaces are read-only runtime capabilities; neither contains
        # competition data or user credentials.
        Path("/opt/homebrew/Cellar"),
        Path("/opt/homebrew/lib"),
        Path("/opt/homebrew/opt"),
        Path("/usr/local/lib"),
    }
    return tuple(sorted((path.resolve() for path in candidates if path.exists()), key=str))


def render_profile(policy: SandboxPolicy) -> str:
    """Render deterministic SBPL; absence of an allow rule is an explicit deny."""

    readable = tuple(dict.fromkeys((*_runtime_read_paths(), *policy.read_paths)))
    writable = tuple(dict.fromkeys(policy.write_paths))
    lines = [
        "(version 1)",
        "(deny default)",
        "(deny dynamic-code-generation)",
        '(import "system.sb")',
        "(allow process*)",
        "(allow signal (target self))",
        "(allow sysctl-read)",
        "(allow ipc-posix-shm)",
        "(allow file-read-metadata)",
        _path_rule("file-read*", readable),
        _path_rule("file-read* file-write*", writable),
        "(deny network*)",
    ]
    return "\n".join(line for line in lines if line) + "\n"


class MacOSSandboxBackend(SandboxBackend):
    name = "macos-sandbox-exec"

    def doctor(self) -> SandboxDoctorResult:
        executable = shutil.which(str(SANDBOX_EXEC))
        if executable is None or not os.access(executable, os.X_OK):
            return SandboxDoctorResult(
                backend=self.name,
                available=False,
                safe_for_production=False,
                detail=f"sandbox executable is unavailable: {SANDBOX_EXEC}",
            )
        try:
            with tempfile.TemporaryDirectory(prefix="rex-sandbox-doctor-") as temp:
                root = Path(temp)
                allowed = (root / "allowed.txt").resolve()
                denied = (root / "denied.txt").resolve()
                allowed.write_text("allowed", encoding="utf-8")
                denied.write_text("denied", encoding="utf-8")
                profile = root / "doctor.sb"
                profile.write_text(
                    "(version 1)\n"
                    "(deny default)\n"
                    "(deny dynamic-code-generation)\n"
                    '(import "system.sb")\n'
                    "(allow process*)\n"
                    "(allow file-read-metadata)\n"
                    f"(allow file-read* (subpath {_quote('/System')}) "
                    f"(subpath {_quote('/usr/lib')}) (literal {_quote('/bin/cat')}) "
                    f"(literal {_quote(str(allowed))}))\n"
                    "(deny network*)\n",
                    encoding="utf-8",
                )
                permitted = subprocess.run(
                    [executable, "-f", str(profile), "/bin/cat", str(allowed)],
                    text=True,
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
                blocked = subprocess.run(
                    [executable, "-f", str(profile), "/bin/cat", str(denied)],
                    text=True,
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            if permitted.returncode != 0 or permitted.stdout != "allowed":
                return SandboxDoctorResult(
                    backend=self.name,
                    available=True,
                    safe_for_production=False,
                    detail=f"allowed-read probe failed: {permitted.stderr[-500:]}",
                )
            if blocked.returncode == 0:
                return SandboxDoctorResult(
                    backend=self.name,
                    available=True,
                    safe_for_production=False,
                    detail="denied-read probe unexpectedly succeeded",
                )
        except (OSError, subprocess.SubprocessError) as error:
            return SandboxDoctorResult(
                backend=self.name,
                available=True,
                safe_for_production=False,
                detail=f"sandbox doctor failed: {error}",
            )
        return SandboxDoctorResult(
            backend=self.name,
            available=True,
            safe_for_production=True,
            detail="allowed-read succeeded and undeclared-read was denied",
        )

    def prepare(
        self,
        policy: SandboxPolicy,
        command: Sequence[str],
        environment: Mapping[str, str],
        policy_path: Path,
    ) -> PreparedSandbox:
        doctor = self.doctor()
        if not doctor.safe_for_production:
            raise SandboxError(doctor.detail)
        profile = render_profile(policy)
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        policy_path.write_text(profile, encoding="utf-8")
        profile_hash = hashlib.sha256(profile.encode("utf-8")).hexdigest()
        evidence: dict[str, object] = {
            "schema_version": "1.0",
            "mode": "production",
            "backend": self.name,
            "sandboxed": True,
            "doctor": doctor.to_dict(),
            "policy_sha256": policy_sha256(policy),
            "profile_sha256": profile_hash,
            "policy": policy.to_dict(),
            "environment_keys": sorted(environment),
            "command_executable": str(command[0]) if command else "",
        }
        return PreparedSandbox(
            command=(str(SANDBOX_EXEC), "-f", str(policy_path), *command),
            environment=dict(environment),
            preexec_fn=resource_limit_preexec(policy.resource_limits),
            evidence=evidence,
        )
