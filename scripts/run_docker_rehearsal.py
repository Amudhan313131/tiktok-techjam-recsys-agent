#!/usr/bin/env python3
"""Host-side supervisor for one clean, crash-tested Docker production run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


class DockerRehearsalError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _runs_path_on_host(value: object, output_root: Path) -> Path:
    """Translate a controller-visible /runs artifact into its verified host path."""

    container_path = Path(str(value))
    try:
        relative = container_path.relative_to("/runs")
    except ValueError as error:
        raise DockerRehearsalError(
            f"bundle artifact is not rooted in the controller /runs mount: {container_path}"
        ) from error
    if not relative.parts:
        raise DockerRehearsalError("bundle artifact may not name the /runs mount root")
    root = output_root.resolve(strict=True)
    try:
        host_path = (root / relative).resolve(strict=True)
        host_path.relative_to(root)
    except (OSError, ValueError) as error:
        raise DockerRehearsalError(
            f"bundle artifact escapes or is missing from the host run root: {container_path}"
        ) from error
    return host_path


def _run(
    arguments: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: float = 120,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(arguments),
            cwd=cwd,
            text=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise DockerRehearsalError(f"could not execute {arguments[0]}: {error}") from error
    if check and completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()[-3000:]
        raise DockerRehearsalError(
            f"command failed ({completed.returncode}): {' '.join(arguments)}: {detail}"
        )
    return completed


def _git(root: Path, *arguments: str) -> str:
    return _run(("git", *arguments), cwd=root).stdout.strip()


def _safe_host_path(value: str | Path, *, directory: bool = True) -> Path:
    raw = str(value)
    if any(character in raw for character in ("\x00", "\n", "\r", ",")):
        raise DockerRehearsalError(f"unsafe bind path: {raw!r}")
    path = Path(value).expanduser().resolve(strict=True)
    if path.is_symlink() or (directory and not path.is_dir()):
        raise DockerRehearsalError(f"bind path is missing, unsafe, or not a directory: {path}")
    return path


def _image_identity(image: str, source_commit: str) -> tuple[str, dict[str, str]]:
    raw = _run(("docker", "image", "inspect", image), timeout=60).stdout
    values = json.loads(raw)
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise DockerRehearsalError("Docker image inspection is malformed")
    inspection = values[0]
    image_id = str(inspection.get("Id", ""))
    if IMAGE_ID.fullmatch(image_id) is None:
        raise DockerRehearsalError("Docker image has no immutable local image ID")
    config = inspection.get("Config")
    labels = config.get("Labels") if isinstance(config, dict) else None
    if not isinstance(labels, dict):
        raise DockerRehearsalError("Docker image has no provenance labels")
    required = {
        "source_commit": "org.opencontainers.image.revision",
        "dependency_lock_sha256": "org.rex.dependency-lock-sha256",
        "pyproject_sha256": "org.rex.pyproject-sha256",
        "starter_kit_sha256": "org.rex.starter-kit-sha256",
        "base_image_digest": "org.rex.base-image-digest",
    }
    metadata = {name: str(labels.get(label, "")) for name, label in required.items()}
    if metadata["source_commit"] != source_commit:
        raise DockerRehearsalError("Docker image source commit does not match the clean repository")
    for name in ("dependency_lock_sha256", "pyproject_sha256", "starter_kit_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", metadata[name]) is None:
            raise DockerRehearsalError(f"Docker image has invalid {name} provenance")
    if IMAGE_ID.fullmatch(metadata["base_image_digest"]) is None:
        raise DockerRehearsalError("Docker image has invalid base-image provenance")
    return image_id, metadata


class Supervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.source = _safe_host_path(args.source_root)
        self.data = _safe_host_path(args.data_dir)
        self.output = Path(args.output_dir).expanduser().resolve()
        if not SAFE_RUN_ID.fullmatch(args.run_id):
            raise DockerRehearsalError("run ID is not one safe path component")
        if self.output.exists() and any(self.output.iterdir()):
            raise DockerRehearsalError("Docker rehearsal output directory must be new or empty")
        self.output.mkdir(parents=True, exist_ok=True)
        if self.output.is_symlink():
            raise DockerRehearsalError("Docker rehearsal output may not be a symlink")
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            os.chown(self.output, 10001, 10001)
        self.commit = _git(self.source, "rev-parse", "HEAD")
        if not re.fullmatch(r"[0-9a-f]{40}", self.commit):
            raise DockerRehearsalError("source HEAD is not a full Git commit")
        if _git(self.source, "status", "--porcelain", "--untracked-files=normal"):
            raise DockerRehearsalError("source must be clean and fully committed")
        self.image_id, self.image_metadata = _image_identity(args.image, self.commit)
        self.started = time.time()
        self.deadline = self.started + args.wall_clock_seconds
        self.state_path = self.output / "docker_rehearsal_state.json"
        self.logs = self.output / "controller-logs"
        self.logs.mkdir()
        self.containers: list[str] = []
        self.fault: dict[str, Any] = {"state": "pending", "count": 0}
        self._state("initializing")

    def _remaining(self) -> float:
        return max(0.0, self.deadline - time.time())

    def _state(self, phase: str, **extra: Any) -> None:
        _atomic_json(
            self.state_path,
            {
                "schema_version": "rex.docker-rehearsal.v1",
                "phase": phase,
                "run_id": self.args.run_id,
                "source_commit": self.commit,
                "worker_image_digest": self.image_id,
                "image_metadata": self.image_metadata,
                "started_at": datetime.fromtimestamp(self.started, timezone.utc).isoformat(),
                "deadline_epoch": self.deadline,
                "elapsed_seconds": max(0.0, time.time() - self.started),
                "remaining_seconds": self._remaining(),
                "validation_only": True,
                "test_prediction_enabled": False,
                "manual_interventions": 0,
                "controlled_fault": self.fault,
                "controllers": list(self.containers),
                "updated_at": _utc_now(),
                **extra,
            },
        )

    def _mount(self, source: Path, target: str, *, read_only: bool) -> str:
        value = f"type=bind,src={source},dst={target}"
        return value + (",readonly" if read_only else "")

    def _controller_create(self, name: str, command: Sequence[str]) -> str:
        host_uid = os.getuid() if hasattr(os, "getuid") else 10001
        host_gid = os.getgid() if hasattr(os, "getgid") else 10001
        controller_uid = host_uid if host_uid != 0 else 10001
        controller_gid = host_gid if host_uid != 0 else 10001
        socket_path = Path(self.args.docker_socket).resolve(strict=True)
        # Docker Desktop forwards the macOS socket into its Linux VM as
        # root:root, regardless of the forwarding socket's host-side GID.
        socket_gid = 0 if sys.platform == "darwin" else socket_path.stat().st_gid
        arguments = [
            "docker",
            "container",
            "create",
            "--name",
            name,
            "--label",
            "rex.managed=true",
            "--label",
            "rex.role=controller",
            "--label",
            f"rex.run_id={self.args.run_id}",
            "--security-opt",
            "no-new-privileges=true",
            "--user",
            f"{controller_uid}:{controller_gid}",
            "--group-add",
            str(socket_gid),
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,exec,mode=1777",
            "--mount",
            self._mount(self.source, "/source", read_only=True),
            "--mount",
            self._mount(self.data, "/data", read_only=True),
            "--mount",
            self._mount(self.output, "/runs", read_only=False),
            "--mount",
            self._mount(socket_path, "/var/run/docker.sock", read_only=False),
            "--env",
            "HOME=/tmp/rex-home",
            "--env",
            "REX_PRODUCTION_RUNTIME=docker",
            "--env",
            f"REX_WORKER_IMAGE={self.image_id}",
            "--env",
            f"REX_EXPECTED_IMAGE_DIGEST={self.image_id}",
            "--env",
            f"REX_IMAGE_SOURCE_COMMIT={self.commit}",
            "--env",
            f"REX_PROCESS_SESSION_HOST={name}",
            "--env",
            "REX_MOUNTED_SOURCE_ROOT=/source",
            "--env",
            "REX_DATA_ROOT=/data",
            "--env",
            "REX_RUNS_ROOT=/runs",
            "--env",
            "REX_BASELINE_CACHE_DIR=/runs/cache/baseline",
            "--env",
            "REX_CONTROL_CACHE_DIR=/runs/cache/controls",
        ]
        for key in ("OPENAI_API_KEY", "OPENAI_MODEL"):
            if os.environ.get(key):
                arguments.extend(("--env", key))
        for host_path, target in (
            (self.args.codex_home, "/run/rex-auth/codex"),
            (self.args.claude_home, "/run/rex-auth/claude"),
        ):
            if host_path:
                arguments.extend(
                    ("--mount", self._mount(_safe_host_path(host_path), target, read_only=True))
                )
        arguments.extend((self.image_id, *command))
        container_id = _run(arguments, timeout=90).stdout.strip()
        if re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
            raise DockerRehearsalError("Docker did not return a full controller container ID")
        self.containers.append(container_id)
        return container_id

    def _capture_logs(self, container_id: str, label: str) -> None:
        completed = _run(("docker", "container", "logs", container_id), check=False, timeout=60)
        (self.logs / f"{label}.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (self.logs / f"{label}.stderr.log").write_text(completed.stderr, encoding="utf-8")

    def _remove(self, container_id: str) -> None:
        _run(("docker", "container", "rm", "--force", container_id), check=False, timeout=30)

    def _one_shot(self, label: str, command: Sequence[str], *, timeout: float) -> None:
        name = f"rex-{self.args.run_id}-{label}".lower()
        container = self._controller_create(name, command)
        try:
            _run(("docker", "container", "start", container), timeout=30)
            result = _run(("docker", "container", "wait", container), timeout=timeout)
            try:
                exit_code = int(result.stdout.strip())
            except ValueError as error:
                raise DockerRehearsalError(f"{label} returned an invalid exit status") from error
            self._capture_logs(container, label)
            if exit_code:
                raise DockerRehearsalError(f"{label} controller exited with {exit_code}")
        finally:
            self._remove(container)

    def _active_worker_lease(self, controller: str) -> tuple[Path, dict[str, Any]] | None:
        run_root = self.output / self.args.run_id
        if not run_root.is_dir():
            return None
        controller_state = _run(
            ("docker", "container", "inspect", "--format", "{{.State.Running}}", controller),
            check=False,
            timeout=10,
        )
        for path in sorted(run_root.rglob("worker_lease.json")):
            try:
                lease = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if lease.get("runtime_kind") != "docker" or lease.get("state") != "active":
                continue
            container_id = str(lease.get("container_id", ""))
            if re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
                raise DockerRehearsalError("active Docker lease has an invalid container ID")
            raw_result = _run(
                ("docker", "container", "inspect", container_id),
                check=False,
                timeout=30,
            )
            if raw_result.returncode != 0:
                try:
                    refreshed = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    refreshed = {}
                if refreshed.get("state") != "active":
                    continue
                raise DockerRehearsalError(
                    "active Docker lease still names a missing worker container"
                )
            raw = raw_result.stdout
            inspected = json.loads(raw)[0]
            labels = inspected.get("Config", {}).get("Labels", {})
            for key in (
                "rex.run_id",
                "rex.experiment_id",
                "rex.attempt_id",
                "rex.request_sha256",
                "rex.execution_sha256",
            ):
                expected = str(lease.get(key.removeprefix("rex."), ""))
                if str(labels.get(key, "")) != expected:
                    raise DockerRehearsalError(
                        f"active worker label {key} does not match its lease"
                    )
            if controller_state.stdout.strip() != "true":
                return None
            return path, lease
        return None

    def _wait_for_worker(self, controller: str) -> tuple[Path, dict[str, Any]]:
        while self._remaining() > 60:
            observed = self._active_worker_lease(controller)
            if observed is not None:
                return observed
            running = _run(
                ("docker", "container", "inspect", "--format", "{{.State.Running}}", controller),
                check=False,
                timeout=10,
            )
            if running.returncode or running.stdout.strip() != "true":
                self._capture_logs(controller, "run-initial")
                raise DockerRehearsalError("controller exited before leasing a Docker worker")
            time.sleep(1)
        raise DockerRehearsalError("deadline approached before controlled fault injection")

    def _start_main(self, label: str, *, resume: bool) -> str:
        clone = f"/runs/control/repos/{self.commit}"
        command = [
            "run",
            "--config",
            f"{clone}/configs/run/production.yaml",
            "--external-deadline-epoch-ms",
            str(int(self.deadline * 1000)),
            "--llm",
            self.args.llm,
        ]
        command.extend(("--resume" if resume else "--run-id", self.args.run_id))
        if self.args.llm == "openai_api":
            command.append("--authorize-paid-api")
        name = f"rex-{self.args.run_id}-{label}".lower()
        container = self._controller_create(name, command)
        _run(("docker", "container", "start", container), timeout=30)
        return container

    def _wait_controller(self, controller: str, label: str) -> int:
        result = _run(
            ("docker", "container", "wait", controller),
            timeout=max(30.0, self._remaining()),
        )
        self._capture_logs(controller, label)
        try:
            return int(result.stdout.strip())
        except ValueError as error:
            raise DockerRehearsalError("Docker wait returned an invalid controller exit") from error

    def _seal(self) -> dict[str, Any]:
        best_root = self.output / self.args.run_id / "best-valid"
        best_manifest = best_root / "best_valid_manifest.json"
        try:
            best_payload = json.loads(best_manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise DockerRehearsalError(f"best-valid manifest is invalid: {error}") from error
        if not isinstance(best_payload, dict) or {
            "schema_version": best_payload.get("schema_version"),
            "kind": best_payload.get("kind"),
            "run_id": best_payload.get("run_id"),
            "test_prediction_created": best_payload.get("test_prediction_created"),
            "test_scored": best_payload.get("test_scored"),
        } != {
            "schema_version": "1.0",
            "kind": "best_valid",
            "run_id": self.args.run_id,
            "test_prediction_created": False,
            "test_scored": False,
        }:
            raise DockerRehearsalError("best-valid manifest contract is invalid")
        best_artifacts = best_payload.get("artifacts")
        if not isinstance(best_artifacts, dict) or not best_artifacts:
            raise DockerRehearsalError("best-valid manifest has no artifact index")
        verified_best_files = [best_manifest]
        for name, reference in sorted(best_artifacts.items()):
            if not isinstance(reference, dict):
                raise DockerRehearsalError(f"best-valid artifact {name} is malformed")
            path = _runs_path_on_host(reference.get("path", ""), self.output)
            try:
                path.relative_to(best_root.resolve(strict=True))
            except ValueError as error:
                raise DockerRehearsalError(
                    f"best-valid artifact {name} escapes its immutable bundle"
                ) from error
            if path.is_symlink() or _sha256(path) != reference.get("sha256"):
                raise DockerRehearsalError(f"best-valid artifact {name} hash drifted")
            if path.stat().st_size != int(reference.get("size_bytes", -1)):
                raise DockerRehearsalError(f"best-valid artifact {name} size drifted")
            verified_best_files.append(path)
        required = [
            self.output / self.args.run_id / "state.sqlite3",
            self.output / self.args.run_id / "report" / "evidence_index.json",
            self.output / self.args.run_id / "report" / "results.json",
            self.output / self.args.run_id / "report" / "iteration_logs.json",
            self.output / self.args.run_id / "report" / "resources.json",
            self.output / self.args.run_id / "report" / "manual_interventions.json",
            self.output / self.args.run_id / "report" / "environment_identity.json",
            *verified_best_files,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise DockerRehearsalError(
                "completed Docker run is missing report evidence: " + ", ".join(missing)
            )
        interventions_path = self.output / self.args.run_id / "report" / "manual_interventions.json"
        try:
            interventions = json.loads(interventions_path.read_text(encoding="utf-8"))
            manual_interventions = int(interventions["manual_intervention_count"])
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise DockerRehearsalError(f"manual-intervention report is invalid: {error}") from error
        if manual_interventions < 0:
            raise DockerRehearsalError("manual-intervention count cannot be negative")
        manifest = {
            "schema_version": "rex.docker-r3-manifest.v1",
            "run_id": self.args.run_id,
            "source_commit": self.commit,
            "worker_image_digest": self.image_id,
            "controlled_failure": self.fault,
            "manual_interventions": manual_interventions,
            "validation_only": True,
            "test_prediction_created": False,
            "test_scored": False,
            "elapsed_seconds": time.time() - self.started,
            "files": {
                str(path.relative_to(self.output)): {
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in required
            },
            "sealed_at": _utc_now(),
        }
        path = self.output / "docker_r3_manifest.json"
        _atomic_json(path, manifest)
        return {**manifest, "manifest_path": str(path), "manifest_sha256": _sha256(path)}

    def run(self) -> dict[str, Any]:
        try:
            clone = f"/runs/control/repos/{self.commit}"
            if not (self.output / "data" / "data_manifest.json").is_file():
                self._one_shot(
                    "bootstrap",
                    ("bootstrap", "--data-dir", "/data", "--output-dir", "/runs/data"),
                    timeout=min(1800, self._remaining()),
                )
            doctor = [
                "doctor",
                "--config",
                f"{clone}/configs/run/production.yaml",
                "--tree",
                "--llm",
                self.args.llm,
            ]
            if self.args.llm != "fixed":
                doctor.append("--live")
            self._one_shot("doctor", doctor, timeout=min(900, self._remaining()))
            initial = self._start_main("initial", resume=False)
            self._state("running", controller_id=initial)
            lease_path, lease = self._wait_for_worker(initial)
            self.fault = {
                "state": "intent-recorded",
                "count": 0,
                "controller_id": initial,
                "worker_container_id": lease["container_id"],
                "lease_path": str(lease_path),
                "lease_sha256": _sha256(lease_path),
                "recorded_at": _utc_now(),
            }
            self._state("fault-intent-recorded")
            _run(("docker", "container", "kill", initial), timeout=30)
            self._capture_logs(initial, "run-initial")
            self.fault = {
                **self.fault,
                "state": "injected",
                "count": 1,
                "injected_at": _utc_now(),
            }
            self._state("fault-injected")
            resumed = self._start_main("resumed", resume=True)
            self.fault = {**self.fault, "state": "resume-started", "resume_id": resumed}
            self._state("running-resumed", controller_id=resumed)
            exit_code = self._wait_controller(resumed, "run-resumed")
            if exit_code:
                raise DockerRehearsalError(f"resumed controller exited with {exit_code}")
            self.fault = {**self.fault, "state": "recovered-and-complete"}
            manifest = self._seal()
            self._state("complete", manifest=manifest)
            return manifest
        except BaseException as error:
            for container in self.containers:
                _run(("docker", "container", "kill", container), check=False, timeout=10)
            cleanup = self._cleanup_failed_workers()
            self._state(
                "failed",
                error_type=type(error).__name__,
                error=str(error),
                failed_worker_cleanup=cleanup,
            )
            raise
        finally:
            for container in self.containers:
                self._remove(container)

    def _cleanup_failed_workers(self) -> list[dict[str, object]]:
        """Remove only exact, label-verified workers referenced by durable leases."""

        records: list[dict[str, object]] = []
        run_root = self.output / self.args.run_id
        if not run_root.is_dir():
            return records
        for lease_path in sorted(run_root.rglob("worker_lease.json")):
            try:
                lease = json.loads(lease_path.read_text(encoding="utf-8"))
                container_id = str(lease["container_id"])
            except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
                continue
            if re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
                continue
            inspected = _run(
                ("docker", "container", "inspect", container_id), check=False, timeout=30
            )
            if inspected.returncode != 0:
                records.append({"container_id": container_id, "outcome": "already-absent"})
                continue
            try:
                container = json.loads(inspected.stdout)[0]
            except (json.JSONDecodeError, IndexError, TypeError):
                records.append({"container_id": container_id, "outcome": "inspect-invalid"})
                continue
            labels = container.get("Config", {}).get("Labels", {})
            expected = {
                key: str(lease.get(key.removeprefix("rex."), ""))
                for key in (
                    "rex.run_id",
                    "rex.experiment_id",
                    "rex.attempt_id",
                    "rex.request_sha256",
                    "rex.execution_sha256",
                )
            }
            if (
                not isinstance(labels, dict)
                or labels.get("rex.managed") != "true"
                or container.get("Image") != lease.get("worker_image_id")
                or any(labels.get(key) != value for key, value in expected.items())
            ):
                records.append({"container_id": container_id, "outcome": "identity-mismatch"})
                continue
            running = bool(container.get("State", {}).get("Running"))
            if running:
                _run(("docker", "container", "kill", container_id), check=False, timeout=30)
            removed = _run(
                ("docker", "container", "rm", "--force", container_id),
                check=False,
                timeout=30,
            )
            records.append(
                {
                    "container_id": container_id,
                    "outcome": "removed" if removed.returncode == 0 else "remove-failed",
                    "was_running": running,
                }
            )
        if records:
            _atomic_json(self.output / "failed_worker_cleanup.json", {"workers": records})
        return records


def command_start(args: argparse.Namespace) -> int:
    result = Supervisor(args).run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_status(args: argparse.Namespace) -> int:
    state = Path(args.output_dir).expanduser().resolve() / "docker_rehearsal_state.json"
    if not state.is_file():
        raise DockerRehearsalError(f"Docker rehearsal state is missing: {state}")
    payload = json.loads(state.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(required=True)
    start = sub.add_parser("start")
    start.add_argument("--source-root", type=Path, required=True)
    start.add_argument("--data-dir", type=Path, required=True)
    start.add_argument("--output-dir", type=Path, required=True)
    start.add_argument("--run-id", required=True)
    start.add_argument("--image", default="rex:local")
    start.add_argument(
        "--llm", choices=("fixed", "codex_cli", "claude_cli", "openai_api", "auto"), default="fixed"
    )
    start.add_argument("--codex-home", type=Path)
    start.add_argument("--claude-home", type=Path)
    start.add_argument("--docker-socket", type=Path, default=Path("/var/run/docker.sock"))
    start.add_argument("--wall-clock-seconds", type=int, default=21600)
    start.set_defaults(handler=command_start)
    status = sub.add_parser("status")
    status.add_argument("--output-dir", type=Path, required=True)
    status.set_defaults(handler=command_status)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (DockerRehearsalError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"docker-rehearsal: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
