"""Trusted Docker-controller entrypoint and immutable source preparation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from rex.execution.artifacts import atomic_write_json


_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONTAINER_ID = re.compile(r"^[0-9a-f]{64}$")


class DockerControllerError(RuntimeError):
    """The trusted controller boundary could not be established safely."""


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
        env={
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        },
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()[-1000:]
        raise DockerControllerError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def _inside(root: Path, candidate: Path, *, label: str) -> Path:
    root = root.resolve(strict=True)
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise DockerControllerError(f"{label} escapes the runs capability: {candidate}") from error
    return candidate


def _source_identity(source: Path, expected_commit: str) -> dict[str, object]:
    if source.is_symlink() or not source.is_dir():
        raise DockerControllerError("/source must be a regular directory, not a symlink")
    top = Path(_git(source, "rev-parse", "--show-toplevel")).resolve()
    if top != source:
        raise DockerControllerError("/source must be the Git repository root")
    commit = _git(source, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise DockerControllerError(
            f"source commit mismatch: image expects {expected_commit}, mounted source is {commit}"
        )
    if _git(source, "status", "--porcelain", "--untracked-files=normal"):
        raise DockerControllerError("production source must be clean and fully committed")
    return {
        "source": str(source),
        "commit": commit,
        "tree": _git(source, "rev-parse", "HEAD^{tree}"),
    }


def _verify_clone(clone: Path, expected_commit: str) -> None:
    if clone.is_symlink() or not clone.is_dir():
        raise DockerControllerError("controller clone is missing, unsafe, or a symlink")
    if _git(clone, "rev-parse", "HEAD") != expected_commit:
        raise DockerControllerError("existing controller clone has the wrong commit")
    if _git(clone, "status", "--porcelain", "--untracked-files=normal"):
        raise DockerControllerError("existing controller clone is not clean")


def _configure_git_identity(clone: Path) -> None:
    """Install a deterministic repository-local identity for agent commits."""

    _git(clone, "config", "--local", "user.name", "REX Autonomous Researcher")
    _git(clone, "config", "--local", "user.email", "rex@localhost.invalid")
    if _git(clone, "config", "--local", "--get", "user.name") != "REX Autonomous Researcher":
        raise DockerControllerError("controller clone Git author name was not persisted")
    if _git(clone, "config", "--local", "--get", "user.email") != "rex@localhost.invalid":
        raise DockerControllerError("controller clone Git author email was not persisted")


def prepare_controller_source(
    *,
    source_root: str | Path,
    runs_root: str | Path,
    expected_commit: str,
    expected_image_digest: str,
) -> tuple[Path, Path]:
    """Verify the mounted source and return a private writable controller clone."""

    if _COMMIT.fullmatch(expected_commit) is None:
        raise DockerControllerError("REX_IMAGE_SOURCE_COMMIT must be a full Git SHA-1")
    if _DIGEST.fullmatch(expected_image_digest) is None:
        raise DockerControllerError("REX_EXPECTED_IMAGE_DIGEST must be an immutable sha256 digest")
    source = Path(source_root).resolve(strict=True)
    runs = Path(runs_root).resolve(strict=True)
    identity = _source_identity(source, expected_commit)
    control = _inside(runs, runs / "control", label="controller root")
    repositories = control / "repos"
    repositories.mkdir(parents=True, exist_ok=True)
    clone = _inside(runs, repositories / expected_commit, label="controller clone")
    if clone.exists():
        _verify_clone(clone, expected_commit)
    else:
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{expected_commit[:12]}-", dir=str(repositories))
        )
        try:
            completed = subprocess.run(
                ["git", "clone", "--no-hardlinks", "--no-checkout", str(source), str(temporary)],
                text=True,
                capture_output=True,
                timeout=180,
                check=False,
                env={
                    "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_TERMINAL_PROMPT": "0",
                },
            )
            if completed.returncode:
                raise DockerControllerError(
                    "could not create controller clone: "
                    + (completed.stderr or completed.stdout).strip()[-1000:]
                )
            _git(temporary, "checkout", "--detach", expected_commit)
            _verify_clone(temporary, expected_commit)
            temporary.replace(clone)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    _configure_git_identity(clone)
    evidence = control / "controller_source.json"
    atomic_write_json(
        evidence,
        {
            "schema_version": "rex.docker-controller-source.v1",
            **identity,
            "controller_clone": str(clone),
            "worker_image_digest": expected_image_digest,
            "identity_sha256": hashlib.sha256(
                json.dumps(
                    {**identity, "worker_image_digest": expected_image_digest},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
    )
    return clone, evidence


def controller_environment(
    source_clone: Path, source: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Return the trusted controller environment without changing secret values."""

    environment = dict(os.environ if source is None else source)
    environment["REX_SOURCE_ROOT"] = str(source_clone)
    platform_name = environment.get("REX_IMAGE_PLATFORM", "")
    architecture = platform_name.removeprefix("linux/")
    if architecture not in {"amd64", "arm64"}:
        raise DockerControllerError("image platform does not select a supported Linux lock")
    environment_lock = source_clone / f"requirements-lock-linux-{architecture}.txt"
    if not environment_lock.is_file():
        raise DockerControllerError("controller clone is missing its Linux dependency lock")
    observed_lock_sha256 = hashlib.sha256(environment_lock.read_bytes()).hexdigest()
    if observed_lock_sha256 != environment.get("REX_DEPENDENCY_LOCK_SHA256"):
        raise DockerControllerError("Linux dependency lock differs from image provenance")
    environment["REX_ENVIRONMENT_LOCK"] = str(environment_lock)
    container_id = environment.get("HOSTNAME", "")
    environment["REX_CONTROLLER_CONTAINER_ID"] = container_id
    environment["REX_CONTROLLER_ID"] = container_id
    if not environment.get("REX_PROCESS_SESSION_HOST"):
        completed = subprocess.run(
            ["docker", "container", "inspect", container_id],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
            env=environment,
        )
        try:
            inspected = json.loads(completed.stdout)
            container = inspected[0]
            full_id = str(container["Id"])
            labels = container["Config"]["Labels"]
        except (json.JSONDecodeError, IndexError, KeyError, TypeError) as error:
            raise DockerControllerError(
                "could not resolve the controller's exact Docker identity"
            ) from error
        if (
            completed.returncode != 0
            or _CONTAINER_ID.fullmatch(full_id) is None
            or not isinstance(labels, dict)
            or labels.get("rex.managed") != "true"
            or labels.get("rex.role") != "controller"
            or container.get("Image") != environment.get("REX_EXPECTED_IMAGE_DIGEST")
        ):
            raise DockerControllerError("controller Docker identity failed closed")
        environment["REX_PROCESS_SESSION_HOST"] = full_id
    return environment


def main(argv: Sequence[str] | None = None) -> int:
    environment = os.environ
    if os.geteuid() == 0:
        raise DockerControllerError("trusted Docker controller must run as a non-root user")
    source = environment.get("REX_MOUNTED_SOURCE_ROOT", "/source")
    runs = environment.get("REX_RUNS_ROOT", "/runs")
    worker_image = environment.get("REX_WORKER_IMAGE", "")
    if not worker_image:
        raise DockerControllerError("REX_WORKER_IMAGE is required")
    if not Path("/var/run/docker.sock").is_socket():
        raise DockerControllerError("trusted controller has no Docker daemon socket")
    clone, _ = prepare_controller_source(
        source_root=source,
        runs_root=runs,
        expected_commit=environment.get("REX_IMAGE_SOURCE_COMMIT", ""),
        expected_image_digest=environment.get("REX_EXPECTED_IMAGE_DIGEST", ""),
    )
    updated = controller_environment(clone)
    os.environ.clear()
    os.environ.update(updated)
    os.chdir(clone)
    from rex.cli import main as rex_main

    return int(rex_main(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
