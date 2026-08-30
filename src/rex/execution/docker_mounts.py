"""Resolve controller paths to daemon-side Docker bind sources safely."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Protocol

from rex.execution.runtime import ExecutionRuntimeError, RuntimeMount


class DockerMountError(ExecutionRuntimeError):
    """A requested Docker mount could not be proven to be within an approved root."""


class ContainerInspector(Protocol):
    def inspect_container(self, container_id: str) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ApprovedDockerRoot:
    controller_path: PurePosixPath
    daemon_source: str
    writable: bool


@dataclass(frozen=True)
class ResolvedDockerMount:
    controller_source: Path
    daemon_source: str
    target: str
    read_only: bool


_EXPECTED_ROOTS = {
    PurePosixPath("/source"): False,
    PurePosixPath("/data"): False,
    PurePosixPath("/runs"): True,
}


def _safe_daemon_source(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise DockerMountError("controller bind mount has no daemon source")
    if any(character in value for character in ("\x00", "\n", "\r", ",")):
        raise DockerMountError("controller bind source contains unsafe characters")
    normalized = value.replace("\\", "/").lower()
    if normalized.endswith("/docker.sock") or normalized.endswith("/containerd.sock"):
        raise DockerMountError("container runtime sockets are never approved mount roots")
    if not (value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value)):
        raise DockerMountError("controller bind source must be an absolute daemon path")
    return value


def _join_daemon_source(root: str, relative: PurePosixPath) -> str:
    if not relative.parts:
        return root
    if re.match(r"^[A-Za-z]:[\\/]", root):
        return str(PureWindowsPath(root).joinpath(*relative.parts))
    return str(PurePosixPath(root).joinpath(*relative.parts))


def _reject_symlink_components(root: Path, candidate: Path, *, allow_missing_leaf: bool) -> None:
    """Reject every symlink from an approved root through the requested path."""

    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise DockerMountError(f"mount path escaped approved root: {candidate}") from error
    current = root
    if current.is_symlink():
        raise DockerMountError(f"approved controller root may not be a symlink: {root}")
    if not current.is_dir():
        raise DockerMountError(f"approved controller root is unavailable: {root}")
    for index, part in enumerate(relative.parts):
        current = current / part
        if current.is_symlink():
            raise DockerMountError(f"mount path contains a symlink component: {current}")
        if not os.path.lexists(current):
            if allow_missing_leaf and index == len(relative.parts) - 1:
                return
            raise DockerMountError(f"mount path does not exist inside the controller: {current}")


class DockerMountResolver:
    """Translate trusted controller paths through daemon-reported root mounts.

    The daemon-side source is never guessed from a controller path.  It is
    derived only from the controller's inspected ``/source``, ``/data`` and
    ``/runs`` bind mounts.
    """

    def __init__(
        self,
        inspector: ContainerInspector,
        controller_id: str,
        *,
        expected_roots: Mapping[PurePosixPath, bool] | None = None,
    ) -> None:
        if not controller_id or any(character in controller_id for character in "\x00\n\r"):
            raise DockerMountError("controller container identity is invalid")
        self._inspector = inspector
        self._controller_id = controller_id
        self._expected_roots = dict(_EXPECTED_ROOTS if expected_roots is None else expected_roots)
        self._roots = self._inspect_roots()

    @property
    def roots(self) -> tuple[ApprovedDockerRoot, ...]:
        return tuple(self._roots[path] for path in sorted(self._roots, key=str))

    def refresh(self) -> None:
        self._roots = self._inspect_roots()

    def _inspect_roots(self) -> dict[PurePosixPath, ApprovedDockerRoot]:
        try:
            inspection = self._inspector.inspect_container(self._controller_id)
        except Exception as error:  # client implementations expose different errors
            raise DockerMountError(f"could not inspect controller container: {error}") from error
        observed: dict[PurePosixPath, ApprovedDockerRoot] = {}
        mounts = inspection.get("Mounts")
        if not isinstance(mounts, list):
            raise DockerMountError("controller inspection has no mount inventory")
        for value in mounts:
            if not isinstance(value, Mapping):
                continue
            destination_value = value.get("Destination")
            if not isinstance(destination_value, str):
                continue
            destination = PurePosixPath(destination_value)
            if destination not in self._expected_roots:
                continue
            if value.get("Type") != "bind":
                raise DockerMountError(f"controller root {destination} is not a bind mount")
            propagation = value.get("Propagation")
            if propagation not in (None, "rprivate"):
                raise DockerMountError(
                    f"controller root {destination} has unsafe mount propagation {propagation!r}"
                )
            if destination in observed:
                raise DockerMountError(f"controller root {destination} is mounted more than once")
            writable = bool(value.get("RW"))
            if writable != self._expected_roots[destination]:
                expected = "writable" if self._expected_roots[destination] else "read-only"
                raise DockerMountError(f"controller root {destination} must be {expected}")
            source = _safe_daemon_source(value.get("Source"))
            observed[destination] = ApprovedDockerRoot(destination, source, writable)
        missing = sorted(set(self._expected_roots) - set(observed), key=str)
        if missing:
            raise DockerMountError(
                "controller is missing required bind root(s): " + ", ".join(map(str, missing))
            )
        return observed

    def resolve(
        self,
        mount: RuntimeMount,
        *,
        allow_missing_leaf: bool = False,
    ) -> ResolvedDockerMount:
        raw = str(mount.source)
        if not raw.startswith("/") or any(character in raw for character in "\x00\n\r"):
            raise DockerMountError("controller mount source must be an absolute POSIX path")
        raw_parts = raw.split("/")
        if "." in raw_parts or ".." in raw_parts:
            raise DockerMountError("controller mount source contains traversal components")
        candidate = PurePosixPath(raw)
        matches = [root for root in self._roots if candidate == root or root in candidate.parents]
        if not matches:
            raise DockerMountError(f"mount source is outside approved controller roots: {raw}")
        selected = max(matches, key=lambda item: len(item.parts))
        root = self._roots[selected]
        if not mount.read_only and not root.writable:
            raise DockerMountError(f"write access is forbidden beneath {selected}")
        local_root = Path(str(selected))
        local_candidate = Path(raw)
        _reject_symlink_components(
            local_root,
            local_candidate,
            allow_missing_leaf=allow_missing_leaf,
        )
        relative = candidate.relative_to(selected)
        daemon_source = _join_daemon_source(root.daemon_source, relative)
        _safe_daemon_source(daemon_source)
        return ResolvedDockerMount(
            controller_source=local_candidate,
            daemon_source=daemon_source,
            target=mount.target,
            read_only=mount.read_only,
        )

    def resolve_all(
        self,
        mounts: tuple[RuntimeMount, ...],
        *,
        allow_missing_leaf: bool = False,
    ) -> tuple[ResolvedDockerMount, ...]:
        resolved = tuple(
            self.resolve(mount, allow_missing_leaf=allow_missing_leaf) for mount in mounts
        )
        daemon_targets = [item.target for item in resolved]
        if len(daemon_targets) != len(set(daemon_targets)):
            raise DockerMountError("resolved Docker mount targets are not unique")
        writable = [item for item in resolved if not item.read_only]
        if len(writable) != 1:
            raise DockerMountError("Docker worker requires exactly one writable result mount")
        if writable[0].target != "/output":
            raise DockerMountError(
                "Docker worker writable result mount must target exactly /output"
            )
        return resolved
