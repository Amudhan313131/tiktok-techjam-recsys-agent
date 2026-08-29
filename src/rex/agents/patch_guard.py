"""Unified-diff parser and protected-path enforcement."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

import yaml


class PatchRejected(RuntimeError):
    pass


@dataclass(frozen=True)
class PatchPolicy:
    allowed: tuple[str, ...]
    denied: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PatchPolicy":
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(tuple(value.get("allow", value.get("allowed", []))), tuple(value.get("deny", value.get("denied", []))))


def _normalize_patch_path(raw: str) -> str | None:
    value = raw.split("\t", 1)[0].strip()
    if value == "/dev/null":
        return None
    if value.startswith(("a/", "b/")):
        value = value[2:]
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not value:
        raise PatchRejected(f"unsafe patch path: {raw!r}")
    return path.as_posix()


def changed_paths(patch: str) -> tuple[str, ...]:
    if "GIT binary patch" in patch or "Binary files " in patch:
        raise PatchRejected("binary patches are forbidden")
    if "new file mode 120000" in patch or "old mode 120000" in patch:
        raise PatchRejected("symlink patches are forbidden")
    paths: set[str] = set()
    old_path: str | None = None
    saw_hunk = False
    for line in patch.splitlines():
        if line.startswith("--- "):
            old_path = _normalize_patch_path(line[4:])
        elif line.startswith("+++ "):
            new_path = _normalize_patch_path(line[4:])
            path = new_path or old_path
            if path:
                paths.add(path)
        elif line.startswith("@@"):
            saw_hunk = True
    if not paths or not saw_hunk:
        raise PatchRejected("response is not a unified diff with at least one hunk")
    return tuple(sorted(paths))


def _matches(path: str, pattern: str) -> bool:
    if pattern.endswith("/**"):
        return path == pattern[:-3] or path.startswith(pattern[:-2])
    return fnmatch.fnmatchcase(path, pattern)


def validate_patch(
    patch: str,
    policy: PatchPolicy,
    *,
    declared_files: Iterable[str] | None = None,
) -> tuple[str, ...]:
    paths = changed_paths(patch)
    declared = set(declared_files or [])
    for path in paths:
        if path == ".gitmodules":
            raise PatchRejected("submodule configuration is forbidden")
        if any(_matches(path, denied) for denied in policy.denied):
            raise PatchRejected(f"protected path modified: {path}")
        if not any(_matches(path, allowed) for allowed in policy.allowed):
            raise PatchRejected(f"path is outside the patch allowlist: {path}")
        if declared and path not in declared:
            raise PatchRejected(f"path was not declared by the proposal: {path}")
    return paths
