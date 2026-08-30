"""Explicit, one-time filesystem handoff for a sealed submission bundle."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from rex.data.manifest import sha256_file

from rex.submission.repository import SubmissionRepositoryError


class FilesystemHandoff:
    """Copy one sealed bundle without pretending an organizer upload API exists."""

    manifest_name = "submission_seal.json"

    @classmethod
    def _validate_tree(cls, root: Path, seal_sha256: str) -> None:
        manifest = root / cls.manifest_name
        if not manifest.is_file() or manifest.is_symlink() or sha256_file(manifest) != seal_sha256:
            raise SubmissionRepositoryError("sealed manifest is missing or has drifted")
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SubmissionRepositoryError(f"sealed manifest is invalid: {error}") from error
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict):
            raise SubmissionRepositoryError("sealed artifact inventory is missing")
        observed: set[str] = set()
        for path in root.rglob("*"):
            if path == manifest:
                continue
            if path.is_symlink():
                raise SubmissionRepositoryError("sealed handoff may not contain symlinks")
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            observed.add(relative)
            reference = artifacts.get(relative)
            if (
                not isinstance(reference, dict)
                or reference.get("sha256") != sha256_file(path)
                or reference.get("size_bytes") != path.stat().st_size
            ):
                raise SubmissionRepositoryError(f"sealed handoff artifact drifted: {relative}")
        if observed != set(artifacts):
            raise SubmissionRepositoryError("sealed handoff inventory drifted")

    def __call__(self, sealed_dir: Path, target_dir: Path, seal_sha256: str) -> Path:
        source = sealed_dir.resolve(strict=True)
        self._validate_tree(source, seal_sha256)
        target = target_dir.resolve()
        if target == source or source in target.parents:
            raise SubmissionRepositoryError("handoff target may not be inside the sealed source")
        temporary = target.with_name(target.name + ".handoff-tmp")
        if target.exists():
            try:
                self._validate_tree(target, seal_sha256)
                return target
            except SubmissionRepositoryError as error:
                raise SubmissionRepositoryError(
                    f"handoff target already exists and differs: {target}"
                ) from error
        if temporary.exists():
            self._validate_tree(temporary, seal_sha256)
        else:
            temporary.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, temporary, symlinks=False)
        self._validate_tree(temporary, seal_sha256)
        os.replace(temporary, target)
        self._validate_tree(target, seal_sha256)
        return target
