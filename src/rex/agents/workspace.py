"""Isolated git worktree lifecycle for agent-authored patches."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rex.agents.patch_guard import PatchPolicy, validate_patch


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.")
    if not cleaned:
        raise ValueError("workspace name is empty after sanitization")
    return cleaned


@dataclass(frozen=True)
class GitWorkspace:
    root: Path
    branch: str

    @classmethod
    def create(
        cls,
        repository: str | Path,
        worktree_root: str | Path,
        experiment_id: str,
        parent_commit: str,
    ) -> "GitWorkspace":
        repository = Path(repository).resolve()
        name = _safe_name(experiment_id)
        root = Path(worktree_root).resolve() / name
        branch = f"codex/rex-{name}"
        if root.exists():
            raise RuntimeError(f"worktree target already exists: {root}")
        root.parent.mkdir(parents=True, exist_ok=True)
        _git(repository, "worktree", "add", "-b", branch, str(root), parent_commit)
        return cls(root=root, branch=branch)

    def apply(self, patch: str, policy: PatchPolicy, declared_files: list[str]) -> tuple[str, ...]:
        paths = validate_patch(patch, policy, declared_files=declared_files)
        check = subprocess.run(
            ["git", "apply", "--check", "-"],
            cwd=self.root,
            input=patch,
            text=True,
            capture_output=True,
        )
        if check.returncode:
            raise RuntimeError(f"git apply --check failed: {check.stderr[-2000:]}")
        applied = subprocess.run(
            ["git", "apply", "-"], cwd=self.root, input=patch, text=True, capture_output=True
        )
        if applied.returncode:
            raise RuntimeError(f"git apply failed: {applied.stderr[-2000:]}")
        return paths

    def commit(self, message: str) -> str:
        _git(self.root, "add", "--all")
        _git(self.root, "commit", "-m", message)
        return _git(self.root, "rev-parse", "HEAD").strip()


def _git(cwd: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"git {' '.join(arguments)} failed: {result.stderr[-2000:]}")
    return result.stdout
