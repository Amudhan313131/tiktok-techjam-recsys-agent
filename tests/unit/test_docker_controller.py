from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from rex.execution.docker_controller import DockerControllerError, prepare_controller_source


DIGEST = "sha256:" + "a" * 64


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init")
    _git(source, "config", "user.name", "REX Test")
    _git(source, "config", "user.email", "rex@example.invalid")
    (source / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(source, "add", "model.py")
    _git(source, "commit", "-m", "fixture")
    return source, _git(source, "rev-parse", "HEAD")


def test_controller_clones_verified_clean_source_idempotently(tmp_path: Path) -> None:
    source, commit = _repository(tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()

    clone, evidence = prepare_controller_source(
        source_root=source,
        runs_root=runs,
        expected_commit=commit,
        expected_image_digest=DIGEST,
    )
    replayed, replay_evidence = prepare_controller_source(
        source_root=source,
        runs_root=runs,
        expected_commit=commit,
        expected_image_digest=DIGEST,
    )

    assert clone == replayed == runs / "control" / "repos" / commit
    assert _git(clone, "rev-parse", "HEAD") == commit
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert evidence == replay_evidence
    assert payload["worker_image_digest"] == DIGEST
    assert payload["commit"] == commit
    assert _git(clone, "config", "--local", "--get", "user.name") == ("REX Autonomous Researcher")
    assert _git(clone, "config", "--local", "--get", "user.email") == ("rex@localhost.invalid")


def test_controller_rejects_dirty_source(tmp_path: Path) -> None:
    source, commit = _repository(tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    (source / "untracked.txt").write_text("unsafe\n", encoding="utf-8")

    with pytest.raises(DockerControllerError, match="clean"):
        prepare_controller_source(
            source_root=source,
            runs_root=runs,
            expected_commit=commit,
            expected_image_digest=DIGEST,
        )


def test_controller_rejects_source_or_image_identity_drift(tmp_path: Path) -> None:
    source, commit = _repository(tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()

    with pytest.raises(DockerControllerError, match="source commit mismatch"):
        prepare_controller_source(
            source_root=source,
            runs_root=runs,
            expected_commit="b" * 40,
            expected_image_digest=DIGEST,
        )
    with pytest.raises(DockerControllerError, match="immutable"):
        prepare_controller_source(
            source_root=source,
            runs_root=runs,
            expected_commit=commit,
            expected_image_digest="rex:latest",
        )
