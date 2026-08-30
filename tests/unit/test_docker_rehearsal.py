from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "rex_run_docker_rehearsal",
    ROOT / "scripts/run_docker_rehearsal.py",
)
assert SPEC is not None and SPEC.loader is not None
REHEARSAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REHEARSAL)


def test_runs_artifact_path_maps_to_the_host_output_root(tmp_path: Path) -> None:
    output = tmp_path / "run-output"
    artifact = output / "run-id/best-valid/model/model.npz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"model")

    resolved = REHEARSAL._runs_path_on_host(
        "/runs/run-id/best-valid/model/model.npz",
        output,
    )

    assert resolved == artifact


@pytest.mark.parametrize(
    "container_path",
    (
        "/etc/passwd",
        "/runs",
        "/runs/../outside/model.npz",
        "run-id/best-valid/model/model.npz",
    ),
)
def test_runs_artifact_path_rejects_unmounted_or_escaping_paths(
    tmp_path: Path,
    container_path: str,
) -> None:
    output = tmp_path / "run-output"
    output.mkdir()

    with pytest.raises(REHEARSAL.DockerRehearsalError):
        REHEARSAL._runs_path_on_host(container_path, output)


def test_runs_artifact_path_rejects_a_symlink_escape(tmp_path: Path) -> None:
    output = tmp_path / "run-output"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model.npz").write_bytes(b"outside")
    output.mkdir()
    (output / "run-id").symlink_to(outside, target_is_directory=True)

    with pytest.raises(REHEARSAL.DockerRehearsalError):
        REHEARSAL._runs_path_on_host("/runs/run-id/model.npz", output)
