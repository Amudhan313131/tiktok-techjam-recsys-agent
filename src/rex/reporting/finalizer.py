"""Final bundle creation with artifact re-verification and immutable manifest."""

from __future__ import annotations

import shutil
from pathlib import Path

from rex.contracts import ArtifactRef, Metrics
from rex.data.manifest import sha256_file
from rex.execution.artifacts import artifact_ref, atomic_write_json
from rex.evaluation.submission import require_valid_submission


def _verify(ref: ArtifactRef) -> Path:
    path = Path(ref.path)
    if not path.is_file() or sha256_file(path) != ref.sha256:
        raise RuntimeError(f"artifact drifted or is missing: {ref.artifact_id} ({path})")
    return path


def create_final_bundle(
    output_dir: str | Path,
    *,
    run_id: str,
    experiment_id: str,
    submission: ArtifactRef,
    checkpoint: ArtifactRef,
    predictions: ArtifactRef,
    evidence_index: ArtifactRef,
    metrics: Metrics,
    commit_sha: str,
    config_sha256: str,
    data_dir: str | Path | None = None,
) -> ArtifactRef:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    copied: dict[str, dict[str, object]] = {}
    if data_dir is not None:
        require_valid_submission(submission.path, data_dir=data_dir, split="test")
    for name, ref in (
        ("submission.csv", submission),
        ("checkpoint" + Path(checkpoint.path).suffix, checkpoint),
        ("predictions.npz", predictions),
        ("evidence_index.json", evidence_index),
    ):
        source = _verify(ref)
        destination = output / name
        shutil.copy2(source, destination)
        copied[name] = artifact_ref(destination, ref.kind).model_dump(mode="json")
    if data_dir is not None:
        require_valid_submission(output / "submission.csv", data_dir=data_dir, split="test")
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "incumbent_experiment_id": experiment_id,
        "commit_sha": commit_sha,
        "config_sha256": config_sha256,
        "metrics": metrics.model_dump(mode="json", by_alias=True),
        "artifacts": copied,
    }
    manifest_path = output / "bundle_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return artifact_ref(manifest_path, "bundle_manifest")
