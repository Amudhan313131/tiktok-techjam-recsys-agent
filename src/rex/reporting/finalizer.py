"""Atomic final bundles with source and copied-artifact re-verification."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

from rex.contracts import ArtifactRef, Metrics
from rex.data.manifest import sha256_file
from rex.data.views import FeatureView, load_feature_view
from rex.execution.artifacts import artifact_ref, atomic_write_json
from rex.evaluation.submission import (
    TEST_ROW_COUNT,
    SubmissionValidation,
    require_valid_submission,
    validate_submission_matches_predictions,
)
from rex.execution.sandbox import SandboxMode
from rex.models.bundle import LoadedModelBundle, validate_model_bundle


def _verify(ref: ArtifactRef) -> Path:
    path = Path(ref.path).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != ref.size_bytes
        or sha256_file(path) != ref.sha256
    ):
        raise RuntimeError(f"artifact drifted or is missing: {ref.artifact_id} ({path})")
    return path


def _verify_config(config_path: str | Path, expected_sha256: str) -> Path:
    config = Path(config_path).resolve()
    if not config.is_file() or sha256_file(config) != expected_sha256:
        raise RuntimeError("configuration drifted or is missing")
    return config


def _staging_directory(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing to replace an existing sealed bundle: {output}")
    return Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    ).resolve()


def _sealed_ref(staged: Path, sealed: Path, kind: str) -> ArtifactRef:
    ref = artifact_ref(staged, kind)
    return ref.model_copy(update={"path": str(sealed.resolve())})


def _copy_artifact(
    source: Path,
    stage_root: Path,
    output_root: Path,
    relative: str,
    kind: str,
) -> tuple[Path, ArtifactRef]:
    destination = stage_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(destination) != sha256_file(source):
        raise RuntimeError(f"artifact changed while being copied: {source}")
    return destination, _sealed_ref(destination, output_root / relative, kind)


def _regular_tree_files(root: Path) -> dict[str, Path]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"finalization source tree is missing or unsafe: {root}")
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"finalization source tree contains a symlink: {path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path
    return files


def _copy_file_atomically(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"finalization source is missing or unsafe: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".copying")
    try:
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != sha256_file(source):
            raise RuntimeError(f"artifact changed while being staged: {source}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def stage_submission_bundle(
    output_dir: str | Path,
    *,
    best_valid_dir: str | Path,
    source_report_dir: str | Path,
    predictions_path: str | Path,
    submission_path: str | Path,
    first_check_transcript: str | Path,
    expected_commit_sha: str,
    expected_config_sha256: str,
) -> Path:
    """Crash-resumably stage the exact tree consumed by final submission sealing.

    This is the shared production boundary used by the submission coordinator.
    Files are copied one at a time through atomic replacement, so an interruption
    leaves a replayable partial staging tree rather than a falsely sealed bundle.
    Unexpected files and symlinks fail closed.
    """

    raw_paths = {
        "output": Path(output_dir),
        "best-valid": Path(best_valid_dir),
        "source-report": Path(source_report_dir),
        "predictions": Path(predictions_path),
        "submission": Path(submission_path),
        "first-check": Path(first_check_transcript),
    }
    for name, path in raw_paths.items():
        if path.is_symlink():
            raise RuntimeError(f"finalization {name} path may not be a symlink: {path}")
    output = raw_paths["output"].resolve()
    best_valid = raw_paths["best-valid"].resolve()
    source_report = raw_paths["source-report"].resolve()
    predictions = raw_paths["predictions"].resolve()
    submission = raw_paths["submission"].resolve()
    first_check = raw_paths["first-check"].resolve()
    validate_model_bundle(
        best_valid / "model/model_bundle.json",
        expected_commit_sha=expected_commit_sha,
        expected_config_sha256=expected_config_sha256,
    )

    sources: dict[str, Path] = {}
    for prefix, root in (("best-valid", best_valid), ("source-report", source_report)):
        for relative, source in _regular_tree_files(root).items():
            sources[f"{prefix}/{relative}"] = source
    sources.update(
        {
            "test_predictions.npz": predictions,
            "submission.csv": submission,
            "checks/first.json": first_check,
        }
    )
    for source in sources.values():
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"finalization source is missing or unsafe: {source}")

    output.mkdir(parents=True, exist_ok=True)
    expected_names = set(sources)
    for candidate in sorted(output.rglob("*")):
        if candidate.is_symlink():
            raise RuntimeError(f"submission staging contains a symlink: {candidate}")
        if candidate.is_file():
            relative = candidate.relative_to(output).as_posix()
            # A killed atomic copy may leave this deterministic scratch name.
            if relative.endswith(".copying"):
                candidate.unlink()
                continue
            if relative not in expected_names:
                raise RuntimeError(f"submission staging contains an unexpected file: {relative}")

    for relative, source in sources.items():
        destination = output / relative
        if destination.is_file() and not destination.is_symlink():
            if (
                destination.stat().st_size == source.stat().st_size
                and sha256_file(destination) == sha256_file(source)
            ):
                continue
        _copy_file_atomically(source, destination)

    for relative, source in sources.items():
        destination = output / relative
        if (
            not destination.is_file()
            or destination.is_symlink()
            or destination.stat().st_size != source.stat().st_size
            or sha256_file(destination) != sha256_file(source)
        ):
            raise RuntimeError(f"staged artifact differs from source: {relative}")
    validate_model_bundle(
        output / "best-valid/model/model_bundle.json",
        expected_commit_sha=expected_commit_sha,
        expected_config_sha256=expected_config_sha256,
    )
    return output


def _copy_model_bundle(
    loaded: LoadedModelBundle,
    stage_root: Path,
    output_root: Path,
) -> tuple[Path, dict[str, ArtifactRef]]:
    copied: dict[str, ArtifactRef] = {}
    source_root = loaded.manifest_path.parent.resolve()
    for source in (*loaded.member_paths, loaded.manifest_path):
        relative = source.relative_to(source_root).as_posix()
        name = f"model/{relative}"
        kind = "model_bundle" if source == loaded.manifest_path else "checkpoint"
        destination, ref = _copy_artifact(
            source,
            stage_root,
            output_root,
            name,
            kind,
        )
        copied[name] = ref
    copied_manifest = stage_root / "model/model_bundle.json"
    reloaded = validate_model_bundle(
        copied_manifest,
        expected_plugin=loaded.manifest.plugin,
        expected_config_sha256=loaded.manifest.config_sha256,
        expected_commit_sha=loaded.manifest.commit_sha,
        expected_data_view_sha256=loaded.manifest.data_view_sha256,
    )
    if reloaded.manifest != loaded.manifest:
        raise RuntimeError("copied model bundle manifest differs from its source")
    return copied_manifest, copied


def _validation_payload(result: SubmissionValidation) -> dict[str, object]:
    return {
        "valid": result.valid,
        "command": list(result.command),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "sandbox": result.sandbox_evidence,
        "score_mode_used": False,
        "make_mode_used": False,
    }


def _seal(stage: Path, output: Path) -> None:
    os.replace(stage, output)


def _validate_existing_seal(
    output: Path,
    manifest_name: str,
    *,
    kind: str,
    run_id: str,
    experiment_id: str,
    commit_sha: str,
    config_sha256: str,
) -> tuple[Path, dict[str, object]] | None:
    """Return a verified prior seal so crash recovery never overwrites it."""

    manifest_path = output / manifest_name
    if not output.exists():
        return None
    if output.is_symlink() or not output.is_dir() or not manifest_path.is_file():
        raise RuntimeError(f"existing bundle is not a complete immutable seal: {output}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"existing bundle manifest is invalid: {error}") from error
    expected = {
        "kind": kind,
        "run_id": run_id,
        "incumbent_experiment_id": experiment_id,
        "commit_sha": commit_sha,
        "config_sha256": config_sha256,
        "test_scored": False,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"existing sealed bundle has mismatched {name}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("existing sealed bundle has no artifact index")
    for name, raw_ref in artifacts.items():
        ref = ArtifactRef.model_validate(raw_ref)
        path = _verify(ref)
        try:
            path.relative_to(output)
        except ValueError as error:
            raise RuntimeError(f"sealed artifact escapes final bundle: {name}") from error
    validate_model_bundle(
        output / "model/model_bundle.json",
        expected_commit_sha=commit_sha,
        expected_config_sha256=config_sha256,
    )
    return manifest_path, payload


def create_best_valid_bundle(
    output_dir: str | Path,
    *,
    run_id: str,
    experiment_id: str,
    model_bundle: ArtifactRef,
    valid_predictions: ArtifactRef,
    evidence_index: ArtifactRef,
    metrics: Metrics,
    commit_sha: str,
    config_path: str | Path,
    config_sha256: str,
    additional_evidence: Iterable[ArtifactRef] = (),
) -> ArtifactRef:
    """Freeze the validation-best candidate without touching the test split."""

    if metrics.split != "valid":
        raise RuntimeError("best-valid metrics must come from the validation split")
    output = Path(output_dir).resolve()
    existing = _validate_existing_seal(
        output,
        "best_valid_manifest.json",
        kind="best_valid",
        run_id=run_id,
        experiment_id=experiment_id,
        commit_sha=commit_sha,
        config_sha256=config_sha256,
    )
    if existing is not None:
        return artifact_ref(existing[0], "best_valid_manifest")
    stage = _staging_directory(output)
    try:
        config = _verify_config(config_path, config_sha256)
        loaded = validate_model_bundle(
            _verify(model_bundle),
            expected_commit_sha=commit_sha,
            expected_config_sha256=config_sha256,
        )
        _copied_manifest, copied_model = _copy_model_bundle(loaded, stage, output)
        copied: dict[str, ArtifactRef] = dict(copied_model)

        for name, ref in (
            ("valid_predictions.npz", valid_predictions),
            ("evidence_index.json", evidence_index),
        ):
            source = _verify(ref)
            _destination, copied[name] = _copy_artifact(
                source, stage, output, name, ref.kind
            )
        config_name = "config" + config.suffix
        _destination, copied[config_name] = _copy_artifact(
            config,
            stage,
            output,
            config_name,
            "experiment_config",
        )
        for index, ref in enumerate(additional_evidence):
            source = _verify(ref)
            suffix = source.suffix or ".artifact"
            name = f"evidence/{index:02d}-{ref.kind}{suffix}"
            _destination, copied[name] = _copy_artifact(
                source,
                stage,
                output,
                name,
                ref.kind,
            )

        manifest = {
            "schema_version": "1.0",
            "kind": "best_valid",
            "test_prediction_created": False,
            "test_scored": False,
            "run_id": run_id,
            "incumbent_experiment_id": experiment_id,
            "commit_sha": commit_sha,
            "config_sha256": config_sha256,
            "validation_metrics": metrics.model_dump(mode="json", by_alias=True),
            # Retained for readers created before validation_metrics was named explicitly.
            "metrics": metrics.model_dump(mode="json", by_alias=True),
            "model_bundle_sha256": copied["model/model_bundle.json"].sha256,
            "artifacts": {
                name: ref.model_dump(mode="json") for name, ref in sorted(copied.items())
            },
        }
        manifest_path = stage / "best_valid_manifest.json"
        atomic_write_json(manifest_path, manifest)
        _seal(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    # Revalidate from the final path, not the temporary staging namespace.
    validate_model_bundle(
        output / "model/model_bundle.json",
        expected_commit_sha=commit_sha,
        expected_config_sha256=config_sha256,
    )
    return artifact_ref(output / "best_valid_manifest.json", "best_valid_manifest")


def create_final_bundle(
    output_dir: str | Path,
    *,
    run_id: str,
    experiment_id: str,
    submission: ArtifactRef,
    model_bundle: ArtifactRef,
    predictions: ArtifactRef,
    config_path: str | Path,
    report: ArtifactRef,
    evidence_index: ArtifactRef,
    metrics: Metrics,
    commit_sha: str,
    config_sha256: str,
    expected_test_features: FeatureView | str | Path,
    data_dir: str | Path,
    expected_rows: int = TEST_ROW_COUNT,
    additional_evidence: Iterable[ArtifactRef] = (),
    checker_sandbox_mode: SandboxMode = SandboxMode.PRODUCTION,
) -> ArtifactRef:
    """Check, copy, re-check, and atomically seal the one final test bundle."""

    if metrics.split != "valid":
        raise RuntimeError("final bundle may contain validation metrics only")
    view = (
        expected_test_features
        if isinstance(expected_test_features, FeatureView)
        else load_feature_view(expected_test_features)
    )
    if view.rows != expected_rows:
        raise RuntimeError(
            f"canonical test row count mismatch: expected {expected_rows}, observed {view.rows}"
        )
    submission_path = _verify(submission)
    prediction_path = _verify(predictions)
    config = _verify_config(config_path, config_sha256)
    report_path = _verify(report)
    evidence_path = _verify(evidence_index)
    loaded = validate_model_bundle(
        _verify(model_bundle),
        expected_commit_sha=commit_sha,
        expected_config_sha256=config_sha256,
        expected_features=view,
    )
    validate_submission_matches_predictions(
        submission_path,
        prediction_path,
        expected_features=view,
        expected_rows=expected_rows,
    )
    output = Path(output_dir).resolve()
    existing = _validate_existing_seal(
        output,
        "bundle_manifest.json",
        kind="final_submission",
        run_id=run_id,
        experiment_id=experiment_id,
        commit_sha=commit_sha,
        config_sha256=config_sha256,
    )
    if existing is not None:
        payload = existing[1]
        if payload.get("test_rows") != expected_rows:
            raise RuntimeError("existing sealed bundle has mismatched test_rows")
        if payload.get("test_feature_sha256") != view.sha256:
            raise RuntimeError("existing sealed bundle has mismatched test feature view")
        return artifact_ref(existing[0], "bundle_manifest")
    source_check = require_valid_submission(
        submission_path,
        data_dir=data_dir,
        split="test",
        sandbox_mode=checker_sandbox_mode,
    )

    stage = _staging_directory(output)
    try:
        _copied_manifest, copied_model = _copy_model_bundle(loaded, stage, output)
        copied: dict[str, ArtifactRef] = dict(copied_model)
        sources = (
            ("submission.csv", submission_path, submission.kind),
            ("predictions.npz", prediction_path, predictions.kind),
            ("report.json", report_path, report.kind),
            ("evidence_index.json", evidence_path, evidence_index.kind),
        )
        for name, source, kind in sources:
            _destination, copied[name] = _copy_artifact(
                source, stage, output, name, kind
            )
        config_name = "config" + config.suffix
        _destination, copied[config_name] = _copy_artifact(
            config, stage, output, config_name, "experiment_config"
        )
        for index, ref in enumerate(additional_evidence):
            source = _verify(ref)
            suffix = source.suffix or ".artifact"
            name = f"evidence/{index:02d}-{ref.kind}{suffix}"
            _destination, copied[name] = _copy_artifact(
                source, stage, output, name, ref.kind
            )

        source_check_path = stage / "checks/source.json"
        atomic_write_json(source_check_path, _validation_payload(source_check))
        copied["checks/source.json"] = _sealed_ref(
            source_check_path, output / "checks/source.json", "submission_check"
        )

        validate_submission_matches_predictions(
            stage / "submission.csv",
            stage / "predictions.npz",
            expected_features=view,
            expected_rows=expected_rows,
        )
        copied_check = require_valid_submission(
            stage / "submission.csv",
            data_dir=data_dir,
            split="test",
            sandbox_mode=checker_sandbox_mode,
        )
        copied_check_path = stage / "checks/copied.json"
        atomic_write_json(copied_check_path, _validation_payload(copied_check))
        copied["checks/copied.json"] = _sealed_ref(
            copied_check_path, output / "checks/copied.json", "submission_check"
        )

        manifest = {
            "schema_version": "1.0",
            "kind": "final_submission",
            "sealed": True,
            "run_id": run_id,
            "incumbent_experiment_id": experiment_id,
            "commit_sha": commit_sha,
            "config_sha256": config_sha256,
            "test_rows": expected_rows,
            "test_feature_sha256": view.sha256,
            "test_prediction_created": True,
            "test_scored": False,
            "organizer_checks": 2,
            "validation_metrics": metrics.model_dump(mode="json", by_alias=True),
            "primary_checkpoint": f"model/{loaded.manifest.primary_member}",
            "model_bundle_sha256": copied["model/model_bundle.json"].sha256,
            "artifacts": {
                name: ref.model_dump(mode="json") for name, ref in sorted(copied.items())
            },
        }
        manifest_path = stage / "bundle_manifest.json"
        atomic_write_json(manifest_path, manifest)
        _seal(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    validate_model_bundle(
        output / "model/model_bundle.json",
        expected_commit_sha=commit_sha,
        expected_config_sha256=config_sha256,
        expected_features=view,
    )
    sealed_manifest = output / "bundle_manifest.json"
    payload = json.loads(sealed_manifest.read_text(encoding="utf-8"))
    for name, raw_ref in payload["artifacts"].items():
        ref = ArtifactRef.model_validate(raw_ref)
        path = _verify(ref)
        try:
            path.relative_to(output)
        except ValueError as error:  # pragma: no cover - defensive invariant
            raise RuntimeError(f"sealed artifact escapes final bundle: {name}") from error
    return artifact_ref(sealed_manifest, "bundle_manifest")
