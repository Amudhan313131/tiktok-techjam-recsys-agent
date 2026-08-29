"""Capability-scoped model worker. It never imports the official evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rex.contracts import ArtifactRef, AttemptStatus, RunRequest, RunResult
from rex.data.firewall import CapabilityViolation, validate_worker_request
from rex.data.manifest import canonical_json_bytes, sha256_file
from rex.data.views import load_feature_view, load_target_view
from rex.execution.artifacts import (
    ArtifactError,
    artifact_ref,
    atomic_write_json,
    write_prediction_artifact,
)
from rex.models.base import load_plugin
from rex.models.bundle import create_model_bundle, validate_model_bundle


def _load_config(path: Path) -> dict[str, Any]:
    if sha256_file(path) == "":  # pragma: no cover - keeps the hash read explicit
        raise ValueError("unreachable empty digest")
    with path.open(encoding="utf-8") as handle:
        if path.suffix in {".yaml", ".yml"}:
            value = yaml.safe_load(handle)
        else:
            value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("experiment config must be an object")
    return value


def _classify_exception(error: BaseException) -> AttemptStatus:
    if isinstance(error, SyntaxError):
        return AttemptStatus.SYNTAX
    if isinstance(error, (ImportError, ModuleNotFoundError)):
        return AttemptStatus.IMPORT
    if isinstance(error, CapabilityViolation):
        return AttemptStatus.CONTRACT
    if isinstance(error, MemoryError):
        return AttemptStatus.OOM
    if isinstance(error, FloatingPointError):
        return AttemptStatus.NAN
    if isinstance(error, ArtifactError):
        detail = str(error).lower()
        if "nan" in detail or "non-finite" in detail or "nonfinite" in detail:
            return AttemptStatus.NAN
        return AttemptStatus.INVALID_ARTIFACT
    if isinstance(error, (ValueError, TypeError, KeyError)):
        return AttemptStatus.CONTRACT
    return AttemptStatus.CRASH


def execute(request: RunRequest) -> RunResult:
    started = time.monotonic()
    command_sha = hashlib.sha256(canonical_json_bytes(request.model_dump(mode="json"))).hexdigest()
    artifacts: list[ArtifactRef] = []
    try:
        if int(time.time() * 1000) >= request.deadline_epoch_ms:
            raise TimeoutError("request deadline already expired")
        validate_worker_request(request)
        config_path = Path(request.config_path)
        observed_config_hash = sha256_file(config_path)
        if observed_config_hash != request.config_sha256:
            raise CapabilityViolation(
                f"config hash mismatch: expected {request.config_sha256}, observed {observed_config_hash}"
            )
        config = _load_config(config_path)
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plugin = load_plugin(request.plugin)
        features = load_feature_view(request.feature_view_path)

        if request.effective_operation == "predict":
            if request.model_bundle_path is not None:
                bundle = validate_model_bundle(
                    request.model_bundle_path,
                    expected_plugin=request.plugin,
                    expected_config_sha256=request.config_sha256,
                    expected_commit_sha=request.commit_sha,
                    expected_features=features,
                )
                model_path = bundle.primary_path
            else:
                # Compatibility for runs created before model_bundle_path existed.
                try:
                    model_path = Path(str(config["model_artifact_path"]))
                except KeyError as error:
                    raise ArtifactError("prediction request is missing a model bundle") from error
            try:
                scores = np.asarray(
                    plugin.predict(model_path, features, config, output_dir), dtype=np.float64
                )
            except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
                raise ArtifactError(f"model bundle could not be loaded: {error}") from error
            prediction_path = write_prediction_artifact(output_dir / "predictions.npz", features, scores)
            artifacts.append(artifact_ref(prediction_path, "predictions"))
        else:
            if request.target_view_path is None:
                raise CapabilityViolation("training request is missing target capability")
            targets = load_target_view(request.target_view_path)
            if features.rows != len(targets.labels):
                raise ValueError("training feature/target rows differ")
            model_path = Path(plugin.fit(features, targets, config, request.seed, output_dir))
            bundle_path = create_model_bundle(
                output_dir,
                model_path,
                plugin=request.plugin,
                seed=request.seed,
                commit_sha=request.commit_sha,
                config_sha256=request.config_sha256,
                data_view_sha256=request.data_view_sha256,
                features=features,
            )
            bundle = validate_model_bundle(bundle_path)
            member_by_name = {member.name: member for member in bundle.manifest.members}
            for member_path in bundle.member_paths:
                member = member_by_name[member_path.relative_to(bundle_path.parent).as_posix()]
                artifacts.append(artifact_ref(member_path, member.kind))
            artifacts.append(artifact_ref(bundle_path, "model_bundle"))

        return RunResult(
            run_id=request.run_id,
            experiment_id=request.experiment_id,
            attempt_id=request.attempt_id,
            status=AttemptStatus.SUCCESS,
            exit_code=0,
            command_sha256=command_sha,
            commit_sha=request.commit_sha,
            config_sha256=request.config_sha256,
            data_view_sha256=request.data_view_sha256,
            environment_sha256=request.environment_sha256,
            artifacts=artifacts,
            wall_seconds=time.monotonic() - started,
        )
    except TimeoutError as error:
        return RunResult(
            run_id=request.run_id,
            experiment_id=request.experiment_id,
            attempt_id=request.attempt_id,
            status=AttemptStatus.TIMEOUT,
            exit_code=1,
            error_type=type(error).__name__,
            error_summary=str(error),
            command_sha256=command_sha,
            commit_sha=request.commit_sha,
            config_sha256=request.config_sha256,
            data_view_sha256=request.data_view_sha256,
            environment_sha256=request.environment_sha256,
            wall_seconds=time.monotonic() - started,
        )
    except BaseException as error:  # worker must always emit a typed result
        traceback.print_exc(file=sys.stderr)
        return RunResult(
            run_id=request.run_id,
            experiment_id=request.experiment_id,
            attempt_id=request.attempt_id,
            status=_classify_exception(error),
            exit_code=1,
            error_type=type(error).__name__,
            error_summary=str(error)[-2000:],
            command_sha256=command_sha,
            commit_sha=request.commit_sha,
            config_sha256=request.config_sha256,
            data_view_sha256=request.data_view_sha256,
            environment_sha256=request.environment_sha256,
            wall_seconds=time.monotonic() - started,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    request = RunRequest.model_validate_json(Path(args.request).read_text(encoding="utf-8"))
    result = execute(request)
    atomic_write_json(args.result, result.model_dump(mode="json", by_alias=True))
    return 0 if result.status == AttemptStatus.SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
