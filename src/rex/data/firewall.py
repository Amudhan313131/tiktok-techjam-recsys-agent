"""Capability checks preventing outcome-label access from inference workers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rex.contracts import RunRequest
from rex.data.manifest import load_benchmark_manifest
from rex.data.views import ENGINEERED_PREFIX, FEATURE_COLUMNS, load_feature_view


class CapabilityViolation(PermissionError):
    pass


def assert_sanitized_feature_view(arrays: dict[str, np.ndarray]) -> None:
    forbidden = set(load_benchmark_manifest()["forbidden_inference_columns"])
    overlap = forbidden.intersection(arrays)
    if overlap:
        raise CapabilityViolation(f"feature view contains forbidden outcomes: {sorted(overlap)}")
    missing = set(FEATURE_COLUMNS) - set(arrays)
    invalid = {
        name
        for name in arrays
        if name not in FEATURE_COLUMNS and not name.startswith(ENGINEERED_PREFIX)
    }
    if missing or invalid:
        raise CapabilityViolation(
            f"feature view schema drifted: missing={sorted(missing)}, invalid={sorted(invalid)}"
        )


def validate_sanitized_feature_view(path: str | Path) -> None:
    view = load_feature_view(path)
    assert_sanitized_feature_view(view.arrays)


def validate_worker_request(request: RunRequest) -> None:
    validate_sanitized_feature_view(request.feature_view_path)
    if request.split in {"valid", "test"} and request.rung == "predict":
        if request.target_view_path is not None:
            raise CapabilityViolation("prediction request received validation/test targets")
    if request.target_view_path:
        target = Path(request.target_view_path).resolve()
        features = Path(request.feature_view_path).resolve()
        if target == features:
            raise CapabilityViolation("target and feature capabilities must be separate files")


def assert_outcome_poison_invariant(
    predictor,
    feature_path: str | Path,
    *,
    repeats: int = 2,
) -> None:
    """Test helper: sanitized views cannot expose a current-row outcome to mutate."""
    view = load_feature_view(feature_path)
    forbidden = set(load_benchmark_manifest()["forbidden_inference_columns"])
    if forbidden.intersection(view.arrays):
        raise CapabilityViolation("poison invariant failed: outcome present in feature view")
    reference = np.asarray(predictor(view.arrays))
    for _ in range(repeats):
        observed = np.asarray(predictor(view.arrays))
        if not np.array_equal(reference, observed):
            raise AssertionError("deterministic predictor changed without feature changes")
