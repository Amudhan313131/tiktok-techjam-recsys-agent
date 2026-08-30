"""Capability checks preventing outcome-label access from inference workers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rex.contracts import RunRequest
from rex.data.manifest import load_benchmark_manifest
from rex.data.views import ENGINEERED_PREFIX, FEATURE_COLUMNS, load_feature_view


class CapabilityViolation(PermissionError):
    pass


@dataclass(frozen=True)
class CapabilityRoots:
    """Trusted roots granted to a worker for one operation."""

    features: Path
    targets: Path | None = None


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_regular_file(path: str | Path, *, kind: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise CapabilityViolation(f"{kind} capability may not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise CapabilityViolation(f"{kind} capability is not a regular file: {resolved}")
    return resolved


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
    _resolve_regular_file(path, kind="feature")
    view = load_feature_view(path)
    assert_sanitized_feature_view(view.arrays)


def validate_worker_request(
    request: RunRequest,
    *,
    roots: CapabilityRoots | None = None,
) -> None:
    validate_sanitized_feature_view(request.feature_view_path)
    operation = request.effective_operation
    if request.split == "test" and (operation != "predict" or request.rung != "predict"):
        raise CapabilityViolation("test capabilities are restricted to prediction requests")
    if operation == "predict" and request.target_view_path is not None:
        raise CapabilityViolation("prediction request received a target capability")
    if operation == "fit" and request.split not in {"train", "shadow"}:
        raise CapabilityViolation(f"fit request cannot use split {request.split!r}")
    if request.target_view_path:
        target = _resolve_regular_file(request.target_view_path, kind="target")
        features = _resolve_regular_file(request.feature_view_path, kind="feature")
        if target == features:
            raise CapabilityViolation("target and feature capabilities must be separate files")
        if target.name == "test_targets.npz" or "test_target" in target.name.lower():
            raise CapabilityViolation("test targets are never an authorized capability")
    if roots is not None:
        feature = _resolve_regular_file(request.feature_view_path, kind="feature")
        feature_root = roots.features.resolve(strict=True)
        if not _is_within(feature, feature_root):
            raise CapabilityViolation(f"feature capability escapes trusted root: {feature}")
        if request.target_view_path:
            if roots.targets is None:
                raise CapabilityViolation("no target root was granted")
            target = _resolve_regular_file(request.target_view_path, kind="target")
            target_root = roots.targets.resolve(strict=True)
            if not _is_within(target, target_root):
                raise CapabilityViolation(f"target capability escapes trusted root: {target}")


def assert_no_test_target_artifact(view_root: str | Path) -> None:
    """Fail closed if any generated artifact resembles a hidden-test target."""

    root = Path(view_root)
    offenders = sorted(
        str(path)
        for path in root.rglob("*")
        if path.is_file() and "test" in path.name.lower() and "target" in path.name.lower()
    )
    if offenders:
        raise CapabilityViolation(f"forbidden test target artifacts: {offenders}")


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
