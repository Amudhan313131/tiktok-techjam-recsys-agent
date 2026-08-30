"""Deterministic materialization of temporal shadow and cheap-rung views."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rex.data.groups import sample_complete_users
from rex.data.manifest import canonical_json_bytes, sha256_bytes, sha256_file
from rex.data.temporal import DEFAULT_SHADOW_FOLDS, ShadowFold, validate_shadow_folds
from rex.data.views import FeatureView, TargetView, load_feature_view, load_target_view


@dataclass(frozen=True)
class MaterializedFold:
    name: str
    root: Path
    train_features: Path
    train_targets: Path
    valid_features: Path
    valid_targets: Path
    manifest: Path
    identity_sha256: str


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _subset_features(view: FeatureView, indices: np.ndarray) -> dict[str, np.ndarray]:
    arrays = {name: values[indices] for name, values in view.arrays.items()}
    source = np.asarray(arrays.get("fx__source_row_id", arrays["row_id"]), dtype=np.int64)
    arrays["row_id"] = np.arange(len(indices), dtype=np.int64)
    arrays["fx__source_row_id"] = source
    return arrays


def _write_partition(
    root: Path,
    name: str,
    features: FeatureView,
    targets: TargetView,
    indices: np.ndarray,
) -> tuple[Path, Path, dict[str, Any]]:
    feature_path = root / f"{name}_features.npz"
    target_path = root / f"{name}_targets.npz"
    subset = _subset_features(features, indices)
    _atomic_npz(feature_path, subset)
    _atomic_npz(target_path, {"long_view": targets.labels[indices]})
    target_path.chmod(0o600)
    return feature_path, target_path, {
        "rows": int(len(indices)),
        "users": int(len(np.unique(features.arrays["user_id"][indices]))),
        "source_row_id_sha256": sha256_bytes(subset["fx__source_row_id"].tobytes()),
        "feature_sha256": sha256_file(feature_path),
        "target_sha256": sha256_file(target_path),
    }


def _materialized(root: Path, name: str, identity: str) -> MaterializedFold:
    return MaterializedFold(
        name=name,
        root=root,
        train_features=root / "train_features.npz",
        train_targets=root / "train_targets.npz",
        valid_features=root / "valid_features.npz",
        valid_targets=root / "valid_targets.npz",
        manifest=root / "manifest.json",
        identity_sha256=identity,
    )


def _cache_matches(candidate: MaterializedFold, manifest: dict[str, Any]) -> bool:
    pairs = (
        (candidate.train_features, manifest.get("train", {}).get("feature_sha256")),
        (candidate.train_targets, manifest.get("train", {}).get("target_sha256")),
        (candidate.valid_features, manifest.get("valid", {}).get("feature_sha256")),
        (candidate.valid_targets, manifest.get("valid", {}).get("target_sha256")),
    )
    return all(path.is_file() and expected == sha256_file(path) for path, expected in pairs)


def materialize_shadow_folds(
    feature_path: str | Path,
    target_path: str | Path,
    output_dir: str | Path,
    *,
    folds: Iterable[ShadowFold] = DEFAULT_SHADOW_FOLDS,
) -> tuple[MaterializedFold, ...]:
    """Create cached, reindexed train/valid views for each rolling fold."""

    features = load_feature_view(feature_path)
    targets = load_target_view(target_path)
    if features.rows != len(targets.labels):
        raise ValueError("shadow source feature/target lengths differ")
    fold_list = tuple(folds)
    validate_shadow_folds(features.arrays["date"], fold_list)
    output = Path(output_dir)
    results: list[MaterializedFold] = []
    for fold in fold_list:
        identity_value = {
            "schema_version": "1.0",
            "implementation_sha256": sha256_file(Path(__file__)),
            "source_feature_sha256": features.sha256,
            "source_target_sha256": targets.sha256,
            "fold": fold.__dict__,
        }
        identity = sha256_bytes(canonical_json_bytes(identity_value))
        root = output / f"fold-{fold.name}-{identity[:12]}"
        manifest_path = root / "manifest.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            candidate = _materialized(root, fold.name, identity)
            if existing.get("identity_sha256") == identity and _cache_matches(candidate, existing):
                results.append(candidate)
                continue
        root.mkdir(parents=True, exist_ok=True)
        train_mask, valid_mask = fold.masks(features.arrays["date"])
        train_indices = np.flatnonzero(train_mask)
        valid_indices = np.flatnonzero(valid_mask)
        train_feature, train_target, train_info = _write_partition(
            root, "train", features, targets, train_indices
        )
        valid_feature, valid_target, valid_info = _write_partition(
            root, "valid", features, targets, valid_indices
        )
        manifest = {
            **identity_value,
            "identity_sha256": identity,
            "train": train_info,
            "valid": valid_info,
        }
        _atomic_json(manifest_path, manifest)
        results.append(
            MaterializedFold(
                fold.name,
                root,
                train_feature,
                train_target,
                valid_feature,
                valid_target,
                manifest_path,
                identity,
            )
        )
    return tuple(results)


def materialize_cheap_view(
    fold: MaterializedFold,
    output_dir: str | Path,
    *,
    fraction: float,
    seed: int,
) -> MaterializedFold:
    """Select complete validation users and retain all of their fold history."""

    train = load_feature_view(fold.train_features)
    train_targets = load_target_view(fold.train_targets)
    valid = load_feature_view(fold.valid_features)
    valid_targets = load_target_view(fold.valid_targets)
    valid_indices = sample_complete_users(valid.arrays["user_id"], fraction=fraction, seed=seed)
    selected_users = np.unique(valid.arrays["user_id"][valid_indices])
    train_indices = np.flatnonzero(np.isin(train.arrays["user_id"], selected_users))
    identity_value = {
        "schema_version": "1.0",
        "implementation_sha256": sha256_file(Path(__file__)),
        "parent_identity_sha256": fold.identity_sha256,
        "fraction": fraction,
        "seed": seed,
        "selected_users_sha256": sha256_bytes(selected_users.tobytes()),
    }
    identity = sha256_bytes(canonical_json_bytes(identity_value))
    root = Path(output_dir) / f"cheap-{fold.name}-{identity[:12]}"
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidate = _materialized(root, fold.name, identity)
        if existing.get("identity_sha256") == identity and _cache_matches(candidate, existing):
            return candidate
    root.mkdir(parents=True, exist_ok=True)
    train_feature, train_target, train_info = _write_partition(
        root, "train", train, train_targets, train_indices
    )
    valid_feature, valid_target, valid_info = _write_partition(
        root, "valid", valid, valid_targets, valid_indices
    )
    manifest = {
        **identity_value,
        "identity_sha256": identity,
        "train": train_info,
        "valid": valid_info,
    }
    _atomic_json(manifest_path, manifest)
    return MaterializedFold(
        fold.name,
        root,
        train_feature,
        train_target,
        valid_feature,
        valid_target,
        manifest_path,
        identity,
    )
