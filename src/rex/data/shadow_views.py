"""Deterministic materialization of temporal shadow and cheap-rung views."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

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
    sample_row_ids: Path | None = None


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
    return (
        feature_path,
        target_path,
        {
            "rows": int(len(indices)),
            "users": int(len(np.unique(features.arrays["user_id"][indices]))),
            "source_row_id_sha256": sha256_bytes(subset["fx__source_row_id"].tobytes()),
            "feature_sha256": sha256_file(feature_path),
            "target_sha256": sha256_file(target_path),
        },
    )


def _materialized(root: Path, name: str, identity: str) -> MaterializedFold:
    sample_rows = root / "sample_row_ids.npz"
    return MaterializedFold(
        name=name,
        root=root,
        train_features=root / "train_features.npz",
        train_targets=root / "train_targets.npz",
        valid_features=root / "valid_features.npz",
        valid_targets=root / "valid_targets.npz",
        manifest=root / "manifest.json",
        identity_sha256=identity,
        sample_row_ids=sample_rows if sample_rows.is_file() else None,
    )


def _cache_matches(candidate: MaterializedFold, manifest: dict[str, Any]) -> bool:
    pairs = (
        (candidate.train_features, manifest.get("train", {}).get("feature_sha256")),
        (candidate.train_targets, manifest.get("train", {}).get("target_sha256")),
        (candidate.valid_features, manifest.get("valid", {}).get("feature_sha256")),
        (candidate.valid_targets, manifest.get("valid", {}).get("target_sha256")),
    )
    if not all(path.is_file() and expected == sha256_file(path) for path, expected in pairs):
        return False
    sample = manifest.get("sample_row_ids")
    if sample is None:
        return True
    path = candidate.root / "sample_row_ids.npz"
    return path.is_file() and sample.get("sha256") == sha256_file(path)


def _history_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count <= 4:
        return "1-4"
    if count <= 19:
        return "5-19"
    return "20+"


def _stratified_complete_user_indices(
    train: FeatureView,
    valid: FeatureView,
    *,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Select complete users across train-known behavioral/context strata."""

    valid_users = np.asarray(valid.arrays["user_id"])
    unique_users = sorted(set(map(str, valid_users)))
    if not unique_users:
        return np.empty(0, dtype=np.int64), {}
    train_users = np.asarray([str(value) for value in train.arrays["user_id"]])
    train_videos = np.asarray([str(value) for value in train.arrays["video_id"]])
    counts: dict[str, int] = defaultdict(int)
    prior_pairs: set[tuple[str, str]] = set()
    for user, video in zip(train_users, train_videos, strict=True):
        counts[user] += 1
        prior_pairs.add((user, video))
    durations = np.asarray(train.arrays["duration_ms"], dtype=np.float64)
    edges = np.quantile(durations, [0.25, 0.5, 0.75]) if len(durations) else np.zeros(3)
    valid_video = np.asarray([str(value) for value in valid.arrays["video_id"]])
    valid_tab = np.asarray([str(value) for value in valid.arrays["tab"]])
    valid_duration = np.asarray(valid.arrays["duration_ms"], dtype=np.float64)
    by_user: dict[str, list[int]] = defaultdict(list)
    for index, user in enumerate(map(str, valid_users)):
        by_user[user].append(index)
    strata: dict[str, list[str]] = defaultdict(list)
    for user in unique_users:
        indices = np.asarray(by_user[user], dtype=np.int64)
        tabs, tab_counts = np.unique(valid_tab[indices], return_counts=True)
        dominant_tab = str(tabs[int(np.argmax(tab_counts))])
        duration_bucket = int(
            np.searchsorted(edges, np.median(valid_duration[indices]), side="right")
        )
        repeated = any((user, valid_video[index]) in prior_pairs for index in indices)
        key = "|".join(
            (
                f"history:{_history_bucket(counts[user])}",
                f"tab:{dominant_tab}",
                f"duration:q{duration_bucket + 1}",
                f"repeat:{int(repeated)}",
            )
        )
        strata[key].append(user)

    target = min(len(unique_users), max(1, int(round(len(unique_users) * fraction))))
    allocation = {key: int(np.floor(len(users) * fraction)) for key, users in strata.items()}
    remaining = target - sum(allocation.values())
    remainder_order = sorted(
        strata,
        key=lambda key: (-(len(strata[key]) * fraction - allocation[key]), key),
    )
    for key in remainder_order:
        if remaining <= 0:
            break
        if allocation[key] < len(strata[key]):
            allocation[key] += 1
            remaining -= 1
    selected: set[str] = set()
    selected_counts: dict[str, int] = {}
    for key in sorted(strata):
        ordered = sorted(
            strata[key],
            key=lambda user: sha256_bytes(f"{seed}\0{user}".encode("utf-8")),
        )
        chosen = ordered[: allocation[key]]
        selected.update(chosen)
        selected_counts[key] = len(chosen)
    indices = np.flatnonzero(
        np.isin(np.asarray([str(value) for value in valid_users]), list(selected))
    )
    return indices.astype(np.int64, copy=False), selected_counts


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
    valid_indices, stratum_counts = _stratified_complete_user_indices(
        train, valid, fraction=fraction, seed=seed
    )
    selected_users = np.unique(valid.arrays["user_id"][valid_indices])
    train_indices = np.flatnonzero(np.isin(train.arrays["user_id"], selected_users))
    valid_source_rows = np.asarray(
        valid.arrays.get("fx__source_row_id", valid.arrays["row_id"])[valid_indices],
        dtype=np.int64,
    )
    train_source_rows = np.asarray(
        train.arrays.get("fx__source_row_id", train.arrays["row_id"])[train_indices],
        dtype=np.int64,
    )
    identity_value = {
        "schema_version": "2.0",
        "implementation_sha256": sha256_file(Path(__file__)),
        "parent_identity_sha256": fold.identity_sha256,
        "fraction": fraction,
        "seed": seed,
        "selected_users_sha256": sha256_bytes(selected_users.tobytes()),
        "valid_source_row_ids_sha256": sha256_bytes(valid_source_rows.tobytes()),
        "train_source_row_ids_sha256": sha256_bytes(train_source_rows.tobytes()),
        "stratification": {
            "dimensions": ["train_history", "dominant_tab", "duration_quartile", "repeat"],
            "selected_counts": stratum_counts,
        },
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
    sample_row_ids = root / "sample_row_ids.npz"
    _atomic_npz(
        sample_row_ids,
        {
            "train_source_row_id": train_source_rows,
            "valid_source_row_id": valid_source_rows,
        },
    )
    manifest = {
        **identity_value,
        "identity_sha256": identity,
        "train": train_info,
        "valid": valid_info,
        "sample_row_ids": {
            "path": sample_row_ids.name,
            "sha256": sha256_file(sample_row_ids),
            "train_rows": int(len(train_source_rows)),
            "valid_rows": int(len(valid_source_rows)),
        },
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
        sample_row_ids,
    )
