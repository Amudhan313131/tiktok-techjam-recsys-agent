"""Portable, content-addressed model bundles shared by fit and prediction workers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from rex.contracts import ModelBundleManifest, ModelBundleMember
from rex.data.manifest import sha256_file
from rex.data.views import FeatureView
from rex.execution.artifacts import ArtifactError, atomic_write_json


BUNDLE_FILENAME = "model_bundle.json"


@dataclass(frozen=True)
class LoadedModelBundle:
    manifest_path: Path
    manifest: ModelBundleManifest
    primary_path: Path
    member_paths: tuple[Path, ...]


def feature_schema(view: FeatureView) -> dict[str, str]:
    """Return a width-independent schema that remains stable across data splits."""

    kinds = {
        "b": "bool",
        "i": "integer",
        "u": "integer",
        "f": "float",
        "c": "complex",
        "m": "timedelta",
        "M": "datetime",
        "O": "object",
        "S": "bytes",
        "U": "string",
        "V": "void",
    }
    return {
        name: f"{kinds.get(array.dtype.kind, array.dtype.kind)}[{array.ndim}]"
        for name, array in sorted(view.arrays.items())
    }


def _inside(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ArtifactError(f"model bundle member escapes bundle directory: {path}") from error


def create_model_bundle(
    bundle_dir: str | Path,
    primary_model: str | Path,
    *,
    plugin: str,
    seed: int,
    commit_sha: str,
    config_sha256: str,
    data_view_sha256: str,
    features: FeatureView,
    member_paths: Iterable[str | Path] | None = None,
) -> Path:
    """Index every plugin-produced file and atomically write its bundle manifest."""

    root = Path(bundle_dir).resolve()
    if not root.is_dir():
        raise ArtifactError(f"model bundle directory is missing: {root}")
    if not plugin.strip():
        raise ArtifactError("model bundle plugin must not be empty")
    if not commit_sha.strip():
        raise ArtifactError("model bundle commit must not be empty")
    primary = Path(primary_model).resolve()
    primary_name = _inside(root, primary).as_posix()
    if not primary.is_file():
        raise ArtifactError(f"primary model artifact is missing: {primary}")

    manifest_path = root / BUNDLE_FILENAME
    members: list[ModelBundleMember] = []
    candidates = (
        sorted((Path(item).resolve() for item in member_paths), key=str)
        if member_paths is not None
        else sorted(root.rglob("*"))
    )
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.is_file() or candidate == manifest_path or ".tmp" in candidate.suffixes:
            continue
        relative = _inside(root, candidate).as_posix()
        if relative in seen:
            continue
        seen.add(relative)
        members.append(
            ModelBundleMember(
                name=relative,
                kind="checkpoint" if relative == primary_name else "checkpoint_sidecar",
                sha256=sha256_file(candidate),
                size_bytes=candidate.stat().st_size,
            )
        )
    if primary_name not in seen:
        raise ArtifactError("primary model artifact must be an indexed bundle member")
    manifest = ModelBundleManifest(
        plugin=plugin,
        seed=seed,
        commit_sha=commit_sha,
        config_sha256=config_sha256,
        data_view_sha256=data_view_sha256,
        primary_member=primary_name,
        feature_schema=feature_schema(features),
        members=members,
    )
    atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
    validate_model_bundle(manifest_path)
    return manifest_path


def validate_model_bundle(
    bundle_path: str | Path,
    *,
    expected_plugin: str | None = None,
    expected_config_sha256: str | None = None,
    expected_commit_sha: str | None = None,
    expected_data_view_sha256: str | None = None,
    expected_features: FeatureView | None = None,
) -> LoadedModelBundle:
    """Load a bundle and fail closed on missing, corrupt, or incompatible members."""

    candidate = Path(bundle_path).resolve()
    manifest_path = candidate / BUNDLE_FILENAME if candidate.is_dir() else candidate
    if not manifest_path.is_file():
        raise ArtifactError(f"model bundle manifest is missing: {manifest_path}")
    try:
        manifest = ModelBundleManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValidationError, json.JSONDecodeError) as error:
        raise ArtifactError(f"invalid model bundle manifest: {error}") from error

    if expected_plugin is not None and manifest.plugin != expected_plugin:
        raise ArtifactError(
            f"model bundle plugin mismatch: expected {expected_plugin}, observed {manifest.plugin}"
        )
    if expected_config_sha256 is not None and manifest.config_sha256 != expected_config_sha256:
        raise ArtifactError("model bundle config hash mismatch")
    if expected_commit_sha is not None and manifest.commit_sha != expected_commit_sha:
        raise ArtifactError("model bundle commit mismatch")
    if (
        expected_data_view_sha256 is not None
        and manifest.data_view_sha256 != expected_data_view_sha256
    ):
        raise ArtifactError("model bundle data-view hash mismatch")
    if expected_features is not None and manifest.feature_schema != feature_schema(expected_features):
        raise ArtifactError("model bundle feature schema mismatch")

    root = manifest_path.parent.resolve()
    member_paths: list[Path] = []
    for member in manifest.members:
        path = (root / member.name).resolve()
        _inside(root, path)
        if not path.is_file():
            raise ArtifactError(f"model bundle member is missing: {member.name}")
        if path.stat().st_size != member.size_bytes or sha256_file(path) != member.sha256:
            raise ArtifactError(f"model bundle member is corrupt: {member.name}")
        member_paths.append(path)
    primary_path = (root / manifest.primary_member).resolve()
    _inside(root, primary_path)
    if primary_path not in member_paths:
        raise ArtifactError("model bundle primary member was not validated")
    return LoadedModelBundle(
        manifest_path=manifest_path,
        manifest=manifest,
        primary_path=primary_path,
        member_paths=tuple(member_paths),
    )
