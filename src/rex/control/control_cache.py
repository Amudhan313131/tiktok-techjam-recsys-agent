"""Immutable, content-addressed cache for reusable control predictions.

The cache is intentionally independent of production-run identifiers and filesystem
locations.  Workers never write it directly: the trusted coordinator publishes a
validated model bundle and aligned predictions, then imports a private copy into a
run before use.
"""

from __future__ import annotations

import fcntl
import os
import platform
import re
import shutil
import sys
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rex.data.manifest import canonical_json_bytes, sha256_bytes, sha256_file
from rex.data.views import load_feature_view, load_target_view
from rex.execution.artifacts import atomic_write_json, load_prediction_artifact
from rex.models.bundle import BUNDLE_FILENAME, LoadedModelBundle, validate_model_bundle


CACHE_SCHEMA_VERSION = "rex.control-prediction-cache.v1"
IDENTITY_SCHEMA_VERSION = "rex.control-prediction-cache.identity.v1"
IMPORT_SCHEMA_VERSION = "rex.control-prediction-cache.import.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROCESS_LOCKS: dict[str, threading.RLock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


class ControlCacheError(RuntimeError):
    """Base class for cache validation and publication failures."""


class ControlCacheContractError(ControlCacheError):
    """The caller attempted to construct an unsafe cache identity or input."""


class ControlCacheCorrupt(ControlCacheError):
    """A published entry is malformed, incomplete, or content-drifted."""


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ControlCacheIdentity(_FrozenStrictModel):
    """Path-independent identity of one fitted control and its predictions."""

    schema_version: Literal[IDENTITY_SCHEMA_VERSION] = IDENTITY_SCHEMA_VERSION
    plugin: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(min_length=7)
    environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    rung: Literal["cheap", "full", "official_valid"]
    split: Literal["shadow", "valid"]
    fold: str = Field(min_length=1)
    partition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    train_feature_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    train_target_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    apply_feature_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_provenance_sha256: tuple[str, ...] = ()

    @field_validator("feature_provenance_sha256")
    @classmethod
    def validate_provenance_hashes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(_SHA256.fullmatch(value) is None for value in values):
            raise ValueError("feature provenance contains a non-SHA-256 identity")
        if tuple(sorted(set(values))) != values:
            raise ValueError("feature provenance hashes must be sorted and unique")
        return values

    @model_validator(mode="after")
    def enforce_valid_only_contract(self) -> "ControlCacheIdentity":
        if self.rung == "official_valid" and self.split != "valid":
            raise ValueError("official-valid control entries require the valid split")
        if self.rung != "official_valid" and self.split != "shadow":
            raise ValueError("cheap/full control entries require the shadow split")
        return self

    @property
    def cache_key(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.model_dump(mode="json")))

    @classmethod
    def from_paths(
        cls,
        *,
        plugin: str,
        config_path: str | Path,
        source_commit: str,
        environment_sha256: str,
        seed: int,
        rung: Literal["cheap", "full", "official_valid"],
        split: Literal["shadow", "valid"],
        fold: str,
        partition_sha256: str,
        train_feature_path: str | Path,
        train_target_path: str | Path,
        apply_feature_path: str | Path,
        feature_provenance_paths: tuple[str | Path, ...] = (),
    ) -> "ControlCacheIdentity":
        """Hash path contents without allowing absolute locations into the identity."""

        provenance = tuple(sorted({sha256_file(Path(path)) for path in feature_provenance_paths}))
        return cls(
            plugin=plugin,
            config_sha256=sha256_file(Path(config_path)),
            source_commit=source_commit,
            environment_sha256=environment_sha256,
            seed=seed,
            rung=rung,
            split=split,
            fold=fold,
            partition_sha256=partition_sha256,
            train_feature_sha256=sha256_file(Path(train_feature_path)),
            train_target_sha256=sha256_file(Path(train_target_path)),
            apply_feature_sha256=sha256_file(Path(apply_feature_path)),
            feature_provenance_sha256=provenance,
        )


class ControlCacheProvenance(_FrozenStrictModel):
    producer_run_id: str = Field(min_length=1)
    producer_experiment_id: str = Field(min_length=1)
    fit_attempt_id: str = Field(min_length=1)
    predict_attempt_id: str = Field(min_length=1)
    fit_request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predict_request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ControlCacheConsumer(_FrozenStrictModel):
    run_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    rung: Literal["cheap", "full", "official_valid"]
    fold: str = Field(min_length=1)


class CachedControlArtifact(_FrozenStrictModel):
    path: str = Field(min_length=1)
    kind: Literal["model_bundle", "bundle_member", "predictions"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("cache artifact paths must use POSIX separators")
        candidate = PurePosixPath(value)
        if (
            candidate.is_absolute()
            or value != candidate.as_posix()
            or value in {"", "."}
            or ".." in candidate.parts
        ):
            raise ValueError(f"unsafe cache artifact path: {value}")
        return value


class ControlCacheManifest(_FrozenStrictModel):
    schema_version: Literal[CACHE_SCHEMA_VERSION] = CACHE_SCHEMA_VERSION
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity: ControlCacheIdentity
    provenance: ControlCacheProvenance
    artifacts: tuple[CachedControlArtifact, ...] = Field(min_length=3)
    created_at: str = Field(min_length=1)
    test_prediction_created: Literal[False] = False
    test_scored: Literal[False] = False

    @model_validator(mode="after")
    def validate_artifact_index(self) -> "ControlCacheManifest":
        paths = [artifact.path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("cache artifact paths must be unique")
        if self.cache_key != self.identity.cache_key:
            raise ValueError("cache manifest key does not match its identity")
        return self


class ControlCacheImportEvidence(_FrozenStrictModel):
    schema_version: Literal[IMPORT_SCHEMA_VERSION] = IMPORT_SCHEMA_VERSION
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    cache_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    producer: ControlCacheProvenance
    consumer: ControlCacheConsumer
    copied_artifacts: tuple[CachedControlArtifact, ...]
    imported_at: str = Field(min_length=1)
    test_prediction_created: Literal[False] = False
    test_scored: Literal[False] = False


@dataclass(frozen=True)
class ControlCacheEntry:
    cache_key: str
    root: Path
    manifest_path: Path
    bundle_path: Path
    prediction_path: Path
    manifest: ControlCacheManifest


@dataclass(frozen=True)
class ControlCacheImport(ControlCacheEntry):
    evidence_path: Path


def stable_environment_sha256(
    *,
    requirements_lock: str | Path,
    python_executable: str | Path = sys.executable,
    pyproject: str | Path | None = None,
    additional_components: Mapping[str, str] | None = None,
) -> str:
    """Return an environment identity that is independent of clone/venv paths."""

    lock = Path(requirements_lock)
    executable = Path(python_executable).resolve(strict=True)
    if not lock.is_file() or lock.is_symlink():
        raise ControlCacheContractError("requirements lock must be a regular non-symlink file")
    components = dict(additional_components or {})
    for name, digest in components.items():
        if not name or _SHA256.fullmatch(digest) is None:
            raise ControlCacheContractError(
                "additional environment components require SHA-256 values"
            )
    files: dict[str, str] = {"requirements_lock": sha256_file(lock)}
    if pyproject is not None:
        project = Path(pyproject)
        if not project.is_file() or project.is_symlink():
            raise ControlCacheContractError("pyproject must be a regular non-symlink file")
        files["pyproject"] = sha256_file(project)
    value: dict[str, Any] = {
        "schema_version": "rex.control-cache.environment.v1",
        "python": {
            "implementation": sys.implementation.name,
            "cache_tag": sys.implementation.cache_tag,
            "version": list(sys.version_info[:5]),
            "executable_sha256": sha256_file(executable),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "files": files,
        "additional_components": dict(sorted(components.items())),
    }
    return sha256_bytes(canonical_json_bytes(value))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ControlCacheContractError(f"{label} must be a regular non-symlink file")


def _artifact(path: Path, root: Path, kind: str) -> CachedControlArtifact:
    return CachedControlArtifact(
        path=path.relative_to(root).as_posix(),
        kind=kind,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


class ControlPredictionCache:
    """Publish and import immutable control model/prediction entries."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def entry_path(self, identity: ControlCacheIdentity) -> Path:
        key = identity.cache_key
        return self.root / "entries" / key[:2] / key

    def _lock_path(self, key: str) -> Path:
        return self.root / "locks" / key[:2] / f"{key}.lock"

    @contextmanager
    def _key_lock(self, key: str) -> Iterator[None]:
        lock_path = self._lock_path(key)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        process_key = str(lock_path)
        with _PROCESS_LOCKS_GUARD:
            process_lock = _PROCESS_LOCKS.setdefault(process_key, threading.RLock())
        with process_lock:
            flags = os.O_CREAT | os.O_RDWR
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(lock_path, flags, 0o600)
            except OSError as error:
                raise ControlCacheError(f"cannot open cache lock safely: {error}") from error
            with os.fdopen(descriptor, "a+b") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _validate_inputs(
        identity: ControlCacheIdentity,
        train_feature_path: str | Path,
        train_target_path: str | Path,
        apply_feature_path: str | Path,
    ) -> tuple[Path, Path, Path]:
        train_feature = Path(train_feature_path)
        train_target = Path(train_target_path)
        apply_feature = Path(apply_feature_path)
        for path, label in (
            (train_feature, "training feature view"),
            (train_target, "training target view"),
            (apply_feature, "apply feature view"),
        ):
            _regular_file(path, label=label)
        train = load_feature_view(train_feature)
        target = load_target_view(train_target)
        apply = load_feature_view(apply_feature)
        if train.rows != len(target.labels):
            raise ControlCacheContractError("training feature/target lengths differ")
        observed = {
            "train_feature_sha256": train.sha256,
            "train_target_sha256": target.sha256,
            "apply_feature_sha256": apply.sha256,
        }
        for field, digest in observed.items():
            if getattr(identity, field) != digest:
                raise ControlCacheContractError(f"control cache identity {field} mismatch")
        return train_feature, train_target, apply_feature

    @staticmethod
    def _tree(root: Path) -> tuple[set[str], set[str]]:
        if root.is_symlink() or not root.is_dir():
            raise ControlCacheCorrupt("cache entry root is not a regular directory")
        files: set[str] = set()
        directories: set[str] = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ControlCacheCorrupt(
                    f"cache entry contains a symlink: {path.relative_to(root)}"
                )
            relative = path.relative_to(root).as_posix()
            if path.is_file():
                files.add(relative)
            elif path.is_dir():
                directories.add(relative)
            else:
                raise ControlCacheCorrupt(f"cache entry contains a special file: {relative}")
        return files, directories

    def _validate_entry(
        self,
        entry_root: Path,
        identity: ControlCacheIdentity,
        train_feature_path: Path,
        apply_feature_path: Path,
    ) -> ControlCacheEntry:
        files, directories = self._tree(entry_root)
        manifest_path = entry_root / "manifest.json"
        if "manifest.json" not in files:
            raise ControlCacheCorrupt("cache manifest is missing")
        try:
            manifest = ControlCacheManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except Exception as error:
            raise ControlCacheCorrupt(f"invalid cache manifest: {error}") from error
        if manifest.identity != identity or manifest.cache_key != identity.cache_key:
            raise ControlCacheCorrupt("cache manifest identity mismatch")
        indexed = {artifact.path for artifact in manifest.artifacts}
        expected_files = {"manifest.json", *indexed}
        if files != expected_files:
            raise ControlCacheCorrupt(
                f"cache file set mismatch: missing={sorted(expected_files - files)}, "
                f"extra={sorted(files - expected_files)}"
            )
        expected_directories: set[str] = set()
        for relative in indexed:
            parent = PurePosixPath(relative).parent
            while parent.as_posix() != ".":
                expected_directories.add(parent.as_posix())
                parent = parent.parent
        if directories != expected_directories:
            raise ControlCacheCorrupt("cache directory set contains missing or extra entries")
        for indexed_artifact in manifest.artifacts:
            path = entry_root / indexed_artifact.path
            if (
                path.stat().st_size != indexed_artifact.size_bytes
                or sha256_file(path) != indexed_artifact.sha256
            ):
                raise ControlCacheCorrupt(f"cache artifact is corrupt: {indexed_artifact.path}")
        bundle_records = [item for item in manifest.artifacts if item.kind == "model_bundle"]
        prediction_records = [item for item in manifest.artifacts if item.kind == "predictions"]
        if (
            len(bundle_records) != 1
            or bundle_records[0].path != f"bundle/{BUNDLE_FILENAME}"
            or len(prediction_records) != 1
            or prediction_records[0].path != "predictions.npz"
        ):
            raise ControlCacheCorrupt("cache requires exactly one canonical bundle and prediction")
        train = load_feature_view(train_feature_path)
        try:
            bundle = validate_model_bundle(
                entry_root / bundle_records[0].path,
                expected_plugin=identity.plugin,
                expected_config_sha256=identity.config_sha256,
                expected_commit_sha=identity.source_commit,
                expected_data_view_sha256=identity.train_feature_sha256,
                expected_features=train,
            )
        except Exception as error:
            raise ControlCacheCorrupt(f"cached model bundle is invalid: {error}") from error
        if bundle.manifest.seed != identity.seed:
            raise ControlCacheCorrupt("cached model bundle seed mismatch")
        expected_members = {f"bundle/{member.name}" for member in bundle.manifest.members}
        indexed_members = {item.path for item in manifest.artifacts if item.kind == "bundle_member"}
        if indexed_members != expected_members:
            raise ControlCacheCorrupt("cache bundle-member index mismatch")
        try:
            load_prediction_artifact(entry_root / "predictions.npz", apply_feature_path)
        except Exception as error:
            raise ControlCacheCorrupt(f"cached predictions are invalid: {error}") from error
        return ControlCacheEntry(
            cache_key=identity.cache_key,
            root=entry_root,
            manifest_path=manifest_path,
            bundle_path=entry_root / "bundle" / BUNDLE_FILENAME,
            prediction_path=entry_root / "predictions.npz",
            manifest=manifest,
        )

    def _quarantine(self, path: Path, key: str, error: BaseException) -> Path:
        quarantine_root = self.root / "quarantine"
        quarantine_root.mkdir(parents=True, exist_ok=True)
        suffix = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex}"
        destination = quarantine_root / f"{key}-{suffix}"
        os.replace(path, destination)
        _fsync_directory(quarantine_root)
        event = self.root / "quarantine-events" / f"{key}-{suffix}.json"
        atomic_write_json(
            event,
            {
                "schema_version": "rex.control-prediction-cache.quarantine.v1",
                "cache_key": key,
                "quarantined_path": str(destination),
                "error_type": type(error).__name__,
                "error": str(error)[-2000:],
                "quarantined_at": _utc_now(),
            },
        )
        return destination

    @staticmethod
    def _copy_entry(source: ControlCacheEntry, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=False)
        records = sorted(source.manifest.artifacts, key=lambda item: item.path)
        for record in records:
            source_path = source.root / record.path
            target = destination / record.path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target, follow_symlinks=False)
        shutil.copy2(source.manifest_path, destination / "manifest.json", follow_symlinks=False)
        for path in sorted(destination.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        destination.chmod(0o555)

    def publish(
        self,
        identity: ControlCacheIdentity,
        *,
        bundle_path: str | Path,
        prediction_path: str | Path,
        train_feature_path: str | Path,
        train_target_path: str | Path,
        apply_feature_path: str | Path,
        provenance: ControlCacheProvenance,
    ) -> ControlCacheEntry:
        """Validate source artifacts and atomically publish one immutable entry."""

        train_feature, train_target, apply_feature = self._validate_inputs(
            identity, train_feature_path, train_target_path, apply_feature_path
        )
        source_bundle_path = Path(bundle_path)
        source_prediction = Path(prediction_path)
        _regular_file(source_bundle_path, label="source model bundle")
        _regular_file(source_prediction, label="source predictions")
        train = load_feature_view(train_feature)
        try:
            source_bundle: LoadedModelBundle = validate_model_bundle(
                source_bundle_path,
                expected_plugin=identity.plugin,
                expected_config_sha256=identity.config_sha256,
                expected_commit_sha=identity.source_commit,
                expected_data_view_sha256=identity.train_feature_sha256,
                expected_features=train,
            )
            load_prediction_artifact(source_prediction, apply_feature)
        except Exception as error:
            raise ControlCacheContractError(
                f"source control artifacts are invalid: {error}"
            ) from error
        if source_bundle.manifest.seed != identity.seed:
            raise ControlCacheContractError("source model bundle seed differs from cache identity")
        key = identity.cache_key
        final = self.entry_path(identity)
        with self._key_lock(key):
            if _lexists(final):
                try:
                    return self._validate_entry(final, identity, train_feature, apply_feature)
                except ControlCacheCorrupt as error:
                    self._quarantine(final, key, error)
            final.parent.mkdir(parents=True, exist_ok=True)
            staging = final.parent / f".{key}.{uuid.uuid4().hex}.tmp"
            try:
                bundle_root = staging / "bundle"
                bundle_root.mkdir(parents=True)
                for member_path in source_bundle.member_paths:
                    relative = member_path.relative_to(source_bundle.manifest_path.parent)
                    target = bundle_root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(member_path, target, follow_symlinks=False)
                shutil.copy2(
                    source_bundle.manifest_path,
                    bundle_root / BUNDLE_FILENAME,
                    follow_symlinks=False,
                )
                shutil.copy2(source_prediction, staging / "predictions.npz", follow_symlinks=False)
                artifacts = [
                    _artifact(bundle_root / BUNDLE_FILENAME, staging, "model_bundle"),
                    *(
                        _artifact(bundle_root / member.name, staging, "bundle_member")
                        for member in source_bundle.manifest.members
                    ),
                    _artifact(staging / "predictions.npz", staging, "predictions"),
                ]
                manifest = ControlCacheManifest(
                    cache_key=key,
                    identity=identity,
                    provenance=provenance,
                    artifacts=tuple(sorted(artifacts, key=lambda item: item.path)),
                    created_at=_utc_now(),
                )
                atomic_write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
                self._validate_entry(staging, identity, train_feature, apply_feature)
                for path in sorted(staging.rglob("*"), reverse=True):
                    path.chmod(0o555 if path.is_dir() else 0o444)
                staging.chmod(0o555)
                os.replace(staging, final)
                _fsync_directory(final.parent)
                return self._validate_entry(final, identity, train_feature, apply_feature)
            except Exception:
                if _lexists(staging):
                    if staging.is_dir() and not staging.is_symlink():
                        staging.chmod(0o755)
                        for path in staging.rglob("*"):
                            if not path.is_symlink():
                                path.chmod(0o755 if path.is_dir() else 0o644)
                        shutil.rmtree(staging)
                    else:
                        staging.unlink(missing_ok=True)
                raise

    def import_entry(
        self,
        identity: ControlCacheIdentity,
        *,
        train_feature_path: str | Path,
        train_target_path: str | Path,
        apply_feature_path: str | Path,
        destination_dir: str | Path,
        consumer: ControlCacheConsumer,
    ) -> ControlCacheImport | None:
        """Import a verified private copy, or return ``None`` after a miss/quarantine."""

        train_feature, train_target, apply_feature = self._validate_inputs(
            identity, train_feature_path, train_target_path, apply_feature_path
        )
        del train_target  # Its digest and schema were verified above; entries never contain labels.
        key = identity.cache_key
        source_root = self.entry_path(identity)
        destination = Path(destination_dir).resolve()
        evidence_path = destination.with_name(f"{destination.name}.import.json")
        with self._key_lock(key):
            if not _lexists(source_root):
                return None
            try:
                source = self._validate_entry(source_root, identity, train_feature, apply_feature)
            except ControlCacheCorrupt as error:
                self._quarantine(source_root, key, error)
                return None
            destination.parent.mkdir(parents=True, exist_ok=True)
            local: ControlCacheEntry | None = None
            if _lexists(destination):
                try:
                    local = self._validate_entry(
                        destination, identity, train_feature, apply_feature
                    )
                except ControlCacheCorrupt:
                    quarantine = destination.with_name(
                        f".{destination.name}.corrupt-{uuid.uuid4().hex}"
                    )
                    os.replace(destination, quarantine)
            if local is None:
                staging = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
                try:
                    self._copy_entry(source, staging)
                    self._validate_entry(staging, identity, train_feature, apply_feature)
                    os.replace(staging, destination)
                    _fsync_directory(destination.parent)
                    local = self._validate_entry(
                        destination, identity, train_feature, apply_feature
                    )
                except Exception:
                    if _lexists(staging):
                        staging.chmod(0o755)
                        for path in staging.rglob("*"):
                            if not path.is_symlink():
                                path.chmod(0o755 if path.is_dir() else 0o644)
                        shutil.rmtree(staging)
                    raise
            evidence = ControlCacheImportEvidence(
                cache_key=key,
                cache_manifest_sha256=sha256_file(local.manifest_path),
                producer=local.manifest.provenance,
                consumer=consumer,
                copied_artifacts=local.manifest.artifacts,
                imported_at=_utc_now(),
            )
            atomic_write_json(evidence_path, evidence.model_dump(mode="json"))
            return ControlCacheImport(
                cache_key=key,
                root=local.root,
                manifest_path=local.manifest_path,
                bundle_path=local.bundle_path,
                prediction_path=local.prediction_path,
                manifest=local.manifest,
                evidence_path=evidence_path,
            )
