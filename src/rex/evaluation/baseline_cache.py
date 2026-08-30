"""Content-addressed, validation-only cache for baseline evidence.

This module deliberately does not decide when a production run may reuse a
baseline.  It provides the smaller trust boundary needed by that decision:
deterministic identity/key construction, an exact payload policy, atomic
publication, full integrity checks, and copy-only materialization.

The cache contains validation predictions and fitted baseline models.  It must
never contain data views, target arrays, test predictions, or submissions.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import fcntl

import numpy as np

from rex.data.manifest import canonical_json_bytes, sha256_bytes, sha256_file
from rex.execution.artifacts import atomic_write_json, load_prediction_artifact
from rex.models.bundle import validate_model_bundle


CACHE_SCHEMA_VERSION = "1.0"
CACHE_KIND = "baseline_evidence_cache"
CACHE_VERSION_DIRECTORY = "v1"
CACHE_MANIFEST_NAME = "cache_manifest.json"
CACHE_PROVENANCE_NAME = "cache_provenance.json"
CACHE_LOCK_DIRECTORY = ".locks"
CACHE_QUARANTINE_DIRECTORY = ".quarantine"
DEFAULT_BASELINE_SEEDS = (0, 1, 2, 3, 4)
OFFICIAL_FM_PLUGIN = "rex.models.official_fm:OfficialFMPlugin"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_LOG_MARKERS = (
    "submission.csv",
    "test_predictions",
    "test-predictions",
    "--score",
    "--make",
    "split=test",
    "split test",
)
_FALSE_ONLY_FIELDS = frozenset({"test_scored", "test_prediction_created", "submission_created"})


class BaselineCacheError(RuntimeError):
    """Base error for an unsafe, corrupt, or conflicting cache operation."""


class BaselineCacheCorrupt(BaselineCacheError):
    """Raised when a present cache entry fails closed validation."""


class BaselineCacheMiss(BaselineCacheError):
    """Raised when no sealed entry exists for an expected identity."""


@dataclass(frozen=True)
class BaselineCacheIdentity:
    """Path-independent identity of everything that can change baseline output."""

    benchmark_sha256: str
    raw_dataset_identity_sha256: str
    train_feature_sha256: str
    train_target_sha256: str
    valid_feature_sha256: str
    valid_target_sha256: str
    valid_row_id_sha256: str
    baseline_code_sha256: str
    baseline_config_sha256: str
    environment_sha256: str
    evaluator_sha256: str
    train_rows: int
    valid_rows: int
    schema_version: str = CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name.endswith("_sha256"):
                value = getattr(self, field.name)
                if not isinstance(value, str) or not _SHA256.fullmatch(value):
                    raise ValueError(f"{field.name} must be a lowercase SHA-256")
        if self.train_rows < 1 or self.valid_rows < 1:
            raise ValueError("baseline cache row counts must be positive")
        if self.schema_version != CACHE_SCHEMA_VERSION:
            raise ValueError(f"unsupported baseline cache schema: {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BaselineCacheIdentity":
        expected = {field.name for field in fields(cls)}
        if set(value) != expected:
            raise BaselineCacheCorrupt("cache identity fields are incomplete or unexpected")
        try:
            return cls(**dict(value))
        except (TypeError, ValueError) as error:
            raise BaselineCacheCorrupt(f"invalid cache identity: {error}") from error

    @property
    def key(self) -> str:
        return baseline_cache_key(self)


@dataclass(frozen=True)
class VerifiedBaselineCache:
    entry_path: Path
    key: str
    manifest_sha256: str
    payload_tree_sha256: str
    identity: BaselineCacheIdentity
    origin_run_id: str
    origin_source_commit: str
    created_at: str
    artifacts: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class BaselineCachePublication:
    cache: VerifiedBaselineCache
    published: bool


@dataclass(frozen=True)
class BaselineCacheMaterialization:
    cache: VerifiedBaselineCache
    destination: Path
    provenance_path: Path


@dataclass(frozen=True)
class BaselineCacheQuarantine:
    key: str
    quarantine_path: Path
    evidence_path: Path
    detected_error: str


def baseline_cache_key(identity: BaselineCacheIdentity) -> str:
    envelope = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": CACHE_KIND,
        "identity": identity.to_dict(),
    }
    return sha256_bytes(canonical_json_bytes(envelope))


def hash_named_files(files: Mapping[str, str | Path]) -> str:
    """Hash a named source closure without including machine-specific paths."""

    if not files:
        raise ValueError("named file hash requires at least one member")
    members: dict[str, str] = {}
    for name, raw_path in sorted(files.items()):
        relative = Path(name)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(f"unsafe named file: {name}")
        path = Path(raw_path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"named file is missing or unsafe: {path}")
        members[relative.as_posix()] = sha256_file(path)
    return sha256_bytes(canonical_json_bytes(members))


def hash_canonical_config(value: Mapping[str, Any]) -> str:
    """Return the stable hash used for an explicit baseline configuration."""

    return sha256_bytes(canonical_json_bytes(dict(value)))


def baseline_cache_entry_path(cache_root: str | Path, identity: BaselineCacheIdentity) -> Path:
    return Path(cache_root) / CACHE_VERSION_DIRECTORY / identity.key


def _cache_version_directory(entry: Path) -> Path:
    version = entry.parent
    if version.is_symlink() or not version.is_dir():
        raise BaselineCacheError(
            f"baseline cache version directory is missing or unsafe: {version}"
        )
    return version.resolve()


@contextmanager
def _key_lock(version: Path, key: str, *, exclusive: bool) -> Iterator[None]:
    """Hold one advisory cache-key lock without placing bytes in the seal."""

    if not _SHA256.fullmatch(key):
        raise BaselineCacheError("baseline cache lock key must be a lowercase SHA-256")
    if version.is_symlink() or not version.is_dir():
        raise BaselineCacheError(f"baseline cache version directory is unsafe: {version}")
    lock_root = version / CACHE_LOCK_DIRECTORY
    if lock_root.is_symlink():
        raise BaselineCacheError(f"baseline cache lock directory is unsafe: {lock_root}")
    created = not lock_root.exists()
    lock_root.mkdir(exist_ok=True)
    if created:
        _fsync_directory(version)
    lock_path = lock_root / f"{key}.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise BaselineCacheError(f"cannot open baseline cache lock: {lock_path}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise BaselineCacheError(f"baseline cache lock is not a regular file: {lock_path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def allowed_payload_paths(
    seeds: tuple[int, ...] = DEFAULT_BASELINE_SEEDS,
) -> frozenset[str]:
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("baseline cache seeds must be unique non-negative integers")
    common = {
        "environment.json",
        "command.json",
        "telemetry.json",
        "stdout.log",
        "stderr.log",
        "predictions.npz",
        "metrics.json",
        "config.json",
    }
    paths = {"summary.json"}
    paths.update(f"random/{name}" for name in common | {"artifacts.json"})
    paths.update(f"item-popularity/{name}" for name in common | {"statistics.json"})
    for seed in seeds:
        paths.update(
            f"seed-{seed}/{name}"
            for name in common
            | {
                "model.npz",
                "encoder.json",
                "model_bundle.json",
                "training.json",
                "artifacts.json",
            }
        )
    return frozenset(paths)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BaselineCacheCorrupt(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BaselineCacheCorrupt(f"{label} is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise BaselineCacheCorrupt(f"{label} must be a JSON object")
    return value


def _assert_valid_only_json(value: Any, *, path: str) -> None:
    if isinstance(value, dict):
        for raw_key, member in value.items():
            key = str(raw_key).lower()
            if key in _FALSE_ONLY_FIELDS and member is not False:
                raise BaselineCacheCorrupt(f"{path} violates valid-only field {raw_key}")
            if key in {"split", "development_split"} and member != "valid":
                raise BaselineCacheCorrupt(f"{path} contains non-validation split {member!r}")
            _assert_valid_only_json(member, path=path)
    elif isinstance(value, list):
        for member in value:
            _assert_valid_only_json(member, path=path)


def _expected_directories(files: frozenset[str]) -> set[str]:
    result = {"."}
    for name in files:
        parent = Path(name).parent
        while parent != Path("."):
            result.add(parent.as_posix())
            parent = parent.parent
    return result


def _inspect_exact_tree(root: Path, expected_files: frozenset[str]) -> dict[str, Path]:
    if root.is_symlink() or not root.is_dir():
        raise BaselineCacheCorrupt(f"baseline cache payload is missing or unsafe: {root}")
    observed_files: dict[str, Path] = {}
    observed_directories = {"."}
    for path in root.rglob("*"):
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise BaselineCacheCorrupt(f"cannot inspect cache payload member: {path}") from error
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(mode):
            raise BaselineCacheCorrupt(f"cache payload contains a symlink: {relative}")
        if stat.S_ISDIR(mode):
            observed_directories.add(relative)
        elif stat.S_ISREG(mode):
            observed_files[relative] = path
        else:
            raise BaselineCacheCorrupt(f"cache payload contains a non-regular file: {relative}")
    if set(observed_files) != set(expected_files):
        missing = sorted(set(expected_files) - set(observed_files))
        extra = sorted(set(observed_files) - set(expected_files))
        raise BaselineCacheCorrupt(
            f"cache payload file set differs; missing={missing[:5]}, extra={extra[:5]}"
        )
    expected_directories = _expected_directories(expected_files)
    if observed_directories != expected_directories:
        extra = sorted(observed_directories - expected_directories)
        raise BaselineCacheCorrupt(f"cache payload contains unexpected directories: {extra[:5]}")
    return observed_files


def _artifact_inventory(files: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for name, path in sorted(files.items())
    }


def _validate_inventory(files: Mapping[str, Path], raw_inventory: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_inventory, dict) or set(raw_inventory) != set(files):
        raise BaselineCacheCorrupt("cache artifact inventory differs from the payload")
    normalized: dict[str, dict[str, Any]] = {}
    for name, path in sorted(files.items()):
        raw = raw_inventory[name]
        if not isinstance(raw, dict) or set(raw) != {"sha256", "size_bytes"}:
            raise BaselineCacheCorrupt(f"cache artifact record is malformed: {name}")
        digest = raw.get("sha256")
        size = raw.get("size_bytes")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise BaselineCacheCorrupt(f"cache artifact SHA-256 is malformed: {name}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise BaselineCacheCorrupt(f"cache artifact size is malformed: {name}")
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise BaselineCacheCorrupt(f"cache artifact drifted: {name}")
        normalized[name] = {"sha256": digest, "size_bytes": size}
    return normalized


def _validate_metrics(
    metrics: Mapping[str, Any],
    identity: BaselineCacheIdentity,
    *,
    label: str,
    seed: int | None,
) -> None:
    if metrics.get("split") != "valid":
        raise BaselineCacheCorrupt(f"{label} metrics are not validation-only")
    if metrics.get("evaluator_sha256") != identity.evaluator_sha256:
        raise BaselineCacheCorrupt(f"{label} metrics use a different evaluator")
    if metrics.get("rows") != identity.valid_rows:
        raise BaselineCacheCorrupt(f"{label} metrics have the wrong validation row count")
    if metrics.get("seed") != seed:
        raise BaselineCacheCorrupt(f"{label} metrics have the wrong seed")
    for name in ("GAUC", "nDCG@5", "primary"):
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
            raise BaselineCacheCorrupt(f"{label} metric {name} is missing or non-finite")


def _validate_prediction(path: Path, identity: BaselineCacheIdentity, *, label: str) -> None:
    try:
        arrays = load_prediction_artifact(path)
    except Exception as error:
        raise BaselineCacheCorrupt(f"{label} predictions are invalid: {error}") from error
    if len(arrays["score"]) != identity.valid_rows:
        raise BaselineCacheCorrupt(f"{label} predictions have the wrong validation row count")
    observed_row_identity = sha256_bytes(np.asarray(arrays["row_id"], dtype=np.int64).tobytes())
    if observed_row_identity != identity.valid_row_id_sha256:
        raise BaselineCacheCorrupt(f"{label} prediction row identity differs")


def _validate_payload_semantics(
    root: Path,
    identity: BaselineCacheIdentity,
    *,
    seeds: tuple[int, ...],
    origin_source_commit: str,
) -> None:
    for path in sorted(root.rglob("*.json")):
        value = _read_json_object(path, label=path.relative_to(root).as_posix())
        _assert_valid_only_json(value, path=path.relative_to(root).as_posix())
    for path in sorted(root.rglob("*.log")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace").lower()
        except OSError as error:
            raise BaselineCacheCorrupt(f"cannot read cache log: {path}") from error
        if any(marker in text for marker in _FORBIDDEN_LOG_MARKERS):
            raise BaselineCacheCorrupt(
                f"cache log contains forbidden test/submission marker: {path.name}"
            )

    summary = _read_json_object(root / "summary.json", label="baseline summary")
    if summary.get("development_split") != "valid" or summary.get("test_scored") is not False:
        raise BaselineCacheCorrupt("baseline summary is not validation-only")
    acceptance = summary.get("acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("accepted") is not True:
        raise BaselineCacheCorrupt("only an accepted baseline may be cached")

    random_metrics = _read_json_object(root / "random/metrics.json", label="random metrics")
    popularity_metrics = _read_json_object(
        root / "item-popularity/metrics.json", label="item-popularity metrics"
    )
    _validate_metrics(random_metrics, identity, label="random", seed=0)
    _validate_metrics(popularity_metrics, identity, label="item-popularity", seed=0)
    if (
        summary.get("random") != random_metrics
        or summary.get("item_popularity") != popularity_metrics
    ):
        raise BaselineCacheCorrupt("baseline summary differs from lower-bound metric evidence")

    fm_summary = summary.get("fm")
    if not isinstance(fm_summary, dict) or not isinstance(fm_summary.get("seeds"), list):
        raise BaselineCacheCorrupt("baseline summary FM evidence is malformed")
    summarized = fm_summary["seeds"]
    if [item.get("seed") for item in summarized if isinstance(item, dict)] != list(seeds):
        raise BaselineCacheCorrupt("baseline summary has an unexpected seed set or order")

    for seed, summarized_seed in zip(seeds, summarized, strict=True):
        if not isinstance(summarized_seed, dict):
            raise BaselineCacheCorrupt(f"baseline summary seed {seed} is malformed")
        label = f"seed-{seed}"
        metrics = _read_json_object(root / label / "metrics.json", label=f"{label} metrics")
        config = _read_json_object(root / label / "config.json", label=f"{label} config")
        _validate_metrics(metrics, identity, label=label, seed=seed)
        if config.get("split") != "valid" or config.get("seed") != seed:
            raise BaselineCacheCorrupt(f"{label} configuration is not valid-only")
        if summarized_seed.get("metrics") != metrics:
            raise BaselineCacheCorrupt(f"baseline summary differs from {label} metrics")
        try:
            validate_model_bundle(
                root / label / "model_bundle.json",
                expected_plugin=OFFICIAL_FM_PLUGIN,
                expected_config_sha256=sha256_file(root / label / "config.json"),
                expected_commit_sha=origin_source_commit,
                expected_data_view_sha256=identity.train_feature_sha256,
            )
        except Exception as error:
            raise BaselineCacheCorrupt(f"{label} model bundle is invalid: {error}") from error

    _validate_prediction(root / "random/predictions.npz", identity, label="random")
    _validate_prediction(
        root / "item-popularity/predictions.npz", identity, label="item-popularity"
    )
    for seed in seeds:
        _validate_prediction(root / f"seed-{seed}/predictions.npz", identity, label=f"seed-{seed}")


def _validate_origin(raw: Any) -> tuple[str, str, str]:
    if not isinstance(raw, dict) or set(raw) != {"run_id", "source_commit", "created_at"}:
        raise BaselineCacheCorrupt("cache origin provenance is malformed")
    values = tuple(raw.get(name) for name in ("run_id", "source_commit", "created_at"))
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise BaselineCacheCorrupt("cache origin provenance values must be non-empty strings")
    return values  # type: ignore[return-value]


def _validate_baseline_cache_unlocked(
    entry_path: str | Path,
    expected_identity: BaselineCacheIdentity,
    *,
    seeds: tuple[int, ...] = DEFAULT_BASELINE_SEEDS,
) -> VerifiedBaselineCache:
    """Re-hash and semantically validate one sealed cache entry."""

    entry = Path(entry_path)
    if not entry.exists():
        raise BaselineCacheMiss(f"baseline cache entry is missing: {entry}")
    if entry.is_symlink() or not entry.is_dir():
        raise BaselineCacheCorrupt(f"baseline cache entry is unsafe: {entry}")
    if entry.name != expected_identity.key:
        raise BaselineCacheCorrupt("baseline cache directory name differs from its identity key")
    top_level = {path.name for path in entry.iterdir()}
    if top_level != {CACHE_MANIFEST_NAME, "payload"}:
        raise BaselineCacheCorrupt("baseline cache entry contains unexpected top-level members")

    manifest_path = entry / CACHE_MANIFEST_NAME
    manifest = _read_json_object(manifest_path, label="cache manifest")
    expected_fields = {
        "schema_version",
        "kind",
        "cache_key_sha256",
        "identity",
        "development_split",
        "test_scored",
        "test_prediction_created",
        "submission_created",
        "seeds",
        "origin",
        "payload_tree_sha256",
        "artifacts",
    }
    if set(manifest) != expected_fields:
        raise BaselineCacheCorrupt("cache manifest fields are incomplete or unexpected")
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION or manifest.get("kind") != CACHE_KIND:
        raise BaselineCacheCorrupt("cache manifest schema or kind is unsupported")
    if (
        manifest.get("development_split") != "valid"
        or manifest.get("test_scored") is not False
        or manifest.get("test_prediction_created") is not False
        or manifest.get("submission_created") is not False
    ):
        raise BaselineCacheCorrupt("cache manifest is not validation-only")
    if manifest.get("seeds") != list(seeds):
        raise BaselineCacheCorrupt("cache manifest seed contract differs")

    raw_identity = manifest.get("identity")
    if not isinstance(raw_identity, dict):
        raise BaselineCacheCorrupt("cache manifest identity is malformed")
    identity = BaselineCacheIdentity.from_dict(raw_identity)
    if identity != expected_identity or manifest.get("cache_key_sha256") != identity.key:
        raise BaselineCacheCorrupt("cache manifest identity or key differs")
    origin_run_id, origin_source_commit, created_at = _validate_origin(manifest.get("origin"))

    expected_files = allowed_payload_paths(seeds)
    files = _inspect_exact_tree(entry / "payload", expected_files)
    inventory = _validate_inventory(files, manifest.get("artifacts"))
    tree_hash = sha256_bytes(canonical_json_bytes(inventory))
    if manifest.get("payload_tree_sha256") != tree_hash:
        raise BaselineCacheCorrupt("cache payload tree hash differs")
    _validate_payload_semantics(
        entry / "payload",
        identity,
        seeds=seeds,
        origin_source_commit=origin_source_commit,
    )
    return VerifiedBaselineCache(
        entry_path=entry.resolve(),
        key=identity.key,
        manifest_sha256=sha256_file(manifest_path),
        payload_tree_sha256=tree_hash,
        identity=identity,
        origin_run_id=origin_run_id,
        origin_source_commit=origin_source_commit,
        created_at=created_at,
        artifacts=inventory,
    )


def validate_baseline_cache(
    entry_path: str | Path,
    expected_identity: BaselineCacheIdentity,
    *,
    seeds: tuple[int, ...] = DEFAULT_BASELINE_SEEDS,
) -> VerifiedBaselineCache:
    """Validate one cache entry while excluding publication or quarantine."""

    entry = Path(entry_path)
    if not entry.exists() and not entry.is_symlink():
        raise BaselineCacheMiss(f"baseline cache entry is missing: {entry}")
    version = _cache_version_directory(entry)
    with _key_lock(version, expected_identity.key, exclusive=False):
        return _validate_baseline_cache_unlocked(entry, expected_identity, seeds=seeds)


def quarantine_baseline_cache(
    entry_path: str | Path,
    expected_identity: BaselineCacheIdentity,
    error: BaseException,
) -> BaselineCacheQuarantine | None:
    """Atomically isolate a corrupt entry and preserve the rejection reason."""

    entry = Path(entry_path)
    if not entry.exists() and not entry.is_symlink():
        return None
    version = _cache_version_directory(entry)
    key = expected_identity.key
    with _key_lock(version, key, exclusive=True):
        if not entry.exists() and not entry.is_symlink():
            return None
        quarantine_root = version / CACHE_QUARANTINE_DIRECTORY
        if quarantine_root.is_symlink():
            raise BaselineCacheError("baseline cache quarantine directory is unsafe")
        quarantine_root.mkdir(exist_ok=True)
        suffix = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex}"
        destination = quarantine_root / f"{key}-{suffix}"
        os.replace(entry, destination)
        evidence_path = atomic_write_json(
            quarantine_root / f"{key}-{suffix}.json",
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "kind": "baseline_cache_quarantine",
                "cache_key_sha256": key,
                "quarantined_path": str(destination.resolve()),
                "error_type": type(error).__name__,
                "error": str(error)[-2000:],
                "quarantined_at": _utc_now(),
                "test_scored": False,
            },
        )
        _fsync_directory(quarantine_root)
        _fsync_directory(version)
        return BaselineCacheQuarantine(
            key=key,
            quarantine_path=destination.resolve(),
            evidence_path=evidence_path.resolve(),
            detected_error=f"{type(error).__name__}: {str(error)[-2000:]}",
        )


def _copy_regular(source: Path, destination: Path) -> None:
    try:
        before = source.lstat()
    except OSError as error:
        raise BaselineCacheCorrupt(f"cache source is missing: {source}") from error
    if source.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise BaselineCacheCorrupt(f"cache source is not a regular file: {source}")
    digest = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or sha256_file(destination) != digest:
        raise BaselineCacheCorrupt(f"cache source changed while being copied: {source}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_exact_payload(source: Path, destination: Path, expected: frozenset[str]) -> None:
    files = _inspect_exact_tree(source, expected)
    destination.mkdir(parents=True)
    for name, path in sorted(files.items()):
        _copy_regular(path, destination / name)
    for directory in sorted(
        (path for path in destination.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        _fsync_directory(directory)
    _fsync_directory(destination)


def _prepare_cache_parent(cache_root: Path) -> Path:
    if cache_root.is_symlink():
        raise BaselineCacheError(f"baseline cache root may not be a symlink: {cache_root}")
    cache_root.mkdir(parents=True, exist_ok=True)
    root = cache_root.resolve()
    version = root / CACHE_VERSION_DIRECTORY
    if version.is_symlink():
        raise BaselineCacheError(f"baseline cache version directory is unsafe: {version}")
    version.mkdir(exist_ok=True)
    return version


def publish_baseline_cache(
    cache_root: str | Path,
    evidence_dir: str | Path,
    identity: BaselineCacheIdentity,
    *,
    origin_run_id: str,
    origin_source_commit: str,
    created_at: str | None = None,
    seeds: tuple[int, ...] = DEFAULT_BASELINE_SEEDS,
) -> BaselineCachePublication:
    """Validate and atomically publish accepted run-local baseline evidence."""

    if not origin_run_id.strip() or not origin_source_commit.strip():
        raise ValueError("baseline cache origin run and source commit must be non-empty")
    expected = allowed_payload_paths(seeds)
    source = Path(evidence_dir)
    source_files = _inspect_exact_tree(source, expected)
    _validate_payload_semantics(
        source,
        identity,
        seeds=seeds,
        origin_source_commit=origin_source_commit,
    )

    version = _prepare_cache_parent(Path(cache_root))
    destination = version / identity.key
    if destination.exists() or destination.is_symlink():
        return BaselineCachePublication(
            validate_baseline_cache(destination, identity, seeds=seeds), published=False
        )

    staging = version / f".{identity.key}.{uuid.uuid4().hex}.tmp"
    try:
        staging.mkdir()
        payload = staging / "payload"
        _copy_exact_payload(source, payload, expected)
        copied = _inspect_exact_tree(payload, expected)
        inventory = _artifact_inventory(copied)
        # Detect a source mutation even if it happened after its individual copy.
        if _artifact_inventory(source_files) != inventory:
            raise BaselineCacheCorrupt("baseline evidence changed during cache publication")
        tree_hash = sha256_bytes(canonical_json_bytes(inventory))
        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "kind": CACHE_KIND,
            "cache_key_sha256": identity.key,
            "identity": identity.to_dict(),
            "development_split": "valid",
            "test_scored": False,
            "test_prediction_created": False,
            "submission_created": False,
            "seeds": list(seeds),
            "origin": {
                "run_id": origin_run_id,
                "source_commit": origin_source_commit,
                "created_at": created_at or _utc_now(),
            },
            "payload_tree_sha256": tree_hash,
            "artifacts": inventory,
        }
        atomic_write_json(staging / CACHE_MANIFEST_NAME, manifest)
        _fsync_directory(staging)
        try:
            os.rename(staging, destination)
        except OSError:
            if not destination.exists() and not destination.is_symlink():
                raise
            # Another publisher won. Its complete seal is the only acceptable result.
            return BaselineCachePublication(
                validate_baseline_cache(destination, identity, seeds=seeds), published=False
            )
        _fsync_directory(version)
        return BaselineCachePublication(
            validate_baseline_cache(destination, identity, seeds=seeds), published=True
        )
    finally:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)


def materialize_baseline_cache(
    entry_path: str | Path,
    destination_dir: str | Path,
    expected_identity: BaselineCacheIdentity,
    *,
    materialized_at: str | None = None,
    seeds: tuple[int, ...] = DEFAULT_BASELINE_SEEDS,
) -> BaselineCacheMaterialization:
    """Copy a verified payload into a new run-local directory and re-hash it."""

    cache = validate_baseline_cache(entry_path, expected_identity, seeds=seeds)
    destination = Path(destination_dir)
    if destination.exists() or destination.is_symlink():
        raise BaselineCacheError(
            f"refusing to replace an existing baseline evidence directory: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.materializing"
    expected = allowed_payload_paths(seeds)
    try:
        _copy_exact_payload(cache.entry_path / "payload", staging, expected)
        copied = _inspect_exact_tree(staging, expected)
        inventory = _artifact_inventory(copied)
        if inventory != dict(cache.artifacts):
            raise BaselineCacheCorrupt("materialized baseline differs from the cache seal")
        provenance = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "kind": "baseline_cache_materialization",
            "cache_hit": True,
            "cache_key_sha256": cache.key,
            "cache_manifest_sha256": cache.manifest_sha256,
            "payload_tree_sha256": cache.payload_tree_sha256,
            "identity": cache.identity.to_dict(),
            "origin": {
                "run_id": cache.origin_run_id,
                "source_commit": cache.origin_source_commit,
                "created_at": cache.created_at,
            },
            "materialized_at": materialized_at or _utc_now(),
            "development_split": "valid",
            "test_scored": False,
            "test_prediction_created": False,
            "submission_created": False,
        }
        provenance_path = atomic_write_json(staging / CACHE_PROVENANCE_NAME, provenance)
        _fsync_directory(staging)
        os.rename(staging, destination)
        _fsync_directory(destination.parent)
        return BaselineCacheMaterialization(
            cache=cache,
            destination=destination.resolve(),
            provenance_path=(destination / provenance_path.name).resolve(),
        )
    finally:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
