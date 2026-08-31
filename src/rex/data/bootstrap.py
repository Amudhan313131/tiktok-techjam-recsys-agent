"""Trusted raw-data bootstrap producing sanitized split views and a label vault.

The implementation is deliberately independent of the organizer ``data.load``
helper. In particular, the test path never reads ``long_view`` into a Python or
NumPy target collection.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from rex.data.manifest import (
    canonical_json_bytes,
    load_benchmark_manifest,
    repo_root,
    sha256_bytes,
    sha256_file,
    verify_raw_dataset,
    verify_starter_manifest,
)
from rex.data.views import AUXILIARY_TARGET_COLUMNS
from rex.features.static_metadata import (
    StaticMetadataTables,
    fit_static_metadata_transform_for_rows,
    load_static_metadata,
    materialize_static_metadata_for_rows,
    static_metadata_manifest,
)


class DataContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawFeatureRow:
    date: int
    user_id: str
    video_id: str
    author_id: str
    tab: str
    duration_ms: float
    hour: int
    is_random: int
    time_ms: int
    source_row_key: int


@dataclass
class _SplitBuffer:
    dates: list[int] = field(default_factory=list)
    user_ids: list[str] = field(default_factory=list)
    video_ids: list[str] = field(default_factory=list)
    author_ids: list[str] = field(default_factory=list)
    tabs: list[str] = field(default_factory=list)
    durations_ms: list[float] = field(default_factory=list)
    hours: list[int] = field(default_factory=list)
    random_exposures: list[int] = field(default_factory=list)
    times_ms: list[int] = field(default_factory=list)
    source_row_keys: list[int] = field(default_factory=list)

    def append(self, row: RawFeatureRow) -> None:
        self.dates.append(row.date)
        self.user_ids.append(row.user_id)
        self.video_ids.append(row.video_id)
        self.author_ids.append(row.author_id)
        self.tabs.append(row.tab)
        self.durations_ms.append(row.duration_ms)
        self.hours.append(row.hour)
        self.random_exposures.append(row.is_random)
        self.times_ms.append(row.time_ms)
        self.source_row_keys.append(row.source_row_key)

    def arrays(self, tables, transform) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        arrays = {
            "row_id": np.arange(len(self.dates), dtype=np.int64),
            "date": np.asarray(self.dates, dtype=np.int32),
            "user_id": _string_array(self.user_ids),
            "video_id": _string_array(self.video_ids),
            "author_id": _string_array(self.author_ids),
            "tab": _string_array(self.tabs),
            "duration_ms": np.asarray(self.durations_ms, dtype=np.float32),
            "time_ms": np.asarray(self.times_ms, dtype=np.int64),
            "source_row_key": np.asarray(self.source_row_keys, dtype=np.int64),
            "fx__hour": np.asarray(self.hours, dtype=np.int8),
            "fx__is_rand": np.asarray(self.random_exposures, dtype=np.int8),
        }
        metadata_arrays, coverage = materialize_static_metadata_for_rows(
            tables,
            self.user_ids,
            self.video_ids,
            self.dates,
            self.durations_ms,
            transform,
        )
        arrays.update(metadata_arrays)
        return arrays, coverage


def _string_array(values: list[Any]) -> np.ndarray:
    width = max((len(str(value)) for value in values), default=1)
    return np.asarray([str(value) for value in values], dtype=f"<U{width}")


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _source_commit() -> str:
    override = os.environ.get("REX_SOURCE_COMMIT")
    if override:
        return override
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _source_key_prefixes(names: tuple[str, ...]) -> dict[str, int]:
    prefixes = {
        # Use the immutable logical source identity, not the source byte hash:
        # a hidden outcome poison must not change an inference feature.
        name: int(sha256_bytes(name.encode("utf-8"))[:8], 16) & 0x7FFFFFFF
        for name in names
    }
    if len(set(prefixes.values())) != len(prefixes):
        raise DataContractError("standard log source-key prefixes collided")
    return prefixes


def read_feature_rows(
    data_dir: str | Path,
    *,
    metadata_tables: StaticMetadataTables | None = None,
) -> Iterator[RawFeatureRow]:
    """Yield inference-safe columns only, for every official log row."""

    root = Path(data_dir)
    tables = metadata_tables or load_static_metadata(root)
    names = (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    )
    prefixes = _source_key_prefixes(names)
    for name in names:
        with (root / name).open(newline="", encoding="utf-8") as handle:
            for source_row_number, raw in enumerate(csv.DictReader(handle)):
                video_id = str(raw["video_id"])
                user_id = str(raw["user_id"])
                video_metadata = tables.videos.get(video_id, {})
                yield RawFeatureRow(
                    date=int(raw["date"]),
                    user_id=user_id,
                    video_id=video_id,
                    author_id=str(video_metadata.get("author_id") or "UNK"),
                    tab=str(raw["tab"]),
                    duration_ms=float(raw["duration_ms"]),
                    hour=int(float(raw.get("hourmin") or 0)) // 100,
                    is_random=int(float(raw.get("is_rand") or 0)),
                    time_ms=int(float(raw.get("time_ms") or 0)),
                    source_row_key=(prefixes[name] << 32) | source_row_number,
                )


def read_train_valid_targets(
    data_dir: str | Path,
    *,
    valid_end: int,
) -> Iterator[tuple[int, float]]:
    """Yield labels only for dates authorized for development.

    ``raw['long_view']`` is intentionally accessed only after the date gate. The
    second source file may contain test rows, but their outcome field is never
    converted or retained.
    """

    root = Path(data_dir)
    for name in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    ):
        with (root / name).open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                date = int(raw["date"])
                if date <= valid_end:
                    yield date, _binary_outcome(raw, "long_view")


def _binary_outcome(raw: dict[str, str], name: str) -> float:
    value = str(raw.get(name) or "0").strip()
    if value not in {"0", "1"}:
        raise DataContractError(f"{name} must be binary in the authorized target window")
    return float(value)


def read_train_valid_feedback_targets(
    data_dir: str | Path,
    *,
    valid_end: int,
) -> Iterator[tuple[int, dict[str, float]]]:
    """Yield capability-separated feedback only after the development date gate.

    In particular, no outcome field is indexed for a row after ``valid_end``.
    """

    root = Path(data_dir)
    for name in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    ):
        with (root / name).open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                date = int(raw["date"])
                if date <= valid_end:
                    yield date, {
                        target: _binary_outcome(raw, target)
                        for target in AUXILIARY_TARGET_COLUMNS
                    }


def _split_for_date(date: int, benchmark: dict[str, Any]) -> str | None:
    for split, contract in benchmark["splits"].items():
        if int(contract["split_start"]) <= date <= int(contract["split_end"]):
            return split
    return None


def build_split_views(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    benchmark: dict[str, Any] | None = None,
    metadata_summary: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Materialize safe feature views and train/valid-only target capabilities."""

    benchmark = benchmark or load_benchmark_manifest()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    label_vault = root / "label_vault"
    if label_vault.is_symlink():
        raise DataContractError(f"label vault may not be a symlink: {label_vault}")
    label_vault.mkdir(mode=0o700, exist_ok=True)
    label_vault.chmod(0o700)
    feedback_vault = root / "feedback_vault"
    if feedback_vault.is_symlink():
        raise DataContractError(f"feedback vault may not be a symlink: {feedback_vault}")
    feedback_vault.mkdir(mode=0o700, exist_ok=True)
    feedback_vault.chmod(0o700)
    forbidden_test_target = label_vault / "test_targets.npz"
    if forbidden_test_target.exists():
        raise DataContractError(
            f"forbidden stale test target artifact exists: {forbidden_test_target}"
        )
    forbidden_test_feedback = feedback_vault / "test_feedback_targets.npz"
    if forbidden_test_feedback.exists():
        raise DataContractError(
            f"forbidden stale test feedback artifact exists: {forbidden_test_feedback}"
        )

    buffers = {name: _SplitBuffer() for name in benchmark["splits"]}
    tables = load_static_metadata(data_dir)
    for row in read_feature_rows(data_dir, metadata_tables=tables):
        split = _split_for_date(row.date, benchmark)
        if split is not None:
            buffers[split].append(row)

    train_buffer = buffers["train"]
    static_transform = fit_static_metadata_transform_for_rows(
        tables,
        train_buffer.user_ids,
        train_buffer.video_ids,
        train_buffer.dates,
        train_buffer.durations_ms,
    )

    valid_end = int(benchmark["splits"]["valid"]["split_end"])
    feedback_targets: dict[str, dict[str, np.ndarray]] = {
        split: {
            name: np.empty(len(buffers[split].dates), dtype=np.float32)
            for name in AUXILIARY_TARGET_COLUMNS
        }
        for split in ("train", "valid")
    }
    feedback_positions = {"train": 0, "valid": 0}
    for date, feedback in read_train_valid_feedback_targets(data_dir, valid_end=valid_end):
        split = _split_for_date(date, benchmark)
        if split in feedback_targets:
            position = feedback_positions[split]
            if position >= len(buffers[split].dates):
                raise DataContractError(f"{split} feedback contains excess rows")
            for name, value in feedback.items():
                feedback_targets[split][name][position] = value
            feedback_positions[split] += 1

    result: dict[str, dict[str, Any]] = {}
    coverage_by_split: dict[str, dict[str, Any]] = {}
    for split, expected in benchmark["splits"].items():
        arrays, coverage = buffers[split].arrays(tables, static_transform)
        coverage_by_split[split] = coverage
        rows = len(arrays["row_id"])
        if rows != int(expected["rows"]):
            raise DataContractError(
                f"{split} row count mismatch: expected {expected['rows']}, observed {rows}"
            )
        dates = arrays["date"]
        if dates.size and (
            int(dates.min()) != int(expected["observed_date_min"])
            or int(dates.max()) != int(expected["observed_date_max"])
        ):
            raise DataContractError(
                f"{split} date mismatch: expected observed range "
                f"{expected['observed_date_min']}-{expected['observed_date_max']}, "
                f"observed {int(dates.min())}-{int(dates.max())}"
            )
        feature_path = root / f"{split}_features.npz"
        _write_npz_atomic(feature_path, **arrays)
        target_path: Path | None = None
        feedback_target_path: Path | None = None
        if split in feedback_targets:
            if feedback_positions[split] != rows:
                raise DataContractError(
                    f"{split} target alignment mismatch: "
                    f"features={rows}, targets={feedback_positions[split]}"
                )
            target_path = label_vault / f"{split}_targets.npz"
            _write_npz_atomic(
                target_path,
                long_view=feedback_targets[split]["long_view"],
            )
            target_path.chmod(0o600)
            feedback_target_path = feedback_vault / f"{split}_feedback_targets.npz"
            _write_npz_atomic(
                feedback_target_path,
                **feedback_targets[split],
            )
            feedback_target_path.chmod(0o600)
        feature_schema = {
            name: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in sorted(arrays.items())
        }
        result[split] = {
            "feature_path": str(feature_path.resolve()),
            "feature_sha256": sha256_file(feature_path),
            "target_path": str(target_path.resolve()) if target_path else None,
            "target_sha256": sha256_file(target_path) if target_path else None,
            "feedback_target_path": (
                str(feedback_target_path.resolve()) if feedback_target_path else None
            ),
            "feedback_target_sha256": (
                sha256_file(feedback_target_path) if feedback_target_path else None
            ),
            "split_start": int(expected["split_start"]),
            "split_end": int(expected["split_end"]),
            "observed_date_min": int(expected["observed_date_min"]),
            "observed_date_max": int(expected["observed_date_max"]),
            "row_count": rows,
            "row_id_sha256": sha256_bytes(arrays["row_id"].tobytes()),
            "user_id_sha256": sha256_bytes(arrays["user_id"].tobytes()),
            "video_id_sha256": sha256_bytes(arrays["video_id"].tobytes()),
            "time_ms_sha256": sha256_bytes(arrays["time_ms"].tobytes()),
            "source_row_key_sha256": sha256_bytes(arrays["source_row_key"].tobytes()),
            "feature_schema": feature_schema,
            "feature_schema_sha256": sha256_bytes(canonical_json_bytes(feature_schema)),
            "static_metadata_coverage": coverage,
        }
    static_summary = static_metadata_manifest(tables, static_transform, coverage_by_split)
    if metadata_summary is not None:
        metadata_summary.update(static_summary)
    return result


def _manifest_identity_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    identity = {
        key: value
        for key, value in manifest.items()
        if key not in {"generation_command", "generation_commit", "manifest_sha256"}
    }
    identity["splits"] = {
        split: {
            key: value
            for key, value in details.items()
            if key not in {"feature_path", "target_path", "feedback_target_path"}
        }
        for split, details in manifest["splits"].items()
    }
    return identity


def bootstrap_views(data_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    benchmark_path = repo_root() / "configs/frozen/benchmark.json"
    benchmark = load_benchmark_manifest(benchmark_path)
    starter = verify_starter_manifest()
    raw = verify_raw_dataset(data_dir, benchmark_path=benchmark_path)
    static_metadata: dict[str, Any] = {}
    splits = build_split_views(
        data_dir,
        output_dir,
        benchmark=benchmark,
        metadata_summary=static_metadata,
    )
    manifest: dict[str, Any] = {
        "schema_version": "2.0",
        "benchmark_sha256": sha256_file(benchmark_path),
        "starter_manifest_sha256": starter.manifest_sha256,
        "raw_dataset_identity_sha256": raw.identity_sha256,
        "raw_files": {
            name: {
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
                "data_rows": item.data_rows,
                "header": list(item.header),
            }
            for name, item in sorted(raw.files.items())
        },
        "generation_commit": _source_commit(),
        "generation_command": "python -m rex.cli bootstrap",
        "static_metadata": static_metadata,
        "temporal_order_contract": {
            "schema_version": "1.0",
            "time_column": "time_ms",
            "source_key_column": "source_row_key",
            "source_key_derivation": (
                "(logical_source_name_sha256_prefix_31bit << 32) | zero_based_source_row"
            ),
            "equal_timestamp_outcomes_visible": False,
        },
        "splits": splits,
    }
    identity = _manifest_identity_payload(manifest)
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(identity))
    _write_json_atomic(Path(output_dir) / "data_manifest.json", manifest)
    return manifest


def default_data_dir() -> Path:
    configured = os.environ.get("REX_DATA_ROOT")
    if configured:
        return Path(configured).resolve()
    return repo_root() / "data/KuaiRand-Pure/data"
