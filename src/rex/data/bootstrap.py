"""Trusted raw-data bootstrap producing sanitized split views and a label vault."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from rex.data.manifest import (
    canonical_json_bytes,
    load_benchmark_manifest,
    repo_root,
    sha256_bytes,
    sha256_file,
    verify_starter_manifest,
)


class DataContractError(RuntimeError):
    pass


def _load_starter_data_module() -> ModuleType:
    starter = verify_starter_manifest()
    path = starter.root / "data.py"
    spec = importlib.util.spec_from_file_location("rex_frozen_starter_data", path)
    if spec is None or spec.loader is None:
        raise DataContractError(f"cannot load frozen starter data module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_raw_splits(data_dir: str | Path) -> dict[str, list[tuple[Any, ...]]]:
    module = _load_starter_data_module()
    data_dir = Path(data_dir)
    required = (
        "video_features_basic_pure.csv",
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    )
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise DataContractError(f"raw data files missing in {data_dir}: {missing}")
    return module.load(str(data_dir))


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


def bootstrap_views(data_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    benchmark = load_benchmark_manifest()
    starter = verify_starter_manifest()
    splits = load_raw_splits(data_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    label_vault = output / "label_vault"
    label_vault.mkdir(mode=0o700, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "starter_manifest_sha256": starter.manifest_sha256,
        "splits": {},
    }
    for split, expected in benchmark["splits"].items():
        rows = splits.get(split, [])
        if len(rows) != expected["rows"]:
            raise DataContractError(
                f"{split} row count mismatch: expected {expected['rows']}, observed {len(rows)}"
            )
        dates = np.asarray([row[0] for row in rows], dtype=np.int32)
        if dates.size and (
            int(dates.min()) != expected["observed_date_min"]
            or int(dates.max()) != expected["observed_date_max"]
        ):
            raise DataContractError(
                f"{split} date mismatch: expected observed range "
                f"{expected['observed_date_min']}-{expected['observed_date_max']}, "
                f"observed {int(dates.min())}-{int(dates.max())}"
            )

        feature_path = output / f"{split}_features.npz"
        _write_npz_atomic(
            feature_path,
            row_id=np.arange(len(rows), dtype=np.int64),
            date=dates,
            user_id=_string_array([row[1] for row in rows]),
            video_id=_string_array([row[2] for row in rows]),
            author_id=_string_array([row[3] for row in rows]),
            tab=_string_array([row[4] for row in rows]),
            duration_ms=np.asarray([row[5] for row in rows], dtype=np.float32),
        )

        target_path: Path | None = None
        if split in {"train", "valid"}:
            target_path = label_vault / f"{split}_targets.npz"
            _write_npz_atomic(
                target_path,
                long_view=np.asarray([row[6] for row in rows], dtype=np.float32),
            )
            target_path.chmod(0o600)

        alignment = {
            "row_count": len(rows),
            "user_id_sha256": sha256_bytes(_string_array([r[1] for r in rows]).tobytes()),
            "video_id_sha256": sha256_bytes(_string_array([r[2] for r in rows]).tobytes()),
        }
        manifest["splits"][split] = {
            "feature_path": str(feature_path.resolve()),
            "feature_sha256": sha256_file(feature_path),
            "target_path": str(target_path.resolve()) if target_path else None,
            "target_sha256": sha256_file(target_path) if target_path else None,
            "split_start": expected["split_start"],
            "split_end": expected["split_end"],
            "observed_date_min": expected["observed_date_min"],
            "observed_date_max": expected["observed_date_max"],
            **alignment,
        }

    identity = {
        "schema_version": manifest["schema_version"],
        "starter_manifest_sha256": manifest["starter_manifest_sha256"],
        "splits": {
            split: {
                key: value
                for key, value in details.items()
                if key not in {"feature_path", "target_path"}
            }
            for split, details in manifest["splits"].items()
        },
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(identity))
    manifest_path = output / "data_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return manifest


def default_data_dir() -> Path:
    return repo_root() / "data/KuaiRand-Pure/data"
