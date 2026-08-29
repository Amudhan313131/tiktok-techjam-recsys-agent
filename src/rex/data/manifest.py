"""Benchmark and starter-kit manifest verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ManifestError(RuntimeError):
    """Raised when frozen benchmark inputs drift from their manifest."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ManifestError(f"manifest must be a JSON object: {path}")
    return value


@dataclass(frozen=True)
class VerifiedStarter:
    root: Path
    hashes: dict[str, str]
    manifest_sha256: str


def verify_starter_manifest(
    manifest_path: str | Path | None = None,
    root: str | Path | None = None,
) -> VerifiedStarter:
    project = repo_root()
    manifest_path = Path(manifest_path or project / "configs/frozen/starter_manifest.json")
    manifest = load_json(manifest_path)
    starter_root = Path(root or project / str(manifest["root"])).resolve()
    if not starter_root.is_dir():
        raise ManifestError(f"starter root missing: {starter_root}")

    actual: dict[str, str] = {}
    mismatches: list[str] = []
    for relative, expected in manifest.get("files", {}).items():
        candidate = starter_root / relative
        if not candidate.is_file():
            mismatches.append(f"missing {relative}")
            continue
        observed = sha256_file(candidate)
        actual[relative] = observed
        if observed != expected:
            mismatches.append(f"{relative}: expected {expected}, observed {observed}")
    if mismatches:
        raise ManifestError("starter manifest verification failed: " + "; ".join(mismatches))

    return VerifiedStarter(
        root=starter_root,
        hashes=actual,
        manifest_sha256=sha256_file(manifest_path),
    )


def load_benchmark_manifest(path: str | Path | None = None) -> dict[str, Any]:
    project = repo_root()
    benchmark_path = Path(path or project / "configs/frozen/benchmark.json")
    manifest = load_json(benchmark_path)
    if manifest.get("label") != "long_view":
        raise ManifestError("benchmark label must be long_view")
    if manifest.get("metrics") != ["GAUC", "nDCG@5"]:
        raise ManifestError("benchmark metric contract drifted")
    return manifest
