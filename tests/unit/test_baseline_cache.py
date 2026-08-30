from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import rex.evaluation.baseline_cache as baseline_cache
from rex.data.manifest import sha256_bytes, sha256_file
from rex.evaluation.baseline_cache import (
    CACHE_PROVENANCE_NAME,
    BaselineCacheCorrupt,
    BaselineCacheError,
    BaselineCacheIdentity,
    allowed_payload_paths,
    baseline_cache_entry_path,
    materialize_baseline_cache,
    publish_baseline_cache,
    quarantine_baseline_cache,
    validate_baseline_cache,
)


HASH = "1" * 64
EVALUATOR_HASH = "e" * 64
SOURCE_COMMIT = "baseline-source-commit"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _identity(rows: int = 3) -> BaselineCacheIdentity:
    row_hash = sha256_bytes(np.arange(rows, dtype=np.int64).tobytes())
    return BaselineCacheIdentity(
        benchmark_sha256="0" * 64,
        raw_dataset_identity_sha256="1" * 64,
        train_feature_sha256="2" * 64,
        train_target_sha256="3" * 64,
        valid_feature_sha256="4" * 64,
        valid_target_sha256="5" * 64,
        valid_row_id_sha256=row_hash,
        baseline_code_sha256="6" * 64,
        baseline_config_sha256="7" * 64,
        environment_sha256="8" * 64,
        evaluator_sha256=EVALUATOR_HASH,
        train_rows=5,
        valid_rows=rows,
    )


def _metrics(identity: BaselineCacheIdentity, *, primary: float, seed: int) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "GAUC": primary,
        "nDCG@5": primary,
        "primary": primary,
        "users": 2,
        "rows": identity.valid_rows,
        "evaluator_sha256": identity.evaluator_sha256,
        "split": "valid",
        "fold": None,
        "seed": seed,
    }


def _prediction(path: Path, rows: int, *, offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            row_id=np.arange(rows, dtype=np.int64),
            user_id=np.asarray([f"u{index % 2}" for index in range(rows)]),
            video_id=np.asarray([f"v{index}" for index in range(rows)]),
            score=np.linspace(offset, offset + 0.2, rows, dtype=np.float64),
        )


def _common_evidence(
    directory: Path,
    identity: BaselineCacheIdentity,
    *,
    operation: str,
    config: dict[str, object],
    metrics: dict[str, object],
    extra_json_name: str,
    extra_json: dict[str, object],
    offset: float,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _prediction(directory / "predictions.npz", identity.valid_rows, offset=offset)
    _write_json(directory / "metrics.json", metrics)
    _write_json(directory / "config.json", config)
    _write_json(directory / "environment.json", {"python": "3.11", "numpy": "test"})
    _write_json(
        directory / "command.json",
        {"operation": operation, "arguments": {"split": "valid"}},
    )
    _write_json(
        directory / "telemetry.json",
        {"wall_seconds": 1.0, "cpu_seconds": 1.0, "gpu_seconds": 0.0},
    )
    _write_json(directory / extra_json_name, extra_json)
    (directory / "stdout.log").write_text("valid-only baseline completed\n", encoding="utf-8")
    (directory / "stderr.log").write_text("", encoding="utf-8")


def _seed_evidence(
    root: Path,
    identity: BaselineCacheIdentity,
    *,
    seed: int,
    primary: float,
) -> dict[str, object]:
    directory = root / f"seed-{seed}"
    config = {
        "model": "fm",
        "seed": seed,
        "k": 16,
        "lr": 0.001,
        "l2": 1e-6,
        "epochs": 40,
        "batch_size": 8192,
        "patience": 4,
        "split": "valid",
    }
    metrics = _metrics(identity, primary=primary, seed=seed)
    _common_evidence(
        directory,
        identity,
        operation="reproduce_fm_seed",
        config=config,
        metrics=metrics,
        extra_json_name="training.json",
        extra_json={"history": [], "best_epoch": 1, "stopped_epoch": 1},
        offset=seed / 10,
    )
    model = directory / "model.npz"
    with model.open("wb") as handle:
        np.savez_compressed(handle, weights=np.asarray([seed], dtype=np.float64))
    _write_json(directory / "encoder.json", {"columns": [], "offsets": {}})
    _write_json(
        directory / "artifacts.json",
        {
            "model_sha256": sha256_file(model),
            "encoder_sha256": sha256_file(directory / "encoder.json"),
            "predictions_sha256": sha256_file(directory / "predictions.npz"),
            "train_feature_sha256": identity.train_feature_sha256,
            "valid_feature_sha256": identity.valid_feature_sha256,
        },
    )
    members = []
    for name, kind in (("model.npz", "checkpoint"), ("encoder.json", "checkpoint_sidecar")):
        path = directory / name
        members.append(
            {
                "name": name,
                "kind": kind,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    _write_json(
        directory / "model_bundle.json",
        {
            "schema_version": "1.0",
            "plugin": "rex.models.official_fm:OfficialFMPlugin",
            "seed": seed,
            "commit_sha": SOURCE_COMMIT,
            "config_sha256": sha256_file(directory / "config.json"),
            "data_view_sha256": identity.train_feature_sha256,
            "primary_member": "model.npz",
            "feature_schema": {},
            "members": members,
        },
    )
    return {
        "seed": seed,
        "best_epoch": 1,
        "metrics": metrics,
        "prediction_sha256": sha256_file(directory / "predictions.npz"),
        "model_sha256": sha256_file(model),
    }


def _evidence(root: Path, identity: BaselineCacheIdentity) -> Path:
    random_metrics = _metrics(identity, primary=0.48, seed=0)
    popularity_metrics = _metrics(identity, primary=0.58, seed=0)
    _common_evidence(
        root / "random",
        identity,
        operation="reproduce_random",
        config={"model": "random", "seed": 0, "split": "valid"},
        metrics=random_metrics,
        extra_json_name="artifacts.json",
        extra_json={"predictions_sha256": HASH, "feature_sha256": identity.valid_feature_sha256},
        offset=0.0,
    )
    _common_evidence(
        root / "item-popularity",
        identity,
        operation="reproduce_item_popularity",
        config={"model": "item_popularity", "prior_strength": 20.0, "split": "valid"},
        metrics=popularity_metrics,
        extra_json_name="statistics.json",
        extra_json={"global_mean": 0.5, "items": 2},
        offset=0.1,
    )
    seeds = [
        _seed_evidence(root, identity, seed=seed, primary=0.60 + seed / 1000) for seed in range(5)
    ]
    primaries = np.asarray([item["metrics"]["primary"] for item in seeds])
    _write_json(
        root / "summary.json",
        {
            "schema_version": "1.0",
            "development_split": "valid",
            "test_scored": False,
            "random": random_metrics,
            "item_popularity": popularity_metrics,
            "fm": {
                "mean_primary": float(primaries.mean()),
                "std_primary": float(primaries.std()),
                "seeds": seeds,
            },
            "acceptance": {"accepted": True, "reasons": [], "observed": {}},
        },
    )
    assert {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()} == set(
        allowed_payload_paths()
    )
    return root


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("train_feature_sha256", "a" * 64),
        ("baseline_code_sha256", "b" * 64),
        ("baseline_config_sha256", "c" * 64),
        ("environment_sha256", "d" * 64),
        ("evaluator_sha256", "f" * 64),
        ("valid_rows", 4),
    ],
)
def test_identity_key_is_deterministic_and_covers_every_boundary(field: str, replacement: object):
    first = _identity()
    second = _identity()
    assert first.key == second.key
    assert replace(first, **{field: replacement}).key != first.key


def test_publish_validate_and_copy_materialization_with_provenance(tmp_path: Path) -> None:
    identity = _identity()
    evidence = _evidence(tmp_path / "evidence", identity)
    publication = publish_baseline_cache(
        tmp_path / "cache",
        evidence,
        identity,
        origin_run_id="run-1",
        origin_source_commit=SOURCE_COMMIT,
        created_at="2026-08-31T00:00:00+00:00",
    )
    assert publication.published
    assert publication.cache.key == identity.key

    verified = validate_baseline_cache(publication.cache.entry_path, identity)
    output = tmp_path / "run/baseline/evidence"
    materialized = materialize_baseline_cache(
        verified.entry_path,
        output,
        identity,
        materialized_at="2026-08-31T01:00:00+00:00",
    )
    provenance = json.loads(materialized.provenance_path.read_text(encoding="utf-8"))
    assert provenance["cache_key_sha256"] == identity.key
    assert provenance["cache_manifest_sha256"] == verified.manifest_sha256
    assert provenance["origin"]["source_commit"] == SOURCE_COMMIT
    assert provenance["test_scored"] is False
    source_prediction = verified.entry_path / "payload/seed-0/predictions.npz"
    copied_prediction = output / "seed-0/predictions.npz"
    assert sha256_file(source_prediction) == sha256_file(copied_prediction)
    assert source_prediction.stat().st_ino != copied_prediction.stat().st_ino
    assert (output / CACHE_PROVENANCE_NAME).is_file()


def test_second_publish_reuses_only_a_fully_verified_seal(tmp_path: Path) -> None:
    identity = _identity()
    evidence = _evidence(tmp_path / "evidence", identity)
    first = publish_baseline_cache(
        tmp_path / "cache",
        evidence,
        identity,
        origin_run_id="run-1",
        origin_source_commit=SOURCE_COMMIT,
    )
    second = publish_baseline_cache(
        tmp_path / "cache",
        evidence,
        identity,
        origin_run_id="run-2",
        origin_source_commit=SOURCE_COMMIT,
    )
    assert first.published
    assert not second.published
    assert second.cache.manifest_sha256 == first.cache.manifest_sha256


def test_publish_rejects_extra_file_symlink_and_test_split(tmp_path: Path) -> None:
    identity = _identity()

    extra = _evidence(tmp_path / "extra", identity)
    (extra / "unexpected.txt").write_text("no", encoding="utf-8")
    with pytest.raises(BaselineCacheCorrupt, match="file set differs"):
        publish_baseline_cache(
            tmp_path / "cache-extra",
            extra,
            identity,
            origin_run_id="run",
            origin_source_commit=SOURCE_COMMIT,
        )

    linked = _evidence(tmp_path / "linked", identity)
    (linked / "random/stdout.log").unlink()
    (linked / "random/stdout.log").symlink_to(linked / "random/stderr.log")
    with pytest.raises(BaselineCacheCorrupt, match="symlink"):
        publish_baseline_cache(
            tmp_path / "cache-linked",
            linked,
            identity,
            origin_run_id="run",
            origin_source_commit=SOURCE_COMMIT,
        )

    test_split = _evidence(tmp_path / "test-split", identity)
    metrics_path = test_split / "seed-2/metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["split"] = "test"
    _write_json(metrics_path, metrics)
    with pytest.raises(BaselineCacheCorrupt, match="non-validation split"):
        publish_baseline_cache(
            tmp_path / "cache-test",
            test_split,
            identity,
            origin_run_id="run",
            origin_source_commit=SOURCE_COMMIT,
        )


def test_validate_rejects_tamper_extra_member_and_entry_symlink(tmp_path: Path) -> None:
    identity = _identity()
    evidence = _evidence(tmp_path / "evidence", identity)
    published = publish_baseline_cache(
        tmp_path / "cache",
        evidence,
        identity,
        origin_run_id="run",
        origin_source_commit=SOURCE_COMMIT,
    ).cache

    (published.entry_path / "payload/seed-4/stdout.log").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(BaselineCacheCorrupt, match="drifted"):
        validate_baseline_cache(published.entry_path, identity)

    identity_extra = replace(identity, environment_sha256="a" * 64)
    evidence_extra = _evidence(tmp_path / "evidence-extra", identity_extra)
    published_extra = publish_baseline_cache(
        tmp_path / "cache",
        evidence_extra,
        identity_extra,
        origin_run_id="run",
        origin_source_commit=SOURCE_COMMIT,
    ).cache
    (published_extra.entry_path / "payload/extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(BaselineCacheCorrupt, match="file set differs"):
        validate_baseline_cache(published_extra.entry_path, identity_extra)

    identity_link = replace(identity, environment_sha256="b" * 64)
    evidence_link = _evidence(tmp_path / "evidence-link", identity_link)
    entry = baseline_cache_entry_path(tmp_path / "cache-link", identity_link)
    entry.parent.mkdir(parents=True)
    entry.symlink_to(evidence_link, target_is_directory=True)
    with pytest.raises(BaselineCacheCorrupt, match="unsafe"):
        validate_baseline_cache(entry, identity_link)


def test_corrupt_entry_is_quarantined_with_failure_evidence(tmp_path: Path) -> None:
    identity = _identity()
    evidence = _evidence(tmp_path / "evidence", identity)
    entry = publish_baseline_cache(
        tmp_path / "cache",
        evidence,
        identity,
        origin_run_id="run",
        origin_source_commit=SOURCE_COMMIT,
    ).cache.entry_path
    (entry / "payload/seed-3/stderr.log").write_text("corrupt\n", encoding="utf-8")
    error = BaselineCacheCorrupt("controlled corruption")

    quarantined = quarantine_baseline_cache(entry, identity, error)

    assert quarantined is not None
    assert not entry.exists()
    assert quarantined.quarantine_path.is_dir()
    event = json.loads(quarantined.evidence_path.read_text(encoding="utf-8"))
    assert event["cache_key_sha256"] == identity.key
    assert event["test_scored"] is False


def test_interrupted_publish_leaves_no_visible_entry_or_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity()
    evidence = _evidence(tmp_path / "evidence", identity)

    def interrupt(_source: Path, _destination: Path) -> None:
        raise OSError("simulated interruption before atomic publication")

    monkeypatch.setattr(baseline_cache.os, "rename", interrupt)
    with pytest.raises(OSError, match="simulated interruption"):
        publish_baseline_cache(
            tmp_path / "cache",
            evidence,
            identity,
            origin_run_id="run",
            origin_source_commit=SOURCE_COMMIT,
        )
    assert not baseline_cache_entry_path(tmp_path / "cache", identity).exists()
    version = tmp_path / "cache/v1"
    assert not list(version.glob(".*.tmp"))


def test_materialization_never_overwrites_existing_run_evidence(tmp_path: Path) -> None:
    identity = _identity()
    evidence = _evidence(tmp_path / "evidence", identity)
    cache = publish_baseline_cache(
        tmp_path / "cache",
        evidence,
        identity,
        origin_run_id="run",
        origin_source_commit=SOURCE_COMMIT,
    ).cache
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "owned-by-run.txt"
    marker.write_text("preserve\n", encoding="utf-8")
    with pytest.raises(BaselineCacheError, match="refusing to replace"):
        materialize_baseline_cache(cache.entry_path, destination, identity)
    assert marker.read_text(encoding="utf-8") == "preserve\n"
