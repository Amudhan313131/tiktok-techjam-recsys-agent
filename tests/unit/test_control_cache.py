from __future__ import annotations

import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest
from pydantic import ValidationError

from rex.control.control_cache import (
    ControlCacheConsumer,
    ControlCacheIdentity,
    ControlCacheProvenance,
    ControlPredictionCache,
    stable_environment_sha256,
)
from rex.data.manifest import sha256_file
from rex.data.views import load_feature_view
from rex.execution.artifacts import load_prediction_artifact, write_prediction_artifact
from rex.models.bundle import create_model_bundle


@dataclass(frozen=True)
class _Source:
    identity: ControlCacheIdentity
    provenance: ControlCacheProvenance
    config: Path
    train_features: Path
    train_targets: Path
    apply_features: Path
    bundle: Path
    predictions: Path


def _write_views(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    rows = 8
    common = {
        "row_id": np.arange(rows, dtype=np.int64),
        "date": np.asarray([20220408, 20220408, 20220409, 20220409] * 2),
        "user_id": np.asarray(["u1", "u1", "u2", "u2"] * 2),
        "video_id": np.asarray(["v1", "v2", "v3", "v4"] * 2),
        "author_id": np.asarray(["a1", "a2", "a3", "a4"] * 2),
        "tab": np.asarray(["1", "1", "2", "2"] * 2),
        "duration_ms": np.arange(rows, dtype=np.float32) + 1,
    }
    train = root / "train.npz"
    target = root / "target.npz"
    apply = root / "apply.npz"
    np.savez_compressed(train, **common)
    np.savez_compressed(target, long_view=np.asarray([0, 1] * 4, dtype=np.float32))
    np.savez_compressed(apply, **common)
    return train, target, apply


def _source(tmp_path: Path) -> _Source:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "control.yaml"
    config.write_text("plugin: fake:Control\nepochs: 1\n", encoding="utf-8")
    train, target, apply = _write_views(tmp_path / "views")
    identity = ControlCacheIdentity.from_paths(
        plugin="fake:Control",
        config_path=config,
        source_commit="source-commit-123",
        environment_sha256="e" * 64,
        seed=7,
        rung="full",
        split="shadow",
        fold="A",
        partition_sha256="a" * 64,
        train_feature_path=train,
        train_target_path=target,
        apply_feature_path=apply,
    )
    model_root = tmp_path / "source-model"
    model_root.mkdir()
    primary = model_root / "model.bin"
    primary.write_bytes(b"immutable-control-model")
    bundle = create_model_bundle(
        model_root,
        primary,
        plugin=identity.plugin,
        seed=identity.seed,
        commit_sha=identity.source_commit,
        config_sha256=identity.config_sha256,
        data_view_sha256=identity.train_feature_sha256,
        features=load_feature_view(train),
    )
    predictions = write_prediction_artifact(
        tmp_path / "source-predictions.npz",
        apply,
        np.linspace(0.1, 0.8, 8),
    )
    provenance = ControlCacheProvenance(
        producer_run_id="producer-run",
        producer_experiment_id="producer-experiment",
        fit_attempt_id="fit-attempt",
        predict_attempt_id="predict-attempt",
        fit_request_sha256="1" * 64,
        predict_request_sha256="2" * 64,
    )
    return _Source(
        identity,
        provenance,
        config,
        train,
        target,
        apply,
        bundle,
        predictions,
    )


def _publish(cache: ControlPredictionCache, source: _Source, provenance=None):
    return cache.publish(
        source.identity,
        bundle_path=source.bundle,
        prediction_path=source.predictions,
        train_feature_path=source.train_features,
        train_target_path=source.train_targets,
        apply_feature_path=source.apply_features,
        provenance=provenance or source.provenance,
    )


def _import(cache: ControlPredictionCache, source: _Source, destination: Path):
    return cache.import_entry(
        source.identity,
        train_feature_path=source.train_features,
        train_target_path=source.train_targets,
        apply_feature_path=source.apply_features,
        destination_dir=destination,
        consumer=ControlCacheConsumer(
            run_id="consumer-run",
            experiment_id="consumer-experiment",
            rung="full",
            fold="A",
        ),
    )


def _make_writable(root: Path) -> None:
    root.chmod(0o755)
    for path in root.rglob("*"):
        if not path.is_symlink():
            path.chmod(0o755 if path.is_dir() else 0o644)


def test_identity_and_environment_are_path_independent_and_test_is_forbidden(
    tmp_path: Path,
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    source = _source(first)
    second.mkdir()
    copied = {}
    for name, original in (
        ("config", source.config),
        ("train", source.train_features),
        ("target", source.train_targets),
        ("apply", source.apply_features),
    ):
        copied[name] = second / original.name
        shutil.copy2(original, copied[name])
    replay = ControlCacheIdentity.from_paths(
        plugin=source.identity.plugin,
        config_path=copied["config"],
        source_commit=source.identity.source_commit,
        environment_sha256=source.identity.environment_sha256,
        seed=source.identity.seed,
        rung="full",
        split="shadow",
        fold="A",
        partition_sha256=source.identity.partition_sha256,
        train_feature_path=copied["train"],
        train_target_path=copied["target"],
        apply_feature_path=copied["apply"],
    )
    assert replay.cache_key == source.identity.cache_key

    lock_a = first / "requirements-lock.txt"
    lock_b = second / "requirements-lock.txt"
    project_a = first / "pyproject.toml"
    project_b = second / "pyproject.toml"
    lock_a.write_text("package==1 --hash=sha256:abc\n", encoding="utf-8")
    project_a.write_text("[project]\nname='cache-test'\n", encoding="utf-8")
    shutil.copy2(lock_a, lock_b)
    shutil.copy2(project_a, project_b)
    assert stable_environment_sha256(
        requirements_lock=lock_a, pyproject=project_a, python_executable=sys.executable
    ) == stable_environment_sha256(
        requirements_lock=lock_b, pyproject=project_b, python_executable=sys.executable
    )

    invalid = source.identity.model_dump(mode="json")
    invalid["split"] = "test"
    with pytest.raises(ValidationError):
        ControlCacheIdentity.model_validate(invalid)


def test_each_scientific_identity_field_invalidates_the_cache_key(tmp_path: Path):
    identity = _source(tmp_path).identity
    changes = {
        "config_sha256": "3" * 64,
        "source_commit": "different-source-commit",
        "environment_sha256": "4" * 64,
        "seed": identity.seed + 1,
        "fold": "B",
        "partition_sha256": "5" * 64,
        "train_feature_sha256": "6" * 64,
        "train_target_sha256": "7" * 64,
        "apply_feature_sha256": "8" * 64,
        "feature_provenance_sha256": ("9" * 64,),
    }
    payload = identity.model_dump(mode="json")
    for field, value in changes.items():
        changed = ControlCacheIdentity.model_validate({**payload, field: value})
        assert changed.cache_key != identity.cache_key, field


def test_publish_and_import_create_verified_run_local_copy_with_provenance(tmp_path: Path):
    source = _source(tmp_path / "source")
    cache = ControlPredictionCache(tmp_path / "shared-cache")
    published = _publish(cache, source)
    imported = _import(cache, source, tmp_path / "run/evidence/control")
    assert imported is not None
    assert imported.cache_key == published.cache_key
    assert imported.root != published.root
    assert imported.bundle_path.is_relative_to(imported.root)
    assert imported.prediction_path.is_relative_to(imported.root)
    arrays = load_prediction_artifact(imported.prediction_path, source.apply_features)
    assert np.allclose(arrays["score"], np.linspace(0.1, 0.8, 8))
    evidence = json.loads(imported.evidence_path.read_text(encoding="utf-8"))
    assert evidence["producer"]["producer_run_id"] == "producer-run"
    assert evidence["consumer"]["run_id"] == "consumer-run"
    assert evidence["cache_manifest_sha256"] == sha256_file(imported.manifest_path)
    assert evidence["test_prediction_created"] is False
    assert evidence["test_scored"] is False

    _make_writable(published.root)
    published.prediction_path.write_bytes(b"global cache later became corrupt")
    assert load_prediction_artifact(imported.prediction_path, source.apply_features)[
        "score"
    ].shape == (8,)


@pytest.mark.parametrize(
    "mutation",
    ["tamper", "symlink", "extra-file", "traversal", "extra-manifest-field"],
)
def test_corrupt_entries_are_never_loaded_and_are_quarantined(tmp_path: Path, mutation: str):
    source = _source(tmp_path / "source")
    cache = ControlPredictionCache(tmp_path / "shared-cache")
    entry = _publish(cache, source)
    _make_writable(entry.root)
    if mutation == "tamper":
        entry.prediction_path.write_bytes(b"tampered")
    elif mutation == "symlink":
        entry.prediction_path.unlink()
        entry.prediction_path.symlink_to(source.predictions)
    elif mutation == "extra-file":
        (entry.root / "unexpected.txt").write_text("extra", encoding="utf-8")
    else:
        payload = json.loads(entry.manifest_path.read_text(encoding="utf-8"))
        if mutation == "traversal":
            payload["artifacts"][0]["path"] = "../escape"
        else:
            payload["unexpected"] = True
        entry.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    assert _import(cache, source, tmp_path / "run/control") is None
    assert not entry.root.exists()
    quarantined = list((cache.root / "quarantine").iterdir())
    events = list((cache.root / "quarantine-events").glob("*.json"))
    assert len(quarantined) == 1
    assert len(events) == 1
    assert source.identity.cache_key in quarantined[0].name


def test_concurrent_publication_has_one_complete_atomic_winner(tmp_path: Path):
    source = _source(tmp_path / "source")
    cache = ControlPredictionCache(tmp_path / "shared-cache")
    barrier = Barrier(2)

    def publish(index: int):
        barrier.wait()
        provenance = source.provenance.model_copy(update={"producer_run_id": f"producer-{index}"})
        return _publish(cache, source, provenance)

    with ThreadPoolExecutor(max_workers=2) as executor:
        entries = list(executor.map(publish, (1, 2)))
    assert entries[0].root == entries[1].root
    assert entries[0].manifest_path.read_bytes() == entries[1].manifest_path.read_bytes()
    assert entries[0].manifest.provenance.producer_run_id in {"producer-1", "producer-2"}
    assert not list(entries[0].root.parent.glob("*.tmp"))
    imported = _import(cache, source, tmp_path / "run/control")
    assert imported is not None


def test_corrupt_run_local_copy_is_replaced_from_valid_shared_entry(tmp_path: Path):
    source = _source(tmp_path / "source")
    cache = ControlPredictionCache(tmp_path / "shared-cache")
    _publish(cache, source)
    destination = tmp_path / "run/control"
    first = _import(cache, source, destination)
    assert first is not None
    _make_writable(first.root)
    first.prediction_path.write_bytes(b"broken local copy")
    repaired = _import(cache, source, destination)
    assert repaired is not None
    assert load_prediction_artifact(repaired.prediction_path, source.apply_features)[
        "score"
    ].shape == (8,)
    assert list(destination.parent.glob(".control.corrupt-*"))
