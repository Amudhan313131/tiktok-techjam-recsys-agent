from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from rex.contracts import RunRequest
from rex.data.bootstrap import build_split_views, default_data_dir
from rex.data.firewall import (
    CapabilityViolation,
    assert_no_test_target_artifact,
    validate_worker_request,
)
from rex.data.manifest import ManifestError, repo_root, sha256_file, verify_raw_dataset
from rex.data.shadow_views import materialize_cheap_view, materialize_shadow_folds
from rex.data.views import load_feature_view
from rex.evaluation.diagnostics import standard_segments
from rex.evaluation.baseline import reproduce_fm_seed, reproduce_item_popularity, reproduce_random
from rex.features.recipes import (
    AUTHOR_DURATION_AFFINITY,
    CANDIDATE_HISTORY,
    HISTORY_LENGTH,
    RECENCY_HISTORY,
    REPEAT_EXPOSURE,
    VIDEO_STATISTICS,
    control_recipe,
    materialize_feature_recipe,
)


LOG_HEADER = [
    "user_id",
    "video_id",
    "date",
    "long_view",
    "duration_ms",
    "tab",
    "hourmin",
    "is_rand",
]


def test_repo_root_honors_the_verified_controller_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_clone = tmp_path / "controller-clone"
    source_clone.mkdir()
    monkeypatch.setenv("REX_SOURCE_ROOT", str(source_clone))

    assert repo_root() == source_clone.resolve()


def test_default_data_dir_honors_the_read_only_data_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("REX_DATA_ROOT", str(data_root))

    assert default_data_dir() == data_root.resolve()


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _tiny_raw(root: Path, *, test_label: int = 0) -> dict[str, object]:
    root.mkdir()
    video = root / "video_features_basic_pure.csv"
    earlier = root / "log_standard_4_08_to_4_21_pure.csv"
    later = root / "log_standard_4_22_to_5_08_pure.csv"
    _write_csv(video, ["video_id", "author_id"], [["v1", "a1"], ["v2", "a2"]])
    _write_csv(
        earlier,
        LOG_HEADER,
        [["u1", "v1", 1, 1, 10, 1, 900, 0], ["u2", "v2", 2, 0, 20, 1, 2200, 1]],
    )
    _write_csv(
        later,
        LOG_HEADER,
        [
            ["u1", "v2", 3, 1, 20, 1, 1300, 0],
            ["u1", "v1", 4, test_label, 10, 1, 2300, 1],
        ],
    )
    return {
        "label": "long_view",
        "metrics": ["GAUC", "nDCG@5"],
        "splits": {
            "train": {
                "split_start": 1,
                "split_end": 2,
                "observed_date_min": 1,
                "observed_date_max": 2,
                "rows": 2,
            },
            "valid": {
                "split_start": 3,
                "split_end": 3,
                "observed_date_min": 3,
                "observed_date_max": 3,
                "rows": 1,
            },
            "test": {
                "split_start": 4,
                "split_end": 4,
                "observed_date_min": 4,
                "observed_date_max": 4,
                "rows": 1,
            },
        },
    }


def test_raw_dataset_verifier_checks_bytes_schema_and_counts(tmp_path: Path) -> None:
    data = tmp_path / "data"
    benchmark = _tiny_raw(data)
    raw_files = {}
    for path in sorted(data.iterdir()):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            rows = sum(1 for _ in reader)
        raw_files[path.name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "data_rows": rows,
            "header": header,
        }
    benchmark["raw_files"] = raw_files
    manifest = tmp_path / "benchmark.json"
    manifest.write_text(json.dumps(benchmark), encoding="utf-8")
    result = verify_raw_dataset(data, benchmark_path=manifest)
    assert set(result.files) == set(raw_files)
    with (data / "video_features_basic_pure.csv").open("a", encoding="utf-8") as handle:
        handle.write("v3,a3\n")
    with pytest.raises(ManifestError):
        verify_raw_dataset(data, benchmark_path=manifest)


def test_test_label_poison_does_not_change_feature_view(tmp_path: Path) -> None:
    left_data = tmp_path / "left-data"
    right_data = tmp_path / "right-data"
    benchmark = _tiny_raw(left_data, test_label=0)
    _tiny_raw(right_data, test_label=1)
    left = build_split_views(left_data, tmp_path / "left", benchmark=benchmark)
    right = build_split_views(right_data, tmp_path / "right", benchmark=benchmark)
    assert left["test"]["feature_sha256"] == right["test"]["feature_sha256"]
    assert left["test"]["target_path"] is None
    assert right["test"]["target_path"] is None
    assert_no_test_target_artifact(tmp_path / "left")
    train = load_feature_view(tmp_path / "left/train_features.npz")
    assert train.arrays["fx__hour"].tolist() == [9, 22]
    assert train.arrays["fx__is_rand"].tolist() == [0, 1]


def test_firewall_rejects_test_fit_request(feature_target_paths, tmp_path: Path) -> None:
    features, targets = feature_target_paths
    request = RunRequest(
        run_id="r",
        experiment_id="e",
        attempt_id="a",
        commit_sha="abc",
        plugin="fm",
        operation="fit",
        config_path=str(tmp_path / "config.json"),
        config_sha256="0" * 64,
        seed=0,
        rung="full",
        split="test",
        feature_view_path=str(features),
        target_view_path=str(targets),
        output_dir=str(tmp_path / "out"),
        deadline_epoch_ms=1,
        timeout_seconds=1,
        data_view_sha256="0" * 64,
        environment_sha256="0" * 64,
    )
    with pytest.raises(CapabilityViolation):
        validate_worker_request(request)


def _temporal_source(tmp_path: Path) -> tuple[Path, Path]:
    dates = np.asarray(
        [
            20220408,
            20220409,
            20220410,
            20220411,
            20220412,
            20220413,
            20220414,
            20220415,
            20220416,
            20220417,
            20220418,
            20220419,
            20220420,
            20220421,
        ]
    )
    users = np.asarray(["u1", "u2"] * 7)
    feature = tmp_path / "train.npz"
    target = tmp_path / "target.npz"
    np.savez_compressed(
        feature,
        row_id=np.arange(len(dates), dtype=np.int64),
        date=dates,
        user_id=users,
        video_id=np.asarray([f"v{i % 4}" for i in range(len(dates))]),
        author_id=np.asarray([f"a{i % 3}" for i in range(len(dates))]),
        tab=np.asarray(["1"] * len(dates)),
        duration_ms=np.asarray([10_000 + i * 100 for i in range(len(dates))], dtype=np.float32),
    )
    np.savez_compressed(target, long_view=np.asarray([i % 2 for i in range(len(dates))]))
    return feature, target


def test_shadow_and_cheap_views_are_cached_and_keep_complete_users(tmp_path: Path) -> None:
    feature, target = _temporal_source(tmp_path)
    folds = materialize_shadow_folds(feature, target, tmp_path / "folds")
    assert [fold.name for fold in folds] == ["A", "B", "C"]
    replay = materialize_shadow_folds(feature, target, tmp_path / "folds")
    assert [fold.identity_sha256 for fold in replay] == [fold.identity_sha256 for fold in folds]
    cheap = materialize_cheap_view(folds[0], tmp_path / "cheap", fraction=0.5, seed=4)
    valid = load_feature_view(cheap.valid_features)
    train = load_feature_view(cheap.train_features)
    selected = set(valid.arrays["user_id"])
    assert set(train.arrays["user_id"]) <= selected
    parent_valid = load_feature_view(folds[0].valid_features)
    for user in selected:
        assert int(np.sum(valid.arrays["user_id"] == user)) == int(
            np.sum(parent_valid.arrays["user_id"] == user)
        )


@pytest.mark.parametrize(
    "recipe",
    [
        VIDEO_STATISTICS,
        HISTORY_LENGTH,
        AUTHOR_DURATION_AFFINITY,
        CANDIDATE_HISTORY,
        REPEAT_EXPOSURE,
        RECENCY_HISTORY,
    ],
)
def test_feature_recipe_cache_is_deterministic(
    recipe, feature_target_paths, tmp_path: Path
) -> None:
    features, targets = feature_target_paths
    first = materialize_feature_recipe(recipe, features, targets, features, tmp_path / "cache")
    second = materialize_feature_recipe(recipe, features, targets, features, tmp_path / "cache")
    assert first.identity_sha256 == second.identity_sha256
    output = load_feature_view(first.apply_features)
    for name in recipe.output_features:
        assert f"fx__{name}" in output.arrays


def test_control_recipe_materializes_zero_ablation(feature_target_paths, tmp_path: Path) -> None:
    features, targets = feature_target_paths
    artifact = materialize_feature_recipe(
        control_recipe(VIDEO_STATISTICS), features, targets, features, tmp_path / "cache"
    )
    output = load_feature_view(artifact.apply_features)
    assert not np.any(output.arrays["fx__video_target_rate"])
    assert not np.any(output.arrays["fx__author_target_rate"])


def test_standard_segments_cover_each_requested_dimension(feature_target_paths) -> None:
    features, _ = feature_target_paths
    view = load_feature_view(features)
    segments = standard_segments(view, history=view)
    assert {"user:warm", "video:warm", "history:1-4", "repeat:seen"} <= set(segments)
    assert any(name.startswith("duration:") for name in segments)
    assert any(name.startswith("tab:") for name in segments)
    assert any(name.startswith("date:") for name in segments)


def test_valid_only_baselines_persist_evidence(feature_target_paths, tmp_path: Path) -> None:
    features, targets = feature_target_paths
    views = tmp_path / "views"
    (views / "label_vault").mkdir(parents=True)
    feature_copy = views / "train_features.npz"
    valid_copy = views / "valid_features.npz"
    target_copy = views / "label_vault/train_targets.npz"
    valid_target_copy = views / "label_vault/valid_targets.npz"
    feature_copy.write_bytes(features.read_bytes())
    valid_copy.write_bytes(features.read_bytes())
    target_copy.write_bytes(targets.read_bytes())
    valid_target_copy.write_bytes(targets.read_bytes())
    random = reproduce_random(views, evidence_dir=tmp_path / "random")
    popularity = reproduce_item_popularity(views, evidence_dir=tmp_path / "pop")
    fm = reproduce_fm_seed(
        views,
        seed=0,
        epochs=1,
        patience=1,
        batch_size=4,
        evidence_dir=tmp_path / "fm",
    )
    assert random.split == popularity.split == fm.metrics.split == "valid"
    for path in (
        tmp_path / "random/predictions.npz",
        tmp_path / "random/environment.json",
        tmp_path / "random/command.json",
        tmp_path / "pop/predictions.npz",
        tmp_path / "fm/model.npz",
        tmp_path / "fm/model_bundle.json",
        tmp_path / "fm/predictions.npz",
        tmp_path / "fm/stdout.log",
        tmp_path / "fm/stderr.log",
        tmp_path / "fm/telemetry.json",
    ):
        assert path.is_file()
