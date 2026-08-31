from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rex.data.views import FeatureView, TargetView
from rex.models.bundle import create_model_bundle, validate_model_bundle
from rex.models.tree_classifier import PLUGIN_PATH, TreeClassifierPlugin


def _view(tmp_path: Path, *, rows: int = 48) -> FeatureView:
    index = np.arange(rows)
    day = index // 8
    arrays = {
        "row_id": index.astype(np.int64),
        "date": (20220408 + day).astype(np.int32),
        "user_id": np.asarray([f"u{value % 8}" for value in index]),
        "video_id": np.asarray([f"v{value % 11}" for value in index]),
        "author_id": np.asarray([f"a{value % 5}" for value in index]),
        "tab": np.asarray([str(value % 3) for value in index]),
        "duration_ms": (4_000 + (index % 9) * 1_900).astype(np.float32),
        "meta__category": np.asarray([f"c{value % 4}" for value in index]),
        "meta_num__quality": ((index % 7) / 7.0).astype(np.float32),
        "fx__history": ((index % 13) / 13.0).astype(np.float32),
    }
    return FeatureView(tmp_path / "features.npz", arrays, "0" * 64)


def _targets(tmp_path: Path, rows: int = 48) -> TargetView:
    index = np.arange(rows)
    labels = ((index % 5 == 0) | (index % 7 == 0)).astype(np.float32)
    return TargetView(tmp_path / "targets.npz", labels, "1" * 64)


def test_tree_classifier_inner_validation_is_strictly_later() -> None:
    dates = np.asarray([20220408, 20220409, 20220410, 20220411], dtype=np.int32)
    masks = TreeClassifierPlugin._inner_temporal_masks(dates, validation_days=2)
    assert masks is not None
    train, valid = masks
    assert dates[train].max() < dates[valid].min()
    assert dates[valid].tolist() == [20220410, 20220411]


def test_tree_classifier_only_uses_optional_fields_when_configured(tmp_path: Path) -> None:
    view = _view(tmp_path, rows=8)
    categorical, numeric = TreeClassifierPlugin._configured_fields(view, {})
    control, _ = TreeClassifierPlugin._matrix(
        view,
        categorical_fields=categorical,
        numeric_fields=numeric,
    )
    assert control.shape == (8, 7)

    config = {
        "categorical_fields": [
            "user_id",
            "video_id",
            "author_id",
            "tab",
            "meta__category",
        ],
        "numeric_fields": ["meta_num__quality", "fx__history"],
    }
    categorical, numeric = TreeClassifierPlugin._configured_fields(view, config)
    treatment, _ = TreeClassifierPlugin._matrix(
        view,
        categorical_fields=categorical,
        numeric_fields=numeric,
    )
    assert treatment.shape == (8, 10)


def test_tree_classifier_uses_train_fitted_unknown_category_code(tmp_path: Path) -> None:
    train = _view(tmp_path, rows=8)
    categorical = ("user_id",)
    matrix, vocabs = TreeClassifierPlugin._matrix(
        train,
        categorical_fields=categorical,
        numeric_fields=(),
    )
    assert matrix.shape == (8, 4)
    apply_arrays = dict(train.arrays)
    apply_arrays["user_id"] = np.asarray(["new-user"] * train.rows)
    apply = FeatureView(train.path, apply_arrays, train.sha256)
    apply_matrix, _ = TreeClassifierPlugin._matrix(
        apply,
        categorical_fields=categorical,
        numeric_fields=(),
        vocabs=vocabs,
    )
    assert np.all(apply_matrix[:, 0] == len(vocabs["user_id"]))


def test_tree_classifier_temporal_refit_and_bundle_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("lightgbm")
    view = _view(tmp_path)
    targets = _targets(tmp_path)
    config = {
        "categorical_fields": [
            "user_id",
            "video_id",
            "author_id",
            "tab",
            "meta__category",
        ],
        "numeric_fields": ["meta_num__quality", "fx__history"],
        "n_estimators": 30,
        "early_stopping_rounds": 5,
        "inner_validation_days": 2,
        "learning_rate": 0.1,
        "num_leaves": 7,
        "min_child_samples": 2,
        "min_data_per_group": 1,
        "n_jobs": 1,
    }
    plugin = TreeClassifierPlugin()
    model_path = plugin.fit(view, targets, config, 23, tmp_path / "model")
    training = json.loads((model_path.parent / "training.json").read_text(encoding="utf-8"))
    evidence = training["inner_temporal_validation"]
    assert evidence["enabled"] is True
    assert evidence["training_max_date"] < evidence["validation_min_date"]
    assert 1 <= training["selected_estimators"] <= config["n_estimators"]

    config_hash = "2" * 64
    bundle_path = create_model_bundle(
        model_path.parent,
        model_path,
        plugin=PLUGIN_PATH,
        seed=23,
        commit_sha="tree-classifier-test",
        config_sha256=config_hash,
        data_view_sha256=view.sha256,
        features=view,
    )
    bundle = validate_model_bundle(
        bundle_path,
        expected_plugin=PLUGIN_PATH,
        expected_config_sha256=config_hash,
        expected_commit_sha="tree-classifier-test",
        expected_features=view,
    )
    assert {path.name for path in bundle.member_paths} == {
        "feature_spec.json",
        "model.txt",
        "training.json",
        "vocabs.json",
    }
    first = plugin.predict(bundle.primary_path, view, config, tmp_path / "predict")
    second = plugin.predict(bundle.primary_path, view, config, tmp_path / "predict-again")
    assert first.shape == (view.rows,)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)

    drifted = dict(config)
    drifted["numeric_fields"] = ["fx__history"]
    with pytest.raises(ValueError, match="differ from the fitted"):
        plugin.predict(bundle.primary_path, view, drifted, tmp_path / "predict-drift")


def test_tree_classifier_model_parameters_are_regularized_and_deterministic() -> None:
    parameters = TreeClassifierPlugin._model_parameters({}, seed=29, n_estimators=71)
    assert parameters["objective"] == "binary"
    assert parameters["metric"] == "binary_logloss"
    assert parameters["n_estimators"] == 71
    assert parameters["reg_lambda"] == 5.0
    assert parameters["deterministic"] is True
    assert parameters["bagging_seed"] == 29
