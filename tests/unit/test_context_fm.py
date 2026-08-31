from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rex.data.views import FeatureView, TargetView
from rex.models.context_fm import (
    ContextCategoricalEncoder,
    ContextEnsembleFMPlugin,
    load_member_predictions,
    reconstruct_member_predictions,
)
from rex.models.bundle import create_model_bundle, validate_model_bundle


def _views() -> tuple[FeatureView, TargetView]:
    rows = 12
    arrays = {
        "row_id": np.arange(rows, dtype=np.int64),
        "date": np.asarray([20220408, 20220409, 20220410] * 4, dtype=np.int32),
        "user_id": np.repeat(np.asarray(["u1", "u2", "u3", "u4"]), 3),
        "video_id": np.asarray(["v1", "v2", "v3"] * 4),
        "author_id": np.asarray(["a1", "a2", "a3"] * 4),
        "tab": np.asarray(["1", "1", "2"] * 4),
        "duration_ms": np.asarray([8000, 15000, 24000] * 4, dtype=np.float32),
        "fx__hour": np.asarray([8, 12, 22] * 4, dtype=np.int8),
        "fx__is_rand": np.asarray([0, 0, 1] * 4, dtype=np.int8),
    }
    return (
        FeatureView(Path("context-features.npz"), arrays, "0" * 64),
        TargetView(
            Path("context-targets.npz"),
            np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0], dtype=np.float32),
            "1" * 64,
        ),
    )


def test_context_ensemble_round_trip_is_deterministic(tmp_path: Path) -> None:
    features, targets = _views()
    config = {
        "k": 4,
        "lr": 0.001,
        "epochs": 2,
        "batch_size": 4,
        "ensemble_members": 3,
        "aggregation": "mean",
    }
    plugin = ContextEnsembleFMPlugin()
    first = plugin.fit(features, targets, config, 7, tmp_path / "first")
    second = plugin.fit(features, targets, config, 7, tmp_path / "second")
    prediction_root = tmp_path / "predict-first"
    first_scores = plugin.predict(first, features, config, prediction_root)
    second_scores = plugin.predict(second, features, config, tmp_path / "predict-second")

    assert np.array_equal(first_scores, second_scores)
    assert np.isfinite(first_scores).all()
    member_scores, persisted, evidence = load_member_predictions(prediction_root)
    assert member_scores.shape == (3, features.rows)
    assert member_scores.dtype == np.float32
    assert np.array_equal(first_scores, persisted)
    assert np.array_equal(first_scores, np.mean(member_scores, axis=0).astype(np.float64))
    assert np.array_equal(first_scores, reconstruct_member_predictions(member_scores, "mean"))
    assert evidence["member_names"] == [
        "model-000.npz",
        "model-001.npz",
        "model-002.npz",
    ]
    training = json.loads((first.parent / "training.json").read_text(encoding="utf-8"))
    assert training["members"] == ["model-000.npz", "model-001.npz", "model-002.npz"]
    bundle_path = create_model_bundle(
        first.parent,
        first,
        plugin="rex.models.context_fm:ContextEnsembleFMPlugin",
        seed=7,
        commit_sha="context-test",
        config_sha256="2" * 64,
        data_view_sha256=features.sha256,
        features=features,
    )
    bundle = validate_model_bundle(bundle_path, expected_features=features)
    assert {path.name for path in bundle.member_paths} >= {
        "model-000.npz",
        "model-001.npz",
        "model-002.npz",
        "encoder.json",
        "training.json",
    }


def test_context_encoder_requires_sanitized_context_fields() -> None:
    features, _ = _views()
    missing = FeatureView(
        features.path,
        {name: values for name, values in features.arrays.items() if name != "fx__hour"},
        features.sha256,
    )
    with pytest.raises(ValueError, match="fx__hour"):
        ContextCategoricalEncoder.fit(missing)


def test_context_ensemble_rejects_unbounded_member_count(tmp_path: Path) -> None:
    features, targets = _views()
    with pytest.raises(ValueError, match="between 1 and 7"):
        ContextEnsembleFMPlugin().fit(
            features,
            targets,
            {"ensemble_members": 8},
            0,
            tmp_path,
        )


def test_context_encoder_audits_constant_and_unknown_fields(tmp_path: Path) -> None:
    features, targets = _views()
    arrays = dict(features.arrays)
    arrays["fx__is_rand"] = np.zeros(features.rows, dtype=np.int8)
    train = FeatureView(features.path, arrays, features.sha256)
    plugin = ContextEnsembleFMPlugin()
    config = {"epochs": 1, "batch_size": 4, "ensemble_members": 1}
    artifact = plugin.fit(train, targets, config, 0, tmp_path / "model")

    training_audit = json.loads(
        (artifact.parent / "feature_audit.json").read_text(encoding="utf-8")
    )
    is_rand = next(item for item in training_audit["fields"] if item["field"] == "is_rand")
    assert is_rand == {
        "constant": True,
        "constant_value": "0",
        "field": "is_rand",
        "unique_count": 1,
        "unknown_count": 0,
        "unknown_rate": 0.0,
    }

    apply_arrays = dict(arrays)
    apply_arrays["user_id"] = np.asarray(["never-seen"] * features.rows)
    apply = FeatureView(features.path, apply_arrays, features.sha256)
    plugin.predict(artifact, apply, config, tmp_path / "prediction")
    _, _, evidence = load_member_predictions(tmp_path / "prediction")
    user = next(item for item in evidence["feature_audit"]["fields"] if item["field"] == "user_id")
    assert user["unknown_count"] == features.rows
    assert user["unknown_rate"] == 1.0


def test_context_encoder_supports_explicit_engineered_categorical_fields() -> None:
    features, _ = _views()
    arrays = dict(features.arrays)
    arrays["fx__user_tab"] = np.arange(features.rows, dtype=np.int32) % 3
    extended = FeatureView(features.path, arrays, features.sha256)
    fields = ("user_id", "video_id", "user_tab")
    encoder = ContextCategoricalEncoder.fit(extended, fields)

    assert encoder.fields == fields
    assert encoder.transform(extended).shape == (features.rows, 3)
    assert ContextCategoricalEncoder.from_json(encoder.to_json()).fields == fields


def test_member_prediction_reconstruction_rejects_non_finite_values() -> None:
    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        reconstruct_member_predictions(np.asarray([[1.0, np.nan]]), "mean")
