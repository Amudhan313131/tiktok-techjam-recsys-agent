from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rex.data.views import FeatureView, TargetView
from rex.models.bundle import create_model_bundle, validate_model_bundle
from rex.models.context_fm import ContextEnsembleFMPlugin, load_member_predictions
from rex.models.field_weighted_fm import FieldWeightedFM, FieldWeightedFMPlugin
from rex.models.official_fm import FM


def _views() -> tuple[FeatureView, TargetView]:
    rows = 16
    arrays = {
        "row_id": np.arange(rows, dtype=np.int64),
        "date": np.asarray([20220408, 20220409] * 8, dtype=np.int32),
        "user_id": np.asarray(["u1", "u2", "u3", "u4"] * 4),
        "video_id": np.asarray(["v1", "v2", "v3", "v4"] * 4),
        "author_id": np.asarray(["a1", "a1", "a2", "a2"] * 4),
        "tab": np.asarray(["1", "2", "1", "2"] * 4),
        "duration_ms": np.asarray([8000, 12000, 18000, 24000] * 4, dtype=np.float32),
        "fx__hour": np.asarray([8, 12, 18, 22] * 4, dtype=np.int8),
        "fx__is_rand": np.zeros(rows, dtype=np.int8),
    }
    targets = np.asarray([1, 0, 1, 0, 0, 1, 0, 1] * 2, dtype=np.float32)
    return (
        FeatureView(Path("fwfm-features.npz"), arrays, "0" * 64),
        TargetView(Path("fwfm-targets.npz"), targets, "1" * 64),
    )


def test_frozen_field_weights_match_standard_fm() -> None:
    features = np.asarray(
        [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]], dtype=np.int32
    )
    labels = np.asarray([1, 0, 1, 0], dtype=np.float32)
    fm = FM(12, 5, 0.001, 1e-6, 17)
    fwfm = FieldWeightedFM(
        12,
        3,
        5,
        0.001,
        1e-6,
        1e-6,
        0.0,
        17,
        learn_field_weights=False,
    )

    assert np.allclose(fm.predict(features), fwfm.predict(features), atol=1e-6, rtol=0)
    for _ in range(2):
        assert fm.step(features, labels) == pytest.approx(fwfm.step(features, labels), abs=1e-6)
    assert np.allclose(fm.predict(features), fwfm.predict(features), atol=1e-5, rtol=0)
    assert np.allclose(fm.V, fwfm.V, atol=1e-5, rtol=0)
    assert np.allclose(fm.W, fwfm.W, atol=1e-6, rtol=0)


def test_fwfm_learns_symmetric_bounded_field_pair_weights() -> None:
    features = np.asarray([[0, 3, 6], [1, 4, 7], [2, 5, 8]], dtype=np.int32)
    labels = np.asarray([1, 0, 1], dtype=np.float32)
    model = FieldWeightedFM(9, 3, 4, 0.01, 0.0, 0.0, 0.0, 3)

    model.step(features, labels)

    assert np.array_equal(model.field_weights, model.field_weights.T)
    assert np.array_equal(np.diag(model.field_weights), np.zeros(3, dtype=np.float32))
    assert not np.allclose(model.field_weights[np.triu_indices(3, 1)], 1.0)
    assert np.max(np.abs(model.field_weights)) <= 4.0


def test_fwfm_plugin_is_deterministic_unknown_safe_and_bundle_compatible(
    tmp_path: Path,
) -> None:
    features, targets = _views()
    config = {
        "k": 4,
        "epochs": 2,
        "batch_size": 4,
        "ensemble_members": 2,
        "aggregation": "mean",
        "l2_embeddings": 1e-6,
        "l2_linear": 1e-6,
        "l2_pairs": 1e-5,
    }
    plugin = FieldWeightedFMPlugin()
    first = plugin.fit(features, targets, config, 11, tmp_path / "first")
    second = plugin.fit(features, targets, config, 11, tmp_path / "second")

    apply_arrays = dict(features.arrays)
    apply_arrays["video_id"] = np.asarray(["unseen"] * features.rows)
    apply = FeatureView(features.path, apply_arrays, features.sha256)
    first_scores = plugin.predict(first, apply, config, tmp_path / "predict-first")
    second_scores = plugin.predict(second, apply, config, tmp_path / "predict-second")

    assert np.array_equal(first_scores, second_scores)
    assert first_scores.shape == (features.rows,)
    assert np.isfinite(first_scores).all()
    members, aggregate, evidence = load_member_predictions(tmp_path / "predict-first")
    assert members.shape == (2, features.rows)
    assert np.array_equal(aggregate, first_scores)
    video = next(
        item for item in evidence["feature_audit"]["fields"] if item["field"] == "video_id"
    )
    assert video["unknown_rate"] == 1.0

    report = json.loads((first.parent / "field_interactions.json").read_text(encoding="utf-8"))
    assert len(report["interactions"]) == 21
    assert report["equivalence_mode"] is False
    bundle_path = create_model_bundle(
        first.parent,
        first,
        plugin="rex.models.field_weighted_fm:FieldWeightedFMPlugin",
        seed=11,
        commit_sha="fwfm-test",
        config_sha256="2" * 64,
        data_view_sha256=features.sha256,
        features=features,
    )
    bundle = validate_model_bundle(bundle_path, expected_features=features)
    assert {path.name for path in bundle.member_paths} >= {
        "model-000.npz",
        "model-001.npz",
        "encoder.json",
        "feature_audit.json",
        "field_interactions.json",
        "training.json",
    }


def test_fwfm_rejects_misaligned_encoded_features() -> None:
    model = FieldWeightedFM(8, 3, 2, 0.001, 0.0, 0.0, 0.0, 0)
    with pytest.raises(ValueError, match="expected 3 encoded fields"):
        model.predict(np.asarray([[0, 1]], dtype=np.int32))


def test_fwfm_equivalence_plugin_matches_context_fm_plugin(tmp_path: Path) -> None:
    features, targets = _views()
    base = {
        "k": 4,
        "lr": 0.001,
        "l2": 1e-6,
        "epochs": 2,
        "batch_size": 4,
        "ensemble_members": 1,
        "aggregation": "mean",
    }
    context = ContextEnsembleFMPlugin()
    fwfm = FieldWeightedFMPlugin()
    context_artifact = context.fit(features, targets, base, 23, tmp_path / "context")
    fwfm_artifact = fwfm.fit(
        features,
        targets,
        {**base, "fm_equivalence_mode": True},
        23,
        tmp_path / "fwfm",
    )

    context_scores = context.predict(
        context_artifact,
        features,
        base,
        tmp_path / "context-predict",
    )
    fwfm_scores = fwfm.predict(
        fwfm_artifact,
        features,
        {**base, "fm_equivalence_mode": True},
        tmp_path / "fwfm-predict",
    )

    assert np.allclose(context_scores, fwfm_scores, atol=1e-5, rtol=0)
