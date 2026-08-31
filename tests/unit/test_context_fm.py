from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rex.data.views import FeatureView, TargetView
from rex.models.context_fm import ContextCategoricalEncoder, ContextEnsembleFMPlugin
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
    first_scores = plugin.predict(first, features, config, tmp_path / "predict-first")
    second_scores = plugin.predict(second, features, config, tmp_path / "predict-second")

    assert np.array_equal(first_scores, second_scores)
    assert np.isfinite(first_scores).all()
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
