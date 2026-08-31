from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rex.data.views import FeatureView
from rex.features.categorical_crosses import (
    RARE_INDEX,
    UNKNOWN_INDEX,
    CategoricalCrossSpec,
    CategoricalCrossState,
    apply_categorical_crosses,
    fit_categorical_crosses,
    fit_transform_categorical_crosses,
)
from rex.features.recipes import USER_TAB_CROSS, materialize_feature_recipe


def _view(users: list[str], tabs: list[int]) -> FeatureView:
    rows = len(users)
    return FeatureView(
        Path("crosses.npz"),
        {
            "row_id": np.arange(rows, dtype=np.int64),
            "date": np.full(rows, 20220408, dtype=np.int32),
            "user_id": np.asarray(users),
            "video_id": np.asarray([f"v{index}" for index in range(rows)]),
            "author_id": np.asarray(["a"] * rows),
            "tab": np.asarray(tabs, dtype=np.int16),
            "duration_ms": np.full(rows, 1000, dtype=np.float32),
        },
        "0" * 64,
    )


def test_crosses_use_distinct_deterministic_rare_and_unknown_buckets() -> None:
    train = _view(["u1", "u2", "u1", "u3"], [1, 1, 1, 2])
    spec = CategoricalCrossSpec("user_tab", "user_id", "tab", min_count=2)
    state, train_bundle = fit_transform_categorical_crosses(train, [spec])

    assert train_bundle.arrays["user_tab"].tolist() == [2, RARE_INDEX, 2, RARE_INDEX]
    assert train_bundle.provenance["user_tab"]["vocabulary_size"] == 1
    apply = _view(["u2", "new-user", "u1"], [1, 1, 9])
    encoded = apply_categorical_crosses(apply, state).arrays["user_tab"]
    assert encoded.tolist() == [RARE_INDEX, UNKNOWN_INDEX, UNKNOWN_INDEX]


def test_cross_vocabulary_is_stable_under_training_row_reordering() -> None:
    spec = CategoricalCrossSpec("user_tab", "user_id", "tab", min_count=1)
    first = fit_categorical_crosses(_view(["z", "a", "m"], [3, 1, 2]), [spec])
    second = fit_categorical_crosses(_view(["m", "z", "a"], [2, 3, 1]), [spec])

    assert first.to_json() == second.to_json()


def test_cross_state_json_round_trip_preserves_encoding() -> None:
    train = _view(["u1", "u1", "u1", "u2"], [1, 1, 1, 2])
    spec = CategoricalCrossSpec("user_tab", "user_id", "tab", min_count=2)
    state = fit_categorical_crosses(train, [spec])
    restored = CategoricalCrossState.from_json(
        json.loads(json.dumps(state.to_json(), sort_keys=True))
    )

    assert np.array_equal(
        apply_categorical_crosses(train, state).arrays["user_tab"],
        apply_categorical_crosses(train, restored).arrays["user_tab"],
    )


def test_crosses_fail_closed_on_invalid_contracts() -> None:
    train = _view(["u1"], [1])
    with pytest.raises(ValueError, match="different columns"):
        CategoricalCrossSpec("bad", "tab", "tab")
    with pytest.raises(ValueError, match="requires feature column"):
        fit_categorical_crosses(
            train,
            [CategoricalCrossSpec("bad", "missing", "tab")],
        )
    with pytest.raises(ValueError, match="names must be unique"):
        fit_categorical_crosses(
            train,
            [
                CategoricalCrossSpec("same", "user_id", "tab"),
                CategoricalCrossSpec("same", "video_id", "tab"),
            ],
        )


def test_cross_recipe_materializes_train_fitted_apply_values(tmp_path: Path) -> None:
    train = _view(["u1", "u1", "u1", "u2"], [1, 1, 1, 2])
    apply = _view(["u1", "u3"], [1, 9])
    train_path = tmp_path / "train.npz"
    apply_path = tmp_path / "apply.npz"
    target_path = tmp_path / "targets.npz"
    np.savez_compressed(train_path, **train.arrays)
    np.savez_compressed(apply_path, **apply.arrays)
    np.savez_compressed(
        target_path, long_view=np.asarray([1, 0, 1, 0], dtype=np.float32)
    )

    artifact = materialize_feature_recipe(
        USER_TAB_CROSS,
        train_path,
        target_path,
        apply_path,
        tmp_path / "cache",
    )
    with np.load(artifact.train_features, allow_pickle=False) as saved:
        assert "fx__user_tab_cross" in saved.files
        assert saved["fx__user_tab_cross"].tolist() == [2, 2, 2, 0]
    with np.load(artifact.apply_features, allow_pickle=False) as saved:
        assert saved["fx__user_tab_cross"].tolist() == [2, 1]
