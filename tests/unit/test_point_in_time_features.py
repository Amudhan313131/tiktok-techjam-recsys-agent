from __future__ import annotations

import numpy as np
import pytest

from rex.features.candidate_recency import (
    apply_candidate_recency_state,
    candidate_recency_features,
    fit_candidate_recency_state,
)
from rex.features.temporal_order import strict_timestamp_groups
from rex.features.multifeedback_history import (
    apply_multifeedback_state,
    fit_multifeedback_state,
    multifeedback_history_features,
)
from rex.data.views import load_feature_view
from rex.features.recipes import (
    CANDIDATE_RECENCY_BUCKETS,
    RICH_TEMPORAL_HISTORY,
    materialize_feature_recipe,
)


def _arrays():
    return {
        "users": np.asarray(["u", "u", "u", "u"]),
        "videos": np.asarray(["v1", "v1", "v1", "v2"]),
        "authors": np.asarray(["a", "a", "a", "b"]),
        "times": np.asarray([3000, 1000, 1000, 2000], dtype=np.int64),
        "keys": np.asarray(["f:3", "f:1", "f:2", "f:4"]),
        "labels": np.asarray([0, 1, 0, 1], dtype=np.float32),
    }


def test_timestamp_groups_are_sorted_and_equal_times_are_atomic() -> None:
    values = _arrays()
    groups = list(strict_timestamp_groups(values["times"], values["keys"]))
    assert [group.tolist() for group in groups] == [[1, 2], [3], [0]]


def test_timestamp_keys_must_be_unique() -> None:
    with pytest.raises(ValueError, match="globally unique"):
        list(
            strict_timestamp_groups(
                np.asarray([1, 2]), np.asarray(["duplicate", "duplicate"])
            )
        )


def test_equal_timestamp_rows_cannot_observe_each_others_outcomes() -> None:
    values = _arrays()
    first = candidate_recency_features(
        values["users"],
        values["videos"],
        values["authors"],
        values["times"],
        values["keys"],
        values["labels"],
    )
    changed = values["labels"].copy()
    changed[1:3] = 1.0 - changed[1:3]
    second = candidate_recency_features(
        values["users"],
        values["videos"],
        values["authors"],
        values["times"],
        values["keys"],
        changed,
    )
    for name in first.arrays:
        np.testing.assert_array_equal(first.arrays[name][1:3], second.arrays[name][1:3])
    assert first.arrays["pt_user_count_h1p0"][3] > 1.99


def test_future_outcomes_do_not_change_earlier_rows() -> None:
    values = _arrays()
    first = candidate_recency_features(
        values["users"],
        values["videos"],
        values["authors"],
        values["times"],
        values["keys"],
        values["labels"],
    )
    changed = values["labels"].copy()
    changed[0] = 1.0 - changed[0]
    second = candidate_recency_features(
        values["users"],
        values["videos"],
        values["authors"],
        values["times"],
        values["keys"],
        changed,
    )
    earlier = np.asarray([1, 2, 3])
    for name in first.arrays:
        np.testing.assert_array_equal(first.arrays[name][earlier], second.arrays[name][earlier])


def test_frozen_apply_state_does_not_accept_apply_outcomes() -> None:
    values = _arrays()
    state = fit_candidate_recency_state(
        values["users"],
        values["videos"],
        values["authors"],
        values["times"],
        values["keys"],
        values["labels"],
    )
    first = apply_candidate_recency_state(
        np.asarray(["u", "cold"]),
        np.asarray(["v1", "new"]),
        np.asarray(["a", "new"]),
        np.asarray([4000, 4000], dtype=np.int64),
        state,
    )
    second = apply_candidate_recency_state(
        np.asarray(["u", "cold"]),
        np.asarray(["v1", "new"]),
        np.asarray(["a", "new"]),
        np.asarray([4000, 4000], dtype=np.int64),
        state,
    )
    for name in first.arrays:
        np.testing.assert_array_equal(first.arrays[name], second.arrays[name])
    assert first.arrays["pt_user_video_count"].tolist() == [3, 0]


def _feedback(values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "is_click": values.copy(),
        "is_like": values.copy(),
        "is_follow": values.copy(),
        "is_hate": 1.0 - values,
        "long_view": values.copy(),
    }


def test_multifeedback_excludes_current_and_equal_timestamp_labels() -> None:
    values = _arrays()
    first = multifeedback_history_features(
        values["users"],
        values["videos"],
        values["authors"],
        values["times"],
        values["keys"],
        _feedback(values["labels"]),
    )
    changed = values["labels"].copy()
    changed[1:3] = 1.0 - changed[1:3]
    second = multifeedback_history_features(
        values["users"],
        values["videos"],
        values["authors"],
        values["times"],
        values["keys"],
        _feedback(changed),
    )
    for name in first.arrays:
        np.testing.assert_array_equal(first.arrays[name][1:3], second.arrays[name][1:3])
    assert first.arrays["pt_feedback_user_count"][3] == 2


def test_multifeedback_frozen_apply_uses_no_apply_labels() -> None:
    values = _arrays()
    state = fit_multifeedback_state(
        values["users"],
        values["videos"],
        values["authors"],
        values["times"],
        values["keys"],
        _feedback(values["labels"]),
    )
    bundle = apply_multifeedback_state(
        np.asarray(["u", "cold"]),
        np.asarray(["v1", "new"]),
        np.asarray(["a", "new"]),
        state,
    )
    assert bundle.arrays["pt_feedback_user_count"].tolist() == [4, 0]
    assert np.isfinite(bundle.arrays["pt_feedback_user_long_view_rate"]).all()


def test_composite_temporal_recipe_materializes_and_caches(tmp_path) -> None:
    values = _arrays()
    features = tmp_path / "features.npz"
    targets = tmp_path / "targets.npz"
    auxiliary = tmp_path / "auxiliary.npz"
    np.savez_compressed(
        features,
        row_id=np.arange(4, dtype=np.int64),
        date=np.asarray([20220408, 20220408, 20220408, 20220408], dtype=np.int32),
        user_id=values["users"],
        video_id=values["videos"],
        author_id=values["authors"],
        tab=np.asarray(["1"] * 4),
        duration_ms=np.asarray([10_000.0] * 4, dtype=np.float32),
        time_ms=values["times"],
        source_row_key=np.asarray([3, 1, 2, 4], dtype=np.int64),
    )
    np.savez_compressed(targets, long_view=values["labels"])
    np.savez_compressed(auxiliary, **_feedback(values["labels"]))
    first = materialize_feature_recipe(
        RICH_TEMPORAL_HISTORY,
        features,
        targets,
        features,
        tmp_path / "cache",
        auxiliary_target_path=auxiliary,
    )
    second = materialize_feature_recipe(
        RICH_TEMPORAL_HISTORY,
        features,
        targets,
        features,
        tmp_path / "cache",
        auxiliary_target_path=auxiliary,
    )
    assert first.identity_sha256 == second.identity_sha256
    output = load_feature_view(first.apply_features)
    assert "fx__pt_user_author_rate" in output.arrays
    assert "fx__pt_feedback_user_long_view_rate" in output.arrays


def test_candidate_recency_buckets_are_train_fitted_and_apply_safe(tmp_path) -> None:
    values = _arrays()
    train_features = tmp_path / "train_features.npz"
    apply_features = tmp_path / "apply_features.npz"
    targets = tmp_path / "targets.npz"
    common = {
        "date": np.asarray([20220408] * 4, dtype=np.int32),
        "user_id": values["users"],
        "video_id": values["videos"],
        "author_id": values["authors"],
        "tab": np.asarray(["1"] * 4),
        "duration_ms": np.asarray([10_000.0] * 4, dtype=np.float32),
    }
    np.savez_compressed(
        train_features,
        row_id=np.arange(4, dtype=np.int64),
        time_ms=values["times"],
        source_row_key=np.asarray([3, 1, 2, 4], dtype=np.int64),
        **common,
    )
    np.savez_compressed(
        apply_features,
        row_id=np.arange(2, dtype=np.int64),
        date=np.asarray([20220409] * 2, dtype=np.int32),
        user_id=np.asarray(["u", "cold"]),
        video_id=np.asarray(["v1", "new"]),
        author_id=np.asarray(["a", "new"]),
        tab=np.asarray(["1", "1"]),
        duration_ms=np.asarray([10_000.0] * 2, dtype=np.float32),
        time_ms=np.asarray([4000, 4000], dtype=np.int64),
        source_row_key=np.asarray([5, 6], dtype=np.int64),
    )
    np.savez_compressed(targets, long_view=values["labels"])

    artifact = materialize_feature_recipe(
        CANDIDATE_RECENCY_BUCKETS,
        train_features,
        targets,
        apply_features,
        tmp_path / "cache",
    )
    train = load_feature_view(artifact.train_features)
    apply = load_feature_view(artifact.apply_features)
    expected = {f"fx__{name}" for name in CANDIDATE_RECENCY_BUCKETS.output_features}
    assert expected <= train.arrays.keys()
    assert expected <= apply.arrays.keys()
    for name in expected:
        assert train.arrays[name].dtype == np.int32
        assert apply.arrays[name].dtype == np.int32
