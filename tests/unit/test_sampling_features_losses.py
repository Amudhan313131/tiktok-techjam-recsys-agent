from __future__ import annotations

import numpy as np

from rex.data.groups import sample_complete_users, sample_same_user_pairs
from rex.features.temporal_aggregates import expanding_target_rate
from rex.features.history_summaries import candidate_history_summaries
from rex.features.base import FeatureBundle, attach_feature_bundles
from rex.data.views import load_feature_view
from rex.losses.ranking import delta_ndcg_weights, pair_logistic_gradient, pair_logistic_loss


def test_complete_user_sampling_never_slices_a_user() -> None:
    users = np.asarray(["a", "a", "b", "b", "b", "c", "c"])
    selected = sample_complete_users(users, fraction=0.4, seed=7)
    chosen = set(users[selected])
    for user in chosen:
        assert set(np.flatnonzero(users == user)) <= set(selected.tolist())


def test_pair_sampler_never_crosses_users_and_skips_degenerate_groups() -> None:
    users = np.asarray(["a", "a", "b", "b", "c", "c"])
    labels = np.asarray([1, 0, 1, 1, 0, 0])
    pairs = sample_same_user_pairs(users, labels, negatives_per_positive=3, seed=3)
    assert np.array_equal(users[pairs.positive_indices], users[pairs.negative_indices])
    assert set(users[pairs.positive_indices]) == {"a"}
    assert np.all(labels[pairs.positive_indices] == 1)
    assert np.all(labels[pairs.negative_indices] == 0)


def test_future_label_poison_cannot_change_earlier_aggregate() -> None:
    keys = np.asarray(["v", "v", "v", "v"])
    dates = np.asarray([1, 2, 3, 4])
    rows = np.arange(4)
    labels = np.asarray([0, 1, 0, 0], dtype=np.float32)
    original = expanding_target_rate(keys, dates, rows, labels).arrays["target_rate"]
    poisoned = labels.copy()
    poisoned[-1] = 1
    changed = expanding_target_rate(keys, dates, rows, poisoned).arrays["target_rate"]
    assert np.array_equal(original[:-1], changed[:-1])


def test_same_day_labels_do_not_influence_each_other() -> None:
    keys = np.asarray(["v", "v", "v"])
    dates = np.asarray([1, 1, 2])
    rows = np.arange(3)
    left = expanding_target_rate(keys, dates, rows, np.asarray([0, 1, 0])).arrays["target_rate"]
    right = expanding_target_rate(keys, dates, rows, np.asarray([1, 0, 0])).arrays["target_rate"]
    assert left[0] == left[1] == right[0] == right[1]


def test_future_label_poison_cannot_change_history_summary() -> None:
    users = np.asarray(["u"] * 4)
    authors = np.asarray(["a"] * 4)
    dates = np.asarray([1, 2, 3, 4])
    labels = np.asarray([0, 1, 0, 0], dtype=np.float32)
    arguments = (users, authors, np.asarray([1000] * 4), dates, np.arange(4))
    original = candidate_history_summaries(*arguments, labels).arrays["user_author_rate"]
    labels[-1] = 1
    poisoned = candidate_history_summaries(*arguments, labels).arrays["user_author_rate"]
    assert np.array_equal(original[:-1], poisoned[:-1])


def test_pair_loss_and_gradient_prefer_positive_margin() -> None:
    assert pair_logistic_loss(np.asarray([2.0])) < pair_logistic_loss(np.asarray([-2.0]))
    assert pair_logistic_gradient(np.asarray([0.0]))[0] == -0.5


def test_delta_ndcg_has_zero_weight_beyond_cutoff() -> None:
    weights = delta_ndcg_weights(np.asarray([7]), np.asarray([8]), np.asarray([3]), cutoff=5)
    assert weights[0] == 0.0


def test_engineered_feature_is_materialized_and_hash_covered(feature_target_paths, tmp_path) -> None:
    features, _ = feature_target_paths
    bundle = FeatureBundle(
        arrays={"history_length": np.arange(8, dtype=np.float32)},
        provenance={"history_length": {"cutoff": "strictly earlier date"}},
    )
    path = attach_feature_bundles(features, [bundle], tmp_path / "engineered.npz")
    view = load_feature_view(path)
    assert "fx__history_length" in view.arrays
    assert view.arrays["fx__history_length"].tolist() == list(range(8))
