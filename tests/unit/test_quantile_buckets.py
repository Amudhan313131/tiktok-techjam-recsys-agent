from __future__ import annotations

import json

import numpy as np
import pytest

from rex.features.base import FeatureBundle
from rex.features.quantile_buckets import (
    CAPPED_COUNT_STRATEGY,
    MISSING_BUCKET,
    QuantileBucketState,
    apply_quantile_buckets,
    fit_quantile_buckets,
    fit_transform_quantile_buckets,
)


def _bundle(**arrays: list[float] | np.ndarray) -> FeatureBundle:
    values = {name: np.asarray(value) for name, value in arrays.items()}
    return FeatureBundle(
        values,
        {name: {"source": "strictly-prior history"} for name in values},
    )


def test_quantile_edges_are_fitted_on_train_and_frozen_for_apply() -> None:
    train = _bundle(pt_user_author_rate=[0.1, 0.2, 0.3, 0.4, 0.5])
    state, encoded_train = fit_transform_quantile_buckets(
        train,
        fields=["pt_user_author_rate"],
        quantile_bins=4,
    )

    bucket = state.buckets[0]
    assert bucket.edges == pytest.approx((0.2, 0.3, 0.4))
    assert encoded_train.arrays[bucket.output].tolist() == [1, 2, 3, 4, 4]

    # Values far outside the training range reuse the frozen end buckets.  No
    # apply-distribution quantile is fitted.
    apply = _bundle(pt_user_author_rate=[-100.0, 0.25, 100.0])
    encoded_apply = apply_quantile_buckets(apply, state)
    assert encoded_apply.arrays[bucket.output].tolist() == [1, 2, 4]
    assert encoded_apply.provenance[bucket.output]["cutoff"].endswith("no targets")


def test_missing_recency_sentinel_has_a_distinct_reserved_bucket() -> None:
    train = _bundle(pt_user_video_days_since=[-1.0, 0.0, 1.0, 4.0, 9.0])
    state = fit_quantile_buckets(
        train,
        fields=["pt_user_video_days_since"],
        quantile_bins=3,
    )
    result = apply_quantile_buckets(
        _bundle(pt_user_video_days_since=[-1.0, -10.0, 0.0, 100.0]),
        state,
    )

    values = result.arrays["bucket__pt_user_video_days_since"]
    assert values[0] == MISSING_BUCKET
    assert values[1] != MISSING_BUCKET
    assert np.all(values[1:] > MISSING_BUCKET)


def test_count_features_use_capped_power_of_two_bands() -> None:
    train = _bundle(pt_feedback_user_count=[0, 1, 2, 3, 4, 9, 100])
    state = fit_quantile_buckets(
        train,
        fields=["pt_feedback_user_count"],
        count_cap=8,
    )

    fitted = state.buckets[0]
    assert fitted.strategy == CAPPED_COUNT_STRATEGY
    assert fitted.edges == (0.5, 1.5, 3.5, 7.5)
    result = apply_quantile_buckets(
        _bundle(pt_feedback_user_count=[0, 1, 2, 3, 4, 8, 999]),
        state,
    )
    assert result.arrays[fitted.output].tolist() == [1, 2, 3, 3, 4, 5, 5]


def test_state_json_round_trip_preserves_bins_and_provenance() -> None:
    train = _bundle(
        pt_user_author_rate=[0.1, 0.2, 0.8, 0.9],
        pt_user_author_count=[0, 1, 2, 8],
    )
    state = fit_quantile_buckets(train)
    payload = json.loads(json.dumps(state.to_json(), sort_keys=True))
    restored = QuantileBucketState.from_json(payload)

    assert restored.to_json() == state.to_json()
    first = apply_quantile_buckets(train, state)
    second = apply_quantile_buckets(train, restored)
    for name in first.arrays:
        np.testing.assert_array_equal(first.arrays[name], second.arrays[name])
        assert second.provenance[name]["source_provenance"] == {
            "source": "strictly-prior history"
        }


@pytest.mark.parametrize(
    "bundle, message",
    [
        (_bundle(pt_user_author_rate=[0.1, np.nan]), "non-finite"),
        (
            FeatureBundle(
                {
                    "pt_user_author_rate": np.asarray([0.1, 0.2]),
                    "pt_user_author_count": np.asarray([1]),
                },
                {
                    "pt_user_author_rate": {},
                    "pt_user_author_count": {},
                },
            ),
            "rows; expected",
        ),
        (_bundle(pt_user_author_rate=np.asarray([[0.1], [0.2]])), "one-dimensional"),
    ],
)
def test_fit_rejects_nonfinite_or_misaligned_numeric_data(
    bundle: FeatureBundle,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_quantile_buckets(bundle, fields=["pt_user_author_rate"])


def test_apply_rejects_missing_source_and_invalid_count_values() -> None:
    state = fit_quantile_buckets(
        _bundle(pt_user_video_count=[0, 1, 2]),
        fields=["pt_user_video_count"],
    )
    with pytest.raises(ValueError, match="source feature is missing"):
        apply_quantile_buckets(_bundle(other=[1, 2, 3]), state)
    with pytest.raises(ValueError, match="negative"):
        apply_quantile_buckets(_bundle(pt_user_video_count=[0, -1, 2]), state)


def test_explicit_strategy_and_missing_overrides_are_validated() -> None:
    train = _bundle(custom=[-1.0, 0.0, 2.0])
    state = fit_quantile_buckets(
        train,
        fields=["custom"],
        strategies={"custom": CAPPED_COUNT_STRATEGY},
        missing_values={"custom": [-1.0]},
    )
    assert apply_quantile_buckets(train, state).arrays["bucket__custom"][0] == 0

    with pytest.raises(ValueError, match="unselected fields"):
        fit_quantile_buckets(train, fields=["custom"], strategies={"other": "quantile"})
    with pytest.raises(ValueError, match="unsupported"):
        fit_quantile_buckets(train, fields=["custom"], strategies={"custom": "bad"})
