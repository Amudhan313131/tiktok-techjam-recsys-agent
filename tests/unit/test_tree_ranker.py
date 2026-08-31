from __future__ import annotations

import numpy as np
import pytest

from rex.data.views import FeatureView
from rex.models.tree_ranker import TreeRankerPlugin, tree_ranker_doctor


def test_tree_ranker_date_offsets_use_real_calendar_days() -> None:
    offsets = TreeRankerPlugin._date_offsets(
        np.asarray([20220408, 20220409, 20220501], dtype=np.int32)
    )
    assert np.allclose(offsets, np.asarray([0.0, 1.0 / 30.0, 23.0 / 30.0]))


def test_tree_ranker_rejects_malformed_dates() -> None:
    with pytest.raises(ValueError, match="YYYYMMDD"):
        TreeRankerPlugin._date_offsets(np.asarray([202204], dtype=np.int32))


def test_tree_ranker_inner_validation_is_strictly_later() -> None:
    dates = np.asarray(
        [20220408, 20220409, 20220410, 20220411, 20220412], dtype=np.int32
    )
    masks = TreeRankerPlugin._inner_temporal_masks(dates, validation_days=2)
    assert masks is not None
    train, valid = masks
    assert dates[train].max() < dates[valid].min()
    assert dates[valid].tolist() == [20220411, 20220412]


def test_tree_ranker_model_parameters_are_regularized_and_objective_bounded() -> None:
    parameters = TreeRankerPlugin._model_parameters(
        {"objective": "rank_xendcg", "bagging_fraction": 0.8},
        seed=19,
        n_estimators=77,
    )
    assert parameters["objective"] == "rank_xendcg"
    assert parameters["lambdarank_truncation_level"] == 8
    assert parameters["min_child_samples"] == 100
    assert parameters["n_estimators"] == 77
    assert parameters["bagging_freq"] == 1
    with pytest.raises(ValueError, match="unsupported"):
        TreeRankerPlugin._model_parameters({"objective": "binary"}, 0, 10)


def test_tree_ranker_static_fields_are_opt_in(tmp_path) -> None:
    rows = 3
    view = FeatureView(
        tmp_path / "view.npz",
        {
            "row_id": np.arange(rows, dtype=np.int64),
            "date": np.asarray([20220408, 20220409, 20220410], dtype=np.int32),
            "user_id": np.asarray(["u1", "u2", "u3"]),
            "video_id": np.asarray(["v1", "v2", "v3"]),
            "author_id": np.asarray(["a1", "a2", "a3"]),
            "tab": np.asarray(["1", "1", "2"]),
            "duration_ms": np.asarray([1000, 2000, 3000], dtype=np.float32),
            "meta__tag": np.asarray(["x", "y", "z"]),
            "meta_num__user_metadata_covered": np.ones(rows, dtype=np.float32),
            "fx__history": np.arange(rows, dtype=np.float32),
        },
        "0" * 64,
    )
    control, _ = TreeRankerPlugin._matrix(view)
    assert control.shape == (rows, 8)
    treatment, _ = TreeRankerPlugin._matrix(
        view,
        config={
            "categorical_fields": ["user_id", "video_id", "author_id", "tab", "meta__tag"],
            "numeric_fields": ["fx__history", "meta_num__user_metadata_covered"],
        },
    )
    assert treatment.shape == (rows, 10)


def test_tree_ranker_synthetic_doctor() -> None:
    pytest.importorskip("lightgbm")
    result = tree_ranker_doctor()
    assert result["ok"] is True
    assert result["deterministic"] is True
    assert result["bundle_members"] >= 3
