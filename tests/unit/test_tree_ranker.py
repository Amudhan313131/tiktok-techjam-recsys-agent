from __future__ import annotations

import numpy as np
import pytest

from rex.models.tree_ranker import TreeRankerPlugin, tree_ranker_doctor


def test_tree_ranker_date_offsets_use_real_calendar_days() -> None:
    offsets = TreeRankerPlugin._date_offsets(
        np.asarray([20220408, 20220409, 20220501], dtype=np.int32)
    )
    assert np.allclose(offsets, np.asarray([0.0, 1.0 / 30.0, 23.0 / 30.0]))


def test_tree_ranker_rejects_malformed_dates() -> None:
    with pytest.raises(ValueError, match="YYYYMMDD"):
        TreeRankerPlugin._date_offsets(np.asarray([202204], dtype=np.int32))


def test_tree_ranker_synthetic_doctor() -> None:
    pytest.importorskip("lightgbm")
    result = tree_ranker_doctor()
    assert result["ok"] is True
    assert result["deterministic"] is True
    assert result["bundle_members"] >= 3
