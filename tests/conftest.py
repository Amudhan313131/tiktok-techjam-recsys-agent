from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def feature_target_paths(tmp_path: Path) -> tuple[Path, Path]:
    feature = tmp_path / "features.npz"
    target = tmp_path / "targets.npz"
    np.savez_compressed(
        feature,
        row_id=np.arange(8, dtype=np.int64),
        date=np.asarray([20220408, 20220408, 20220409, 20220409, 20220410, 20220410, 20220411, 20220411]),
        user_id=np.asarray(["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"]),
        video_id=np.asarray(["v1", "v2", "v1", "v3", "v2", "v4", "v1", "v4"]),
        author_id=np.asarray(["a1", "a2", "a1", "a3", "a2", "a4", "a1", "a4"]),
        tab=np.asarray(["1"] * 8),
        duration_ms=np.asarray([10, 20, 10, 30, 20, 40, 10, 40], dtype=np.float32),
    )
    np.savez_compressed(target, long_view=np.asarray([1, 0, 0, 1, 1, 0, 0, 1], dtype=np.float32))
    return feature, target
