from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from rex.data.bootstrap import _manifest_identity_payload, build_split_views
from rex.data.firewall import (
    CapabilityViolation,
    assert_no_test_feedback_target_artifact,
    assert_sanitized_feature_view,
    validate_feedback_target_view,
)
from rex.data.views import load_feature_view, load_feedback_target_view
from rex.features.static_metadata import StaticMetadataError, load_static_metadata


LOG_HEADER = [
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
    "is_click",
    "is_like",
    "is_follow",
    "is_hate",
    "long_view",
    "duration_ms",
    "is_rand",
    "tab",
]
VIDEO_HEADER = [
    "video_id",
    "author_id",
    "video_type",
    "upload_dt",
    "upload_type",
    "visible_status",
    "video_duration",
    "server_width",
    "server_height",
    "music_id",
    "music_type",
    "tag",
]
USER_HEADER = [
    "user_id",
    "user_active_degree",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num",
    "follow_user_num_range",
    "fans_user_num",
    "fans_user_num_range",
    "friend_user_num",
    "friend_user_num_range",
    "register_days",
    "register_days_range",
]


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _dataset(root: Path, *, active_degree: str = "full_active", test_label: int = 0) -> dict:
    root.mkdir()
    _write_csv(
        root / "video_features_basic_pure.csv",
        VIDEO_HEADER,
        [
            ["v1", "a1", "NORMAL", "2022-04-01", "Web", 0, 10000, 720, 1280, 1, 9, 3],
            ["v2", "a2", "NORMAL", "2022-04-03", "ShortImport", 0, 20000, 1280, 720, 2, 4, 7],
        ],
    )
    _write_csv(
        root / "user_features_pure.csv",
        USER_HEADER,
        [
            ["u1", active_degree, 0, 1, 0, 12, "(10,50]", 3, "[1,10)", 2, "[1,5)", 99, "31-180"],
            ["u2", "middle_active", 0, 0, 1, 2, "(0,10]", 1, "[1,10)", 0, "0", 15, "0-30"],
        ],
    )
    _write_csv(
        root / "log_standard_4_08_to_4_21_pure.csv",
        LOG_HEADER,
        [
            ["u1", "v1", 20220408, 900, 1000, 1, 0, 0, 0, 1, 10000, 0, 1],
            ["u2", "v2", 20220409, 900, 1000, 0, 1, 0, 0, 0, 19000, 0, 1],
        ],
    )
    _write_csv(
        root / "log_standard_4_22_to_5_08_pure.csv",
        LOG_HEADER,
        [
            ["u1", "v2", 20220410, 1000, 2000, 1, 0, 1, 0, 1, 20000, 0, 2],
            ["u3", "v3", 20220411, 1100, 3000, 1, 1, 1, 1, test_label, 30000, 0, 2],
        ],
    )
    return {
        "label": "long_view",
        "metrics": ["GAUC", "nDCG@5"],
        "splits": {
            "train": {
                "split_start": 20220408,
                "split_end": 20220409,
                "observed_date_min": 20220408,
                "observed_date_max": 20220409,
                "rows": 2,
            },
            "valid": {
                "split_start": 20220410,
                "split_end": 20220410,
                "observed_date_min": 20220410,
                "observed_date_max": 20220410,
                "rows": 1,
            },
            "test": {
                "split_start": 20220411,
                "split_end": 20220411,
                "observed_date_min": 20220411,
                "observed_date_max": 20220411,
                "rows": 1,
            },
        },
    }


def test_static_join_temporal_keys_and_feedback_vault_are_aligned(tmp_path: Path) -> None:
    data = tmp_path / "data"
    benchmark = _dataset(data)
    summary: dict = {}
    splits = build_split_views(
        data,
        tmp_path / "views",
        benchmark=benchmark,
        metadata_summary=summary,
    )

    train = load_feature_view(splits["train"]["feature_path"])
    test = load_feature_view(splits["test"]["feature_path"])
    assert train.arrays["time_ms"].tolist() == [1000, 1000]
    assert len(np.unique(train.arrays["source_row_key"])) == 2
    assert np.all(train.arrays["meta__user_active_degree"] > 0)
    assert train.arrays["meta__user_active_degree"][0] != train.arrays[
        "meta__user_active_degree"
    ][1]
    assert np.all(train.arrays["meta__aspect_bucket"] > 0)
    assert train.arrays["meta__upload_age_bucket"].tolist() != [0, 0]
    assert test.arrays["meta__user_active_degree"].tolist() == [0]
    assert test.arrays["meta_num__user_metadata_covered"].tolist() == [0.0]
    assert summary["coverage_by_split"]["test"]["user_coverage_rate"] == 0.0
    assert summary["sources"]["user"]["sha256"]
    assert summary["transform"]["identity_sha256"]

    feedback = load_feedback_target_view(splits["train"]["feedback_target_path"])
    assert set(feedback.arrays) == {"is_click", "is_like", "is_follow", "is_hate", "long_view"}
    assert feedback.arrays["is_click"].tolist() == [1.0, 0.0]
    validate_feedback_target_view(splits["valid"]["feedback_target_path"], split="valid")
    assert splits["test"]["feedback_target_path"] is None
    assert splits["test"]["feedback_target_sha256"] is None
    assert_no_test_feedback_target_artifact(tmp_path / "views")


def test_hidden_test_outcomes_cannot_change_features_or_create_feedback(tmp_path: Path) -> None:
    left_data = tmp_path / "left-data"
    right_data = tmp_path / "right-data"
    benchmark = _dataset(left_data, test_label=0)
    _dataset(right_data, test_label=1)
    left = build_split_views(left_data, tmp_path / "left", benchmark=benchmark)
    right = build_split_views(right_data, tmp_path / "right", benchmark=benchmark)
    assert left["test"]["feature_sha256"] == right["test"]["feature_sha256"]
    assert left["test"]["feedback_target_path"] is None
    assert right["test"]["feedback_target_path"] is None


def test_safe_side_table_change_invalidates_view_identity(tmp_path: Path) -> None:
    left_data = tmp_path / "left-data"
    right_data = tmp_path / "right-data"
    benchmark = _dataset(left_data, active_degree="full_active")
    _dataset(right_data, active_degree="low_active")
    left_summary: dict = {}
    right_summary: dict = {}
    left = build_split_views(
        left_data, tmp_path / "left", benchmark=benchmark, metadata_summary=left_summary
    )
    right = build_split_views(
        right_data, tmp_path / "right", benchmark=benchmark, metadata_summary=right_summary
    )
    assert left["train"]["feature_sha256"] != right["train"]["feature_sha256"]
    assert left_summary["sources"]["user"]["sha256"] != right_summary["sources"]["user"]["sha256"]
    assert left_summary["identity_sha256"] != right_summary["identity_sha256"]


def test_firewall_rejects_outcome_aliases_statistics_and_unknown_metadata() -> None:
    base = {
        "row_id": np.arange(1),
        "date": np.asarray([20220408]),
        "user_id": np.asarray(["u"]),
        "video_id": np.asarray(["v"]),
        "author_id": np.asarray(["a"]),
        "tab": np.asarray(["1"]),
        "duration_ms": np.asarray([1.0]),
    }
    with pytest.raises(CapabilityViolation, match="forbidden outcomes"):
        assert_sanitized_feature_view({**base, "fx__long_view": np.asarray([1.0])})
    with pytest.raises(CapabilityViolation, match="month-level statistics"):
        assert_sanitized_feature_view({**base, "fx__show_cnt": np.asarray([1.0])})
    with pytest.raises(CapabilityViolation, match="schema drifted"):
        assert_sanitized_feature_view({**base, "meta__unreviewed": np.asarray(["x"])})


def test_duplicate_side_table_identity_fails_closed(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _dataset(data)
    with (data / "user_features_pure.csv").open("a", encoding="utf-8") as handle:
        handle.write("u1,full_active,0,1,0,1,(0,10],1,[1,10),1,[1,5),1,0-30\n")
    with pytest.raises(StaticMetadataError, match="duplicate user_id"):
        load_static_metadata(data)


def test_manifest_identity_excludes_every_capability_path() -> None:
    manifest = {
        "generation_command": "command",
        "generation_commit": "commit",
        "other": "stable",
        "splits": {
            "train": {
                "feature_path": "/first/features.npz",
                "target_path": "/first/targets.npz",
                "feedback_target_path": "/first/feedback.npz",
                "feature_sha256": "a" * 64,
                "target_sha256": "b" * 64,
                "feedback_target_sha256": "c" * 64,
            }
        },
    }
    first = _manifest_identity_payload(manifest)
    manifest["splits"]["train"].update(
        {
            "feature_path": "/second/features.npz",
            "target_path": "/second/targets.npz",
            "feedback_target_path": "/second/feedback.npz",
        }
    )
    assert first == _manifest_identity_payload(manifest)
