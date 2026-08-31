from __future__ import annotations

import os

import pytest

from rex.data.bootstrap import bootstrap_views, default_data_dir
from rex.data.firewall import assert_no_test_target_artifact
from rex.data.manifest import verify_raw_dataset
from rex.evaluation.baseline import run_baseline_verification


pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(
        os.environ.get("REX_REAL_DATA") != "1",
        reason="set REX_REAL_DATA=1 to verify local KuaiRand-Pure inputs",
    ),
]


def test_real_raw_data_and_sanitized_split_contract(tmp_path) -> None:
    verified = verify_raw_dataset(default_data_dir())
    assert set(verified.files) == {
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
        "user_features_pure.csv",
        "video_features_basic_pure.csv",
    }
    manifest = bootstrap_views(default_data_dir(), tmp_path / "views")
    assert [manifest["splits"][name]["row_count"] for name in ("train", "valid", "test")] == [
        1_141_112,
        124_909,
        170_588,
    ]
    assert manifest["splits"]["test"]["target_path"] is None
    assert manifest["splits"]["test"]["feedback_target_path"] is None
    assert manifest["static_metadata"]["identity_sha256"]
    assert_no_test_target_artifact(tmp_path / "views")


def test_real_valid_only_baseline_acceptance(tmp_path) -> None:
    evidence = run_baseline_verification(
        default_data_dir(), tmp_path / "views", tmp_path / "baseline"
    )
    assert evidence.acceptance.accepted, evidence.acceptance.reasons
    assert evidence.summary_path.is_file()
