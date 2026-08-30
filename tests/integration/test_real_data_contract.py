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
    assert len(verified.files) == 3
    manifest = bootstrap_views(default_data_dir(), tmp_path / "views")
    assert [manifest["splits"][name]["row_count"] for name in ("train", "valid", "test")] == [
        1_141_112,
        124_909,
        170_588,
    ]
    assert manifest["splits"]["test"]["target_path"] is None
    assert_no_test_target_artifact(tmp_path / "views")


def test_real_valid_only_baseline_acceptance(tmp_path) -> None:
    evidence = run_baseline_verification(
        default_data_dir(), tmp_path / "views", tmp_path / "baseline"
    )
    assert evidence.acceptance.accepted, evidence.acceptance.reasons
    assert evidence.summary_path.is_file()
