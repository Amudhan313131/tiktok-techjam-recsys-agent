from __future__ import annotations

import numpy as np
import pytest

from rex.agents.patch_guard import PatchPolicy, PatchRejected, validate_patch
from rex.agents.static_audit import (
    StaticAuditRejected,
    audit_fixture_bias_only,
    audit_python_file,
)
from rex.data.firewall import CapabilityViolation, assert_sanitized_feature_view


POLICY = PatchPolicy(
    allowed=("src/rex/models/experimental/**", "configs/experiments/**"),
    denied=("src/rex/evaluation/**", "configs/frozen/**"),
)


def _patch(path: str) -> str:
    return f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-old\n+new\n"


def test_allowed_declared_patch_passes() -> None:
    path = "src/rex/models/experimental/new.py"
    assert validate_patch(_patch(path), POLICY, declared_files=[path]) == (path,)


@pytest.mark.parametrize(
    "patch",
    [
        _patch("src/rex/evaluation/official_adapter.py"),
        _patch("README.md"),
        "diff --git a/x b/x\nnew file mode 120000\n--- /dev/null\n+++ b/x\n@@ -0,0 +1 @@\n+target\n",
        "GIT binary patch\nliteral 0\n",
    ],
)
def test_protected_or_special_patch_is_rejected(patch: str) -> None:
    with pytest.raises(PatchRejected):
        validate_patch(patch, POLICY)


def test_undeclared_file_is_rejected() -> None:
    with pytest.raises(PatchRejected):
        validate_patch(
            _patch("src/rex/models/experimental/new.py"),
            POLICY,
            declared_files=["src/rex/models/experimental/other.py"],
        )


def test_sanitized_view_rejects_target_column() -> None:
    arrays = {
        "row_id": np.arange(1),
        "date": np.asarray([1]),
        "user_id": np.asarray(["u"]),
        "video_id": np.asarray(["v"]),
        "author_id": np.asarray(["a"]),
        "tab": np.asarray(["1"]),
        "duration_ms": np.asarray([1.0]),
        "long_view": np.asarray([1]),
    }
    with pytest.raises(CapabilityViolation):
        assert_sanitized_feature_view(arrays)


def test_static_audit_rejects_label_vault_access(tmp_path) -> None:
    path = tmp_path / "bad.py"
    path.write_text("from pathlib import Path\nPath('label_vault/valid_targets.npz').read_bytes()\n")
    with pytest.raises(StaticAuditRejected):
        audit_python_file(path)


def test_static_audit_accepts_model_math(tmp_path) -> None:
    path = tmp_path / "good.py"
    path.write_text("import numpy as np\ndef score(x):\n    return np.asarray(x) * 2\n")
    audit_python_file(path)


def _fixture_source(root, text: str) -> None:
    path = root / "src/rex/models/experimental/fixture.py"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")


def test_fixture_semantic_audit_allows_only_one_bounded_literal(tmp_path) -> None:
    parent = tmp_path / "parent"
    worktree = tmp_path / "worktree"
    _fixture_source(parent, "trusted = True\nDEFAULT_BIAS = 0.0\n")
    _fixture_source(worktree, "trusted = True\nDEFAULT_BIAS = 0.125\n")
    assert audit_fixture_bias_only(
        parent, worktree, ("src/rex/models/experimental/fixture.py",)
    ) == 0.125


@pytest.mark.parametrize(
    "candidate",
    [
        "trusted = False\nDEFAULT_BIAS = 0.1\n",
        "trusted = True\nDEFAULT_BIAS = 0.1\nopen('/tmp/leak')\n",
        "trusted = True\nDEFAULT_BIAS = 1e100\n",
        "trusted = True\nDEFAULT_BIAS = __import__('os')\n",
    ],
)
def test_fixture_semantic_audit_rejects_any_other_code_change(tmp_path, candidate: str) -> None:
    parent = tmp_path / "parent"
    worktree = tmp_path / "worktree"
    _fixture_source(parent, "trusted = True\nDEFAULT_BIAS = 0.0\n")
    _fixture_source(worktree, candidate)
    with pytest.raises(StaticAuditRejected):
        audit_fixture_bias_only(
            parent, worktree, ("src/rex/models/experimental/fixture.py",)
        )
