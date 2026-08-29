from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from rex.contracts import AttemptStatus, RunRequest
from rex.data.manifest import sha256_file
from rex.execution.artifacts import load_prediction_artifact
from rex.execution.runner import execute_request


HASH = "0" * 64


def _request(feature_target_paths, tmp_path: Path, config: dict, **overrides) -> RunRequest:
    features, targets = feature_target_paths
    config_path = tmp_path / f"worktree-config-{time.time_ns()}.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    values = {
        "run_id": "run",
        "experiment_id": "worktree-experiment",
        "attempt_id": f"attempt-{time.time_ns()}",
        "commit_sha": "fixture",
        "plugin": "rex.models.experimental.fixture:FixturePlugin",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "seed": 1,
        "rung": "fixture",
        "split": "train",
        "feature_view_path": str(features),
        "target_view_path": str(targets),
        "output_dir": str(tmp_path / f"worktree-output-{time.time_ns()}"),
        "deadline_epoch_ms": int((time.time() + 20) * 1000),
        "timeout_seconds": 10,
        "data_view_sha256": sha256_file(features),
        "environment_sha256": HASH,
    }
    values.update(overrides)
    return RunRequest(**values)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_runner_imports_model_code_from_verified_worktree(
    feature_target_paths, tmp_path: Path
) -> None:
    trusted_root = tmp_path / "trusted-worktrees"
    project = trusted_root / "candidate"
    source = Path(__file__).resolve().parents[2] / "src" / "rex"
    shutil.copytree(
        source,
        project / "src" / "rex",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        Path(__file__).resolve().parents[2] / "configs" / "frozen",
        project / "configs" / "frozen",
    )
    probe = project / "src" / "rex" / "models" / "experimental" / "worktree_probe.py"
    probe.write_text(
        """from pathlib import Path
import json
import numpy as np

class WorktreeProbe:
    def fit(self, train_features, train_targets, config, seed, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / 'probe.json'
        path.write_text(json.dumps({'value': 0.1}), encoding='utf-8')
        return path

    def predict(self, model_artifact, features, config, output_dir):
        value = json.loads(model_artifact.read_text(encoding='utf-8'))['value']
        return np.full(features.rows, value, dtype=np.float64)
""",
        encoding="utf-8",
    )
    _git(project, "init")
    _git(project, "config", "user.email", "rex@example.invalid")
    _git(project, "config", "user.name", "REX Fixture")
    _git(project, "add", "--all")
    _git(project, "commit", "-m", "fixture base")
    probe.write_text(probe.read_text(encoding="utf-8").replace("0.1", "0.9"), encoding="utf-8")
    _git(project, "add", str(probe.relative_to(project)))
    _git(project, "commit", "-m", "candidate patch")
    commit = _git(project, "rev-parse", "HEAD")

    fit_request = _request(
        feature_target_paths,
        tmp_path,
        {},
        plugin="rex.models.experimental.worktree_probe:WorktreeProbe",
        commit_sha=commit,
        workspace_path=str(project),
    )
    fit = execute_request(
        fit_request,
        tmp_path / "worktree-fit",
        trusted_worktree_root=trusted_root,
    )
    assert fit.status == AttemptStatus.SUCCESS
    bundle = next(artifact for artifact in fit.artifacts if artifact.kind == "model_bundle")

    predict_request = _request(
        feature_target_paths,
        tmp_path,
        {},
        plugin="rex.models.experimental.worktree_probe:WorktreeProbe",
        commit_sha=commit,
        workspace_path=str(project),
        operation="predict",
        rung="predict",
        split="valid",
        target_view_path=None,
        model_bundle_path=bundle.path,
    )
    predicted = execute_request(
        predict_request,
        tmp_path / "worktree-predict",
        trusted_worktree_root=trusted_root,
    )
    assert predicted.status == AttemptStatus.SUCCESS
    prediction = next(artifact for artifact in predicted.artifacts if artifact.kind == "predictions")
    arrays = load_prediction_artifact(prediction.path, feature_target_paths[0])
    assert np.array_equal(arrays["score"], np.full(8, 0.9))


def test_runner_rejects_workspace_outside_trusted_root(feature_target_paths, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    fit_request = _request(
        feature_target_paths,
        tmp_path,
        {},
        workspace_path=str(outside),
    )
    result = execute_request(
        fit_request,
        tmp_path / "untrusted-workspace",
        trusted_worktree_root=tmp_path / "trusted",
    )
    assert result.status == AttemptStatus.CONTRACT
    assert result.error_type == "WorkspaceViolation"
