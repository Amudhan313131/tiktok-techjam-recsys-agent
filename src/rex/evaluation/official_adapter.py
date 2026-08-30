"""Protected subprocess adapter around the frozen organizer evaluator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from rex.contracts import Metrics
from rex.data.manifest import verify_starter_manifest
from rex.data.views import load_feature_view, load_target_view
from rex.execution.artifacts import load_prediction_artifact


class EvaluationError(RuntimeError):
    pass


def official_evaluator_command(input_path: str | Path = "<private-input.npz>") -> list[str]:
    starter = verify_starter_manifest()
    process = Path(__file__).with_name("evaluator_process.py").resolve()
    return [
        sys.executable,
        "-I",
        str(process),
        "--evaluator",
        str(starter.root / "evaluate.py"),
        "--input",
        str(input_path),
    ]


def evaluate_arrays(
    user_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    split: str,
    fold: str | None = None,
    seed: int | None = None,
    timeout_seconds: int = 120,
) -> Metrics:
    if split not in {"train", "valid", "shadow"}:
        raise EvaluationError(f"scoring split {split!r} is disabled in development")
    users = np.asarray(user_ids, dtype=str)
    targets = np.asarray(labels, dtype=np.float32)
    predictions = np.asarray(scores, dtype=np.float64)
    if not (len(users) == len(targets) == len(predictions)):
        raise EvaluationError("evaluator inputs have different lengths")
    if not np.isfinite(predictions).all():
        raise EvaluationError("evaluator predictions contain NaN or Inf")
    starter = verify_starter_manifest()
    with tempfile.TemporaryDirectory(prefix="rex-evaluator-") as temporary:
        input_path = Path(temporary) / "input.npz"
        np.savez_compressed(
            input_path, user_id=users, long_view=targets, score=predictions
        )
        command = official_evaluator_command(input_path)
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "PYTHONHASHSEED": "0",
        }
        completed = subprocess.run(
            command,
            cwd=temporary,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    if completed.returncode != 0:
        summary = (completed.stderr or completed.stdout).strip()[-1000:]
        raise EvaluationError(
            f"official evaluator failed with exit {completed.returncode}: {summary}"
        )
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise EvaluationError("official evaluator emitted invalid JSON") from error
    return Metrics(
        GAUC=float(raw["GAUC"]),
        **{"nDCG@5": float(raw["nDCG@5"])},
        primary=float(raw["primary"]),
        users=int(raw["users"]),
        rows=int(raw["rows"]),
        evaluator_sha256=starter.hashes["evaluate.py"],
        split=split,
        fold=fold,
        seed=seed,
    )


def evaluate_predictions(
    feature_view_path: str | Path,
    target_view_path: str | Path,
    prediction_path: str | Path,
    *,
    split: str,
    fold: str | None = None,
    seed: int | None = None,
) -> Metrics:
    if split not in {"train", "valid", "shadow"}:
        raise EvaluationError(f"scoring split {split!r} is disabled in development")
    features = load_feature_view(feature_view_path)
    targets = load_target_view(target_view_path)
    predictions = load_prediction_artifact(prediction_path, features)
    if features.rows != len(targets.labels):
        raise EvaluationError("feature/target row mismatch")
    return evaluate_arrays(
        features.arrays["user_id"].tolist(),
        targets.labels,
        predictions["score"],
        split=split,
        fold=fold,
        seed=seed,
    )
