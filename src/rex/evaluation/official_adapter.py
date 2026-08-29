"""Protected adapter around the frozen organizer evaluator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from rex.contracts import Metrics
from rex.data.manifest import verify_starter_manifest
from rex.data.views import load_feature_view, load_target_view
from rex.execution.artifacts import load_prediction_artifact


class EvaluationError(RuntimeError):
    pass


def _load_evaluator() -> tuple[ModuleType, str]:
    starter = verify_starter_manifest()
    path = starter.root / "evaluate.py"
    spec = importlib.util.spec_from_file_location("rex_frozen_evaluator", path)
    if spec is None or spec.loader is None:
        raise EvaluationError(f"cannot load evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, starter.hashes["evaluate.py"]


def evaluate_predictions(
    feature_view_path: str | Path,
    target_view_path: str | Path,
    prediction_path: str | Path,
    *,
    split: str,
    fold: str | None = None,
    seed: int | None = None,
) -> Metrics:
    if split == "test":
        raise EvaluationError("hidden-test scoring is disabled in development")
    features = load_feature_view(feature_view_path)
    targets = load_target_view(target_view_path)
    predictions = load_prediction_artifact(prediction_path, features)
    if features.rows != len(targets.labels):
        raise EvaluationError("feature/target row mismatch")
    module, evaluator_hash = _load_evaluator()
    raw = module.evaluate(
        features.arrays["user_id"].tolist(),
        targets.labels.tolist(),
        predictions["score"].tolist(),
    )
    return Metrics(
        GAUC=float(raw["GAUC"]),
        **{"nDCG@5": float(raw["nDCG@5"])},
        primary=float(raw["primary"]),
        users=int(raw["users"]),
        rows=int(raw["rows"]),
        evaluator_sha256=evaluator_hash,
        split=split,
        fold=fold,
        seed=seed,
    )
