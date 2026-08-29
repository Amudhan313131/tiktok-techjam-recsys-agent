"""Trusted baseline reproduction without exposing or scoring hidden-test labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rex.contracts import Metrics
from rex.data.bootstrap import bootstrap_views
from rex.data.manifest import verify_starter_manifest
from rex.data.views import load_feature_view, load_target_view
from rex.models.official_fm import CategoricalEncoder, FM


@dataclass(frozen=True)
class BaselineSeedResult:
    seed: int
    metrics: Metrics
    best_epoch: int


@dataclass(frozen=True)
class BaselineBundle:
    results: tuple[BaselineSeedResult, ...]
    mean_primary: float
    std_primary: float


def _official_metrics(user_ids: np.ndarray, labels: np.ndarray, scores: np.ndarray, seed: int) -> Metrics:
    starter = verify_starter_manifest()
    # Importing through the protected adapter would require a prediction artifact;
    # this verifier is trusted control-plane code and calls the same immutable file.
    import importlib.util

    evaluator_path = starter.root / "evaluate.py"
    spec = importlib.util.spec_from_file_location("rex_baseline_evaluator", evaluator_path)
    if spec is None or spec.loader is None:  # pragma: no cover - import machinery guard
        raise RuntimeError(f"cannot load evaluator: {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = module.evaluate(user_ids.tolist(), labels.tolist(), scores.tolist())
    return Metrics(
        GAUC=float(raw["GAUC"]),
        **{"nDCG@5": float(raw["nDCG@5"])},
        primary=float(raw["primary"]),
        users=int(raw["users"]),
        rows=int(raw["rows"]),
        evaluator_sha256=starter.hashes["evaluate.py"],
        split="valid",
        seed=seed,
    )


def reproduce_fm_seed(
    view_dir: str | Path,
    *,
    seed: int,
    k: int = 16,
    lr: float = 0.001,
    l2: float = 1e-6,
    epochs: int = 40,
    batch_size: int = 8192,
    patience: int = 4,
) -> BaselineSeedResult:
    """Reproduce the organizer FM training loop using valid-only early stopping."""
    root = Path(view_dir)
    train = load_feature_view(root / "train_features.npz")
    valid = load_feature_view(root / "valid_features.npz")
    train_targets = load_target_view(root / "label_vault/train_targets.npz")
    valid_targets = load_target_view(root / "label_vault/valid_targets.npz")
    encoder = CategoricalEncoder.fit(train)
    train_x = encoder.transform(train)
    valid_x = encoder.transform(valid)
    model = FM(encoder.dimension, k=k, lr=lr, l2=l2, seed=seed)
    rng = np.random.default_rng(seed)
    best_primary = -1.0
    best_epoch = 0
    best_state: tuple[np.ndarray, np.ndarray, np.float32, int] | None = None
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(train.rows)
        for offset in range(0, len(order), batch_size):
            batch = order[offset : offset + batch_size]
            model.step(train_x[batch], train_targets.labels[batch])
        metrics = _official_metrics(
            valid.arrays["user_id"], valid_targets.labels, model.predict(valid_x), seed
        )
        if metrics.primary > best_primary + 1e-5:
            best_primary = metrics.primary
            best_epoch = epoch
            bad_epochs = 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b), model.t)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is None:  # pragma: no cover - at least one epoch is required by defaults
        raise RuntimeError("FM baseline did not complete an epoch")
    model.V, model.W, model.b, model.t = best_state
    final = _official_metrics(
        valid.arrays["user_id"], valid_targets.labels, model.predict(valid_x), seed
    )
    return BaselineSeedResult(seed=seed, metrics=final, best_epoch=best_epoch)


def reproduce_fm_bundle(
    data_dir: str | Path,
    view_dir: str | Path,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> BaselineBundle:
    """Bootstrap sanitized views and reproduce five valid seeds; test is never scored."""
    bootstrap_views(data_dir, view_dir)
    results = tuple(reproduce_fm_seed(view_dir, seed=seed) for seed in seeds)
    scores = np.asarray([item.metrics.primary for item in results], dtype=np.float64)
    return BaselineBundle(
        results=results,
        mean_primary=float(scores.mean()),
        std_primary=float(scores.std(ddof=0)),
    )
