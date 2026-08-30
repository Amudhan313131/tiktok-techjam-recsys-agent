"""Trusted valid-only baseline reproduction and evidence persistence."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rex.contracts import Metrics
from rex.data.bootstrap import bootstrap_views
from rex.data.manifest import load_benchmark_manifest, repo_root, sha256_bytes, sha256_file
from rex.data.views import FeatureView, load_feature_view, load_target_view
from rex.execution.artifacts import atomic_write_json, write_prediction_artifact
from rex.evaluation.official_adapter import evaluate_arrays, official_evaluator_command
from rex.models.bundle import create_model_bundle
from rex.models.official_fm import CategoricalEncoder, FM


OFFICIAL_FM_PLUGIN = "rex.models.official_fm:OfficialFMPlugin"


@dataclass(frozen=True)
class BaselineSeedResult:
    seed: int
    metrics: Metrics
    best_epoch: int
    evidence_dir: Path | None = None
    prediction_sha256: str | None = None
    model_sha256: str | None = None


@dataclass(frozen=True)
class BaselineBundle:
    results: tuple[BaselineSeedResult, ...]
    mean_primary: float
    std_primary: float


@dataclass(frozen=True)
class BaselineAcceptance:
    accepted: bool
    reasons: tuple[str, ...]
    observed: dict[str, float]


@dataclass(frozen=True)
class BaselineEvidence:
    random_metrics: Metrics
    popularity_metrics: Metrics
    fm: BaselineBundle
    acceptance: BaselineAcceptance
    summary_path: Path


def _official_metrics(user_ids: np.ndarray, labels: np.ndarray, scores: np.ndarray, seed: int) -> Metrics:
    return evaluate_arrays(user_ids, labels, scores, split="valid", seed=seed)


def _environment_evidence() -> dict[str, object]:
    lock = repo_root() / "requirements-lock.txt"
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo_root(),
        check=False,
        capture_output=True,
    )
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "requirements_lock_path": str(lock.resolve()),
        "requirements_lock_sha256": sha256_file(lock) if lock.is_file() else None,
        "git_head": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_dirty": bool(diff.stdout) if diff.returncode == 0 else None,
        "git_diff_sha256": sha256_bytes(diff.stdout) if diff.returncode == 0 else None,
    }


def _write_common_evidence(
    output: Path,
    *,
    command: dict[str, object],
    wall_seconds: float,
    cpu_seconds: float,
    stdout: str = "valid-only baseline completed\n",
    environment: dict[str, object] | None = None,
) -> None:
    atomic_write_json(output / "environment.json", environment or _environment_evidence())
    atomic_write_json(output / "command.json", command)
    atomic_write_json(
        output / "telemetry.json",
        {"wall_seconds": wall_seconds, "cpu_seconds": cpu_seconds, "gpu_seconds": 0.0},
    )
    (output / "stdout.log").write_text(stdout, encoding="utf-8")
    (output / "stderr.log").write_text("", encoding="utf-8")


def _valid_inputs(view_dir: str | Path) -> tuple[FeatureView, np.ndarray, FeatureView, np.ndarray]:
    root = Path(view_dir)
    train = load_feature_view(root / "train_features.npz")
    valid = load_feature_view(root / "valid_features.npz")
    train_labels = load_target_view(root / "label_vault/train_targets.npz").labels
    valid_labels = load_target_view(root / "label_vault/valid_targets.npz").labels
    return train, train_labels, valid, valid_labels


def reproduce_random(
    view_dir: str | Path,
    *,
    seed: int = 0,
    evidence_dir: str | Path | None = None,
) -> Metrics:
    """Reproduce the seeded random lower bound on validation only."""

    wall_start, cpu_start = time.perf_counter(), time.process_time()
    environment = _environment_evidence()
    _, _, valid, labels = _valid_inputs(view_dir)
    scores = np.random.default_rng(seed).random(valid.rows)
    metrics = _official_metrics(valid.arrays["user_id"], labels, scores, seed)
    if evidence_dir is not None:
        output = Path(evidence_dir)
        output.mkdir(parents=True, exist_ok=True)
        prediction = write_prediction_artifact(output / "predictions.npz", valid, scores)
        atomic_write_json(output / "metrics.json", metrics.model_dump(by_alias=True))
        atomic_write_json(
            output / "config.json", {"model": "random", "seed": seed, "split": "valid"}
        )
        atomic_write_json(
            output / "artifacts.json",
            {"predictions_sha256": sha256_file(prediction), "feature_sha256": valid.sha256},
        )
        _write_common_evidence(
            output,
            command={
                "operation": "reproduce_random",
                "arguments": {"seed": seed, "split": "valid"},
                "official_evaluator": official_evaluator_command(),
            },
            wall_seconds=time.perf_counter() - wall_start,
            cpu_seconds=time.process_time() - cpu_start,
            environment=environment,
        )
    return metrics


def reproduce_item_popularity(
    view_dir: str | Path,
    *,
    prior_strength: float = 20.0,
    evidence_dir: str | Path | None = None,
) -> Metrics:
    """Fit item popularity on train and evaluate validation only."""

    wall_start, cpu_start = time.perf_counter(), time.process_time()
    environment = _environment_evidence()
    train, train_labels, valid, valid_labels = _valid_inputs(view_dir)
    global_mean = float(np.mean(train_labels)) if len(train_labels) else 0.0
    positive: dict[str, float] = {}
    count: dict[str, int] = {}
    for video, label in zip(train.arrays["video_id"], train_labels, strict=True):
        key = str(video)
        positive[key] = positive.get(key, 0.0) + float(label)
        count[key] = count.get(key, 0) + 1
    scores = np.fromiter(
        (
            (positive.get(str(video), 0.0) + prior_strength * global_mean)
            / (count.get(str(video), 0) + prior_strength)
            for video in valid.arrays["video_id"]
        ),
        dtype=np.float64,
        count=valid.rows,
    )
    metrics = _official_metrics(valid.arrays["user_id"], valid_labels, scores, seed=0)
    if evidence_dir is not None:
        output = Path(evidence_dir)
        output.mkdir(parents=True, exist_ok=True)
        prediction = write_prediction_artifact(output / "predictions.npz", valid, scores)
        atomic_write_json(output / "metrics.json", metrics.model_dump(by_alias=True))
        atomic_write_json(
            output / "config.json",
            {"model": "item_popularity", "prior_strength": prior_strength, "split": "valid"},
        )
        atomic_write_json(
            output / "statistics.json",
            {
                "global_mean": global_mean,
                "items": len(count),
                "predictions_sha256": sha256_file(prediction),
                "feature_sha256": valid.sha256,
            },
        )
        _write_common_evidence(
            output,
            command={
                "operation": "reproduce_item_popularity",
                "arguments": {"prior_strength": prior_strength, "split": "valid"},
                "official_evaluator": official_evaluator_command(),
            },
            wall_seconds=time.perf_counter() - wall_start,
            cpu_seconds=time.process_time() - cpu_start,
            environment=environment,
        )
    return metrics


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
    evidence_dir: str | Path | None = None,
) -> BaselineSeedResult:
    """Reproduce the organizer FM training loop using valid-only early stopping."""

    wall_start, cpu_start = time.perf_counter(), time.process_time()
    environment = _environment_evidence()
    train, train_labels, valid, valid_labels = _valid_inputs(view_dir)
    encoder = CategoricalEncoder.fit(train)
    train_x = encoder.transform(train)
    valid_x = encoder.transform(valid)
    model = FM(encoder.dimension, k=k, lr=lr, l2=l2, seed=seed)
    rng = np.random.default_rng(seed)
    best_primary = -1.0
    best_epoch = 0
    best_state: tuple[np.ndarray, np.ndarray, np.float32, int] | None = None
    best_predictions: np.ndarray | None = None
    bad_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(train.rows)
        losses: list[float] = []
        for offset in range(0, len(order), batch_size):
            batch = order[offset : offset + batch_size]
            losses.append(model.step(train_x[batch], train_labels[batch]))
        predictions = model.predict(valid_x)
        metrics = _official_metrics(valid.arrays["user_id"], valid_labels, predictions, seed)
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "GAUC": metrics.GAUC,
                "nDCG@5": metrics.ndcg5,
                "primary": metrics.primary,
            }
        )
        if metrics.primary > best_primary + 1e-5:
            best_primary = metrics.primary
            best_epoch = epoch
            bad_epochs = 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b), model.t)
            best_predictions = predictions.copy()
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is None or best_predictions is None:  # pragma: no cover
        raise RuntimeError("FM baseline did not complete an epoch")
    model.V, model.W, model.b, model.t = best_state
    final = _official_metrics(valid.arrays["user_id"], valid_labels, best_predictions, seed)
    output: Path | None = None
    prediction_hash: str | None = None
    model_hash: str | None = None
    if evidence_dir is not None:
        output = Path(evidence_dir)
        output.mkdir(parents=True, exist_ok=True)
        model_path = output / "model.npz"
        np.savez_compressed(
            model_path,
            V=model.V,
            W=model.W,
            b=np.asarray([model.b]),
            t=np.asarray([model.t]),
        )
        (output / "encoder.json").write_text(
            json.dumps(encoder.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        prediction_path = write_prediction_artifact(
            output / "predictions.npz", valid, best_predictions
        )
        config = {
            "model": "fm",
            "seed": seed,
            "k": k,
            "lr": lr,
            "l2": l2,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "split": "valid",
        }
        atomic_write_json(output / "config.json", config)
        commit_sha = str(environment.get("git_head") or "").strip()
        if not commit_sha:
            raise RuntimeError("cannot create a baseline model bundle without a Git commit")
        create_model_bundle(
            output,
            model_path,
            plugin=OFFICIAL_FM_PLUGIN,
            seed=seed,
            commit_sha=commit_sha,
            config_sha256=sha256_file(output / "config.json"),
            data_view_sha256=train.sha256,
            features=train,
            member_paths=(model_path, output / "encoder.json"),
        )
        atomic_write_json(output / "metrics.json", final.model_dump(by_alias=True))
        atomic_write_json(
            output / "training.json",
            {"history": history, "best_epoch": best_epoch, "stopped_epoch": len(history)},
        )
        model_hash = sha256_file(model_path)
        prediction_hash = sha256_file(prediction_path)
        atomic_write_json(
            output / "artifacts.json",
            {
                "model_sha256": model_hash,
                "encoder_sha256": sha256_file(output / "encoder.json"),
                "predictions_sha256": prediction_hash,
                "train_feature_sha256": train.sha256,
                "valid_feature_sha256": valid.sha256,
            },
        )
        stdout = "".join(
            f"epoch {item['epoch']} loss {item['loss']:.6f} valid_primary {item['primary']:.9f}\n"
            for item in history
        )
        _write_common_evidence(
            output,
            command={
                "operation": "reproduce_fm_seed",
                "arguments": config,
                "official_evaluator": official_evaluator_command(),
            },
            wall_seconds=time.perf_counter() - wall_start,
            cpu_seconds=time.process_time() - cpu_start,
            stdout=stdout,
            environment=environment,
        )
    return BaselineSeedResult(
        seed=seed,
        metrics=final,
        best_epoch=best_epoch,
        evidence_dir=output,
        prediction_sha256=prediction_hash,
        model_sha256=model_hash,
    )


def reproduce_fm_bundle(
    data_dir: str | Path,
    view_dir: str | Path,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    evidence_dir: str | Path | None = None,
) -> BaselineBundle:
    """Bootstrap sanitized views and reproduce valid seeds; test is never scored."""

    bootstrap_views(data_dir, view_dir)
    evidence = Path(evidence_dir) if evidence_dir is not None else None
    results = tuple(
        reproduce_fm_seed(
            view_dir,
            seed=seed,
            evidence_dir=evidence / f"seed-{seed}" if evidence is not None else None,
        )
        for seed in seeds
    )
    scores = np.asarray([item.metrics.primary for item in results], dtype=np.float64)
    return BaselineBundle(
        results=results,
        mean_primary=float(scores.mean()),
        std_primary=float(scores.std(ddof=0)),
    )


def assess_baseline(
    random_metrics: Metrics,
    popularity_metrics: Metrics,
    fm: BaselineBundle,
    *,
    reference_tolerance: float = 0.001,
    fm_tolerance: float = 0.002,
    max_fm_std: float = 0.0015,
) -> BaselineAcceptance:
    reference = load_benchmark_manifest()["references"]["valid"]
    reasons: list[str] = []
    checks = (
        ("random", random_metrics.primary, float(reference["random"]["primary"]), reference_tolerance),
        (
            "item_popularity",
            popularity_metrics.primary,
            float(reference["item_popularity"]["primary"]),
            reference_tolerance,
        ),
        ("fm", fm.mean_primary, float(reference["fm"]["primary"]), fm_tolerance),
    )
    for name, observed, expected, tolerance in checks:
        if abs(observed - expected) > tolerance:
            reasons.append(
                f"{name} primary drift: expected {expected:.6f} ± {tolerance:.6f}, observed {observed:.6f}"
            )
    if fm.std_primary > max_fm_std:
        reasons.append(
            f"fm seed variation too high: maximum {max_fm_std:.6f}, observed {fm.std_primary:.6f}"
        )
    return BaselineAcceptance(
        accepted=not reasons,
        reasons=tuple(reasons),
        observed={
            "random_primary": random_metrics.primary,
            "item_popularity_primary": popularity_metrics.primary,
            "fm_mean_primary": fm.mean_primary,
            "fm_std_primary": fm.std_primary,
        },
    )


def run_baseline_verification(
    data_dir: str | Path,
    view_dir: str | Path,
    evidence_dir: str | Path,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> BaselineEvidence:
    """Bootstrap views, then run the complete valid-only baseline gate."""

    bootstrap_views(data_dir, view_dir)
    return run_baseline_verification_from_views(view_dir, evidence_dir, seeds=seeds)


def run_baseline_verification_from_views(
    view_dir: str | Path,
    evidence_dir: str | Path,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> BaselineEvidence:
    """Run the baseline gate from already verified, sanitized views.

    Production rehearsals bootstrap and hash these views during preflight.  This
    entry point avoids rebuilding the same arrays before baseline training while
    retaining :func:`run_baseline_verification` for standalone callers.
    """

    root = Path(evidence_dir)
    root.mkdir(parents=True, exist_ok=True)
    random_metrics = reproduce_random(view_dir, evidence_dir=root / "random")
    popularity_metrics = reproduce_item_popularity(
        view_dir, evidence_dir=root / "item-popularity"
    )
    results = tuple(
        reproduce_fm_seed(view_dir, seed=seed, evidence_dir=root / f"seed-{seed}")
        for seed in seeds
    )
    scores = np.asarray([item.metrics.primary for item in results], dtype=np.float64)
    fm = BaselineBundle(results, float(scores.mean()), float(scores.std(ddof=0)))
    acceptance = assess_baseline(random_metrics, popularity_metrics, fm)
    summary = {
        "schema_version": "1.0",
        "development_split": "valid",
        "test_scored": False,
        "random": random_metrics.model_dump(by_alias=True),
        "item_popularity": popularity_metrics.model_dump(by_alias=True),
        "fm": {
            "mean_primary": fm.mean_primary,
            "std_primary": fm.std_primary,
            "seeds": [
                {
                    "seed": result.seed,
                    "best_epoch": result.best_epoch,
                    "metrics": result.metrics.model_dump(by_alias=True),
                    "prediction_sha256": result.prediction_sha256,
                    "model_sha256": result.model_sha256,
                }
                for result in results
            ],
        },
        "acceptance": {
            "accepted": acceptance.accepted,
            "reasons": list(acceptance.reasons),
            "observed": acceptance.observed,
        },
    }
    summary_path = atomic_write_json(root / "summary.json", summary)
    return BaselineEvidence(random_metrics, popularity_metrics, fm, acceptance, summary_path)
