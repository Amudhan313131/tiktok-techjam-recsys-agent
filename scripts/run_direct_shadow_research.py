#!/usr/bin/env python3
"""Run reproducible, validation-locked shadow experiments outside the autonomous loop."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rex.agents.provider import FixedQueueProvider
from rex.control.production_supervisor import ProductionContext, ProductionRunConfig
from rex.control.scientific_hooks import (
    REFERENCE_CONFIG_BY_CARD,
    ProductionScientificHooks,
)
from rex.data.manifest import sha256_file
from rex.data.views import load_feature_view, load_target_view
from rex.evaluation.official_adapter import evaluate_arrays
from rex.models.base import load_plugin
from rex.models.bundle import create_model_bundle


SUPPORTED_CARDS = frozenset(
    {
        "E16",
        "E17",
        "E18",
        "E19",
        "E20",
        "E21",
        "E22",
        "E23",
        "E24",
        "E26",
        "E27",
        "E28",
        "E29",
        "E30",
    }
)


def _atomic_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _source_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        return "unavailable"
    commit = completed.stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return f"{commit}-dirty" if status.returncode or status.stdout.strip() else commit


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("plugin"), str):
        raise ValueError(f"invalid experiment config: {path}")
    return dict(value)


def _fit_predict(
    *,
    config_path: Path,
    train_features: Path,
    train_targets: Path,
    apply_features: Path,
    output: Path,
    seed: int,
    commit: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    config = _load_config(config_path)
    plugin = load_plugin(str(config["plugin"]))
    train = load_feature_view(train_features)
    targets = load_target_view(train_targets)
    apply = load_feature_view(apply_features)
    started = time.monotonic()
    fit_dir = output / "fit"
    predict_dir = output / "predict"
    primary = Path(plugin.fit(train, targets, config, seed, fit_dir))
    bundle = create_model_bundle(
        fit_dir,
        primary,
        plugin=str(config["plugin"]),
        seed=seed,
        commit_sha=commit,
        config_sha256=sha256_file(config_path),
        data_view_sha256=train.sha256,
        features=train,
    )
    scores = np.asarray(plugin.predict(primary, apply, config, predict_dir), dtype=np.float64)
    if scores.shape != (apply.rows,) or not np.isfinite(scores).all():
        raise RuntimeError("direct shadow model produced invalid predictions")
    prediction = output / "predictions.npz"
    np.savez_compressed(
        prediction,
        row_id=apply.arrays["row_id"],
        user_id=apply.arrays["user_id"],
        score=scores,
    )
    copied_config = output / "config.yaml"
    shutil.copy2(config_path, copied_config)
    return scores, {
        "plugin": config["plugin"],
        "config_sha256": sha256_file(copied_config),
        "bundle": str(bundle),
        "bundle_sha256": sha256_file(bundle),
        "predictions": str(prediction),
        "predictions_sha256": sha256_file(prediction),
        "wall_seconds": time.monotonic() - started,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.project_root.resolve()
    config = ProductionRunConfig.load(args.config.resolve())
    requested = tuple(dict.fromkeys(args.cards))
    unsupported = sorted(set(requested) - SUPPORTED_CARDS)
    if unsupported:
        raise ValueError(f"unsupported direct shadow cards: {unsupported}")
    if not requested:
        raise ValueError("at least one card is required")
    run_id = args.run_id or f"direct-shadow-{int(time.time())}"
    output = args.output_dir.resolve() / run_id
    if output.exists():
        raise FileExistsError(f"direct research output already exists: {output}")
    output.mkdir(parents=True)
    commit = _source_commit(root)
    context = ProductionContext(
        run_id,
        output,
        root,
        commit,
        int((time.time() + args.deadline_seconds) * 1000),
    )
    hooks = ProductionScientificHooks(
        config,
        FixedQueueProvider([]),
        settings=None,
    )
    partitions = hooks._partition_for_rung(context, args.rung)
    results: dict[str, Any] = {}
    for card_id in requested:
        binding = config.method_cards[card_id]
        reference_relative = REFERENCE_CONFIG_BY_CARD[card_id]
        reference_config = (root / reference_relative).resolve()
        fold_results: dict[str, Any] = {}
        for partition in partitions:
            candidate_views = hooks._views(
                context,
                partition,
                card_id,
                binding.feature_recipe,
                reference=False,
            )
            reference_views = hooks._views(
                context,
                partition,
                card_id,
                binding.feature_recipe,
                reference=True,
            )
            candidate_scores, candidate_artifacts = _fit_predict(
                config_path=binding.config_path,
                train_features=candidate_views.train_features,
                train_targets=partition.train_targets,
                apply_features=candidate_views.apply_features,
                output=output / "experiments" / card_id / partition.name / "candidate",
                seed=args.seed,
                commit=commit,
            )
            reference_scores, reference_artifacts = _fit_predict(
                config_path=reference_config,
                train_features=reference_views.train_features,
                train_targets=partition.train_targets,
                apply_features=reference_views.apply_features,
                output=output / "experiments" / card_id / partition.name / "reference",
                seed=args.seed,
                commit=commit,
            )
            candidate_apply = load_feature_view(candidate_views.apply_features)
            reference_apply = load_feature_view(reference_views.apply_features)
            candidate_rows = candidate_apply.arrays.get(
                "fx__source_row_id", candidate_apply.arrays["row_id"]
            )
            reference_rows = reference_apply.arrays.get(
                "fx__source_row_id", reference_apply.arrays["row_id"]
            )
            if not np.array_equal(candidate_rows, reference_rows):
                raise RuntimeError("candidate and reference rows are not aligned")
            labels = load_target_view(partition.valid_targets).labels
            candidate_metrics = evaluate_arrays(
                candidate_apply.arrays["user_id"],
                labels,
                candidate_scores,
                split="shadow",
                fold=partition.name,
                seed=args.seed,
            )
            reference_metrics = evaluate_arrays(
                reference_apply.arrays["user_id"],
                labels,
                reference_scores,
                split="shadow",
                fold=partition.name,
                seed=args.seed,
            )
            fold_results[partition.name] = {
                "candidate": candidate_metrics.model_dump(mode="json", by_alias=True),
                "reference": reference_metrics.model_dump(mode="json", by_alias=True),
                "primary_delta": candidate_metrics.primary - reference_metrics.primary,
                "candidate_artifacts": candidate_artifacts,
                "reference_artifacts": reference_artifacts,
            }
        deltas = [float(item["primary_delta"]) for item in fold_results.values()]
        results[card_id] = {
            "folds": fold_results,
            "mean_primary_delta": float(np.mean(deltas)),
            "positive_folds": int(sum(value > 0 for value in deltas)),
        }
        _atomic_json(output / "results.partial.json", results)
    report = {
        "schema_version": "rex.direct-shadow-research.v1",
        "run_id": run_id,
        "rung": args.rung,
        "source_commit": commit,
        "seed": args.seed,
        "cards": list(requested),
        "test_scored": False,
        "official_validation_used": False,
        "results": results,
    }
    _atomic_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/run/production.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/direct-research"))
    parser.add_argument("--run-id")
    parser.add_argument("--rung", choices=("cheap", "full"), default="cheap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deadline-seconds", type=int, default=21_600)
    parser.add_argument("cards", nargs="+", help="E16-E24 or E26-E30 cards to evaluate")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
