"""Portable two-branch model selected exclusively on development shadow views."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rex.data.views import FeatureView, TargetView
from rex.models.base import load_plugin
from rex.models.ensemble import blend_scores


class ShadowBlendPlugin:
    """Train and reload the two diverse branches required by method card E10.

    Weight selection is deliberately absent from this worker.  The immutable E10
    config must contain weights chosen from shadow evidence before the official
    validation invocation.  Component predictions are emitted as a diagnostic
    sidecar so the trusted evaluator can compare the blend with its stronger
    input without loading model code in the coordinator process.
    """

    def fit(
        self,
        train_features: FeatureView,
        train_targets: TargetView,
        config: dict[str, Any],
        seed: int,
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        pair_dir = output_dir / "pair"
        tree_dir = output_dir / "tree"
        pair_config = dict(config.get("pair_config", {}))
        pair_config.setdefault("epochs", 8)
        pair_config.setdefault("pair_batch_size", 4096)
        pair_config.setdefault("negatives_per_positive", 3)
        pair_config.setdefault("bce_weight", 0.05)
        tree_config = dict(config.get("tree_config", {}))
        tree_config.setdefault("n_estimators", 300)
        tree_config.setdefault("learning_rate", 0.04)
        tree_config.setdefault("num_leaves", 31)
        tree_config.setdefault("min_child_samples", 50)
        tree_config.setdefault("reg_lambda", 1.0)
        tree_config.setdefault("n_jobs", 1)
        pair_plugin_path = str(
            config.get(
                "pair_plugin",
                "rex.models.experimental.pair_rank_fm:ExperimentalPairRankFMPlugin",
            )
        )
        tree_plugin_path = str(
            config.get(
                "tree_plugin",
                "rex.models.experimental.tree_history:ExperimentalTreeHistoryPlugin",
            )
        )
        pair_model = load_plugin(pair_plugin_path).fit(
            train_features, train_targets, pair_config, seed, pair_dir
        )
        tree_model = load_plugin(tree_plugin_path).fit(
            train_features, train_targets, tree_config, seed, tree_dir
        )
        manifest = output_dir / "blend.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "selection_split": "shadow_only",
                    "normalization": str(config.get("normalization", "percentile")),
                    "weights": [float(value) for value in config.get("weights", [0.5, 0.5])],
                    "pair_model": pair_model.relative_to(output_dir).as_posix(),
                    "tree_model": tree_model.relative_to(output_dir).as_posix(),
                    "pair_config": pair_config,
                    "tree_config": tree_config,
                    "pair_plugin": pair_plugin_path,
                    "tree_plugin": tree_plugin_path,
                    "seed": seed,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest

    def predict(
        self,
        model_artifact: Path,
        features: FeatureView,
        config: dict[str, Any],
        output_dir: Path,
    ) -> np.ndarray:
        del config
        payload = json.loads(model_artifact.read_text(encoding="utf-8"))
        root = model_artifact.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        pair = load_plugin(str(payload["pair_plugin"])).predict(
            root / payload["pair_model"],
            features,
            dict(payload["pair_config"]),
            output_dir / "pair",
        )
        tree = load_plugin(str(payload["tree_plugin"])).predict(
            root / payload["tree_model"],
            features,
            dict(payload["tree_config"]),
            output_dir / "tree",
        )
        components = output_dir / "component_predictions.npz"
        np.savez_compressed(components, pair=pair, tree=tree)
        return blend_scores(
            features.arrays["user_id"],
            [pair, tree],
            np.asarray(payload["weights"], dtype=np.float64),
            normalization=str(payload["normalization"]),
        )
