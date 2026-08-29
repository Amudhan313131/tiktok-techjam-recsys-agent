"""Tiny deterministic plugin used only by fixture and fault-injection rehearsals."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from rex.data.views import FeatureView, TargetView


# Fixture-only patches deliberately change this value to prove that code from an
# isolated worktree, rather than the main checkout, is what the worker imports.
DEFAULT_BIAS = 0.0


class FixturePlugin:
    def fit(
        self,
        train_features: FeatureView,
        train_targets: TargetView,
        config: dict[str, Any],
        seed: int,
        output_dir: Path,
    ) -> Path:
        time.sleep(float(config.get("sleep_seconds", 0)))
        allocation = bytearray(int(config.get("allocate_mb", 0)) * 1024 * 1024)
        if config.get("spawn_child_seconds"):
            child = os.fork()
            if child == 0:  # pragma: no cover - killed with the worker process group
                time.sleep(float(config["spawn_child_seconds"]))
                os._exit(0)
        if config.get("raise_memory_error"):
            raise MemoryError("fixture OOM")
        marker_value = config.get("raise_floating_point_once_marker")
        if marker_value:
            marker = Path(str(marker_value))
            marker.parent.mkdir(parents=True, exist_ok=True)
            try:
                marker.touch(exist_ok=False)
            except FileExistsError:
                pass
            else:
                raise FloatingPointError("fixture controlled one-shot non-finite loss")
        if config.get("raise_floating_point"):
            raise FloatingPointError("fixture non-finite loss")
        if config.get("raise_crash"):
            raise RuntimeError("fixture crash")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "fixture_model.json"
        path.write_text(
            json.dumps(
                {
                    "mean": float(train_targets.labels.mean()),
                    "bias": float(config.get("bias", DEFAULT_BIAS)),
                    "seed": seed,
                    "allocated": len(allocation),
                }
            ),
            encoding="utf-8",
        )
        return path

    def predict(
        self,
        model_artifact: Path,
        features: FeatureView,
        config: dict[str, Any],
        output_dir: Path,
    ) -> np.ndarray:
        if config.get("nan_scores"):
            return np.full(features.rows, np.nan)
        payload = json.loads(model_artifact.read_text(encoding="utf-8"))
        value = float(payload["mean"]) + float(payload.get("bias", 0.0))
        return np.full(features.rows, value, dtype=np.float64)
