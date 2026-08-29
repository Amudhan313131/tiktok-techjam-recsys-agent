"""Common model plugin contract and loader."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from rex.data.views import FeatureView, TargetView


class ModelPlugin(Protocol):
    def fit(
        self,
        train_features: FeatureView,
        train_targets: TargetView,
        config: dict[str, Any],
        seed: int,
        output_dir: Path,
    ) -> Path: ...

    def predict(
        self,
        model_artifact: Path,
        features: FeatureView,
        config: dict[str, Any],
        output_dir: Path,
    ) -> np.ndarray: ...


def load_plugin(import_path: str) -> ModelPlugin:
    if ":" not in import_path:
        raise ValueError("plugin path must have module:attribute form")
    module_name, attribute = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    plugin = getattr(module, attribute)
    return plugin() if isinstance(plugin, type) else plugin
