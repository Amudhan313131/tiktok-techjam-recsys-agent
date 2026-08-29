"""Optional LightGBM LambdaRank plugin with train-fitted categorical encoders."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from rex.data.views import FeatureView, TargetView
from rex.models.bundle import create_model_bundle, validate_model_bundle


PLUGIN_PATH = "rex.models.tree_ranker:TreeRankerPlugin"
BASE_FEATURE_NAMES = (
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration",
    "watch_threshold",
    "date_offset",
)


class TreeRankerPlugin:
    @staticmethod
    def _date_offsets(values: np.ndarray) -> np.ndarray:
        dates: list[np.datetime64] = []
        for value in values:
            text = str(int(value))
            if len(text) != 8:
                raise ValueError(f"invalid YYYYMMDD date for tree ranker: {value}")
            dates.append(np.datetime64(f"{text[:4]}-{text[4:6]}-{text[6:]}", "D"))
        observed = np.asarray(dates, dtype="datetime64[D]")
        return (observed - np.datetime64("2022-04-08", "D")).astype(np.float32) / 30.0

    @staticmethod
    def _feature_names(view: FeatureView) -> list[str]:
        return [*BASE_FEATURE_NAMES, *(name for name in sorted(view.arrays) if name.startswith("fx__"))]

    @staticmethod
    def _matrix(view: FeatureView, vocabs: dict[str, dict[str, int]] | None = None):
        categorical = ("user_id", "video_id", "author_id", "tab")
        fitted = vocabs is None
        vocabs = {} if vocabs is None else vocabs
        columns: list[np.ndarray] = []
        for name in categorical:
            if fitted:
                mapping: dict[str, int] = {}
                for value in view.arrays[name]:
                    key = str(value)
                    if key not in mapping:
                        mapping[key] = len(mapping)
                vocabs[name] = mapping
            mapping = vocabs[name]
            columns.append(
                np.fromiter(
                    (mapping.get(str(value), len(mapping)) for value in view.arrays[name]),
                    dtype=np.float32,
                    count=view.rows,
                )
            )
        duration = view.arrays["duration_ms"].astype(np.float32) / 18_000.0
        threshold = np.minimum(view.arrays["duration_ms"], 18_000).astype(np.float32) / 18_000.0
        date_offset = TreeRankerPlugin._date_offsets(view.arrays["date"])
        columns.extend((duration, threshold, date_offset))
        columns.extend(
            view.arrays[name].astype(np.float32)
            for name in sorted(view.arrays)
            if name.startswith("fx__")
        )
        return np.column_stack(columns).astype(np.float32), vocabs

    def fit(
        self,
        train_features: FeatureView,
        train_targets: TargetView,
        config: dict[str, Any],
        seed: int,
        output_dir: Path,
    ) -> Path:
        try:
            import lightgbm as lgb
        except ImportError as error:
            raise ImportError("install the optional 'tree' dependencies to use TreeRankerPlugin") from error
        output_dir.mkdir(parents=True, exist_ok=True)
        matrix, vocabs = self._matrix(train_features)
        users = train_features.arrays["user_id"]
        order = np.argsort(users.astype(str), kind="stable")
        sorted_users = users.astype(str)[order]
        _, group = np.unique(sorted_users, return_counts=True)
        if int(group.sum()) != train_features.rows or np.any(group <= 0):
            raise ValueError("LightGBM user groups do not cover the training view exactly")
        feature_names = self._feature_names(train_features)
        categorical_names = list(BASE_FEATURE_NAMES[:4])
        model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            lambdarank_truncation_level=int(config.get("truncation_level", 5)),
            n_estimators=int(config.get("n_estimators", 300)),
            learning_rate=float(config.get("learning_rate", 0.04)),
            num_leaves=int(config.get("num_leaves", 31)),
            min_child_samples=int(config.get("min_child_samples", 50)),
            reg_lambda=float(config.get("reg_lambda", 1.0)),
            random_state=seed,
            n_jobs=int(config.get("n_jobs", 4)),
            deterministic=True,
            force_col_wise=True,
            bagging_seed=seed,
            feature_fraction_seed=seed,
            data_random_seed=seed,
            verbosity=-1,
        )
        model.fit(
            matrix[order],
            train_targets.labels[order].astype(np.int32),
            group=group.tolist(),
            eval_at=[5],
            feature_name=feature_names,
            categorical_feature=categorical_names,
        )
        model_path = output_dir / "model.txt"
        model.booster_.save_model(str(model_path))
        (output_dir / "vocabs.json").write_text(json.dumps(vocabs, sort_keys=True), encoding="utf-8")
        (output_dir / "training.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "groups": group.tolist(),
                    "feature_names": feature_names,
                    "categorical_features": categorical_names,
                    "config": config,
                    "lightgbm_version": lgb.__version__,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return model_path

    def predict(
        self,
        model_artifact: Path,
        features: FeatureView,
        config: dict[str, Any],
        output_dir: Path,
    ) -> np.ndarray:
        try:
            import lightgbm as lgb
        except ImportError as error:
            raise ImportError("install the optional 'tree' dependencies to use TreeRankerPlugin") from error
        vocabs = json.loads((model_artifact.parent / "vocabs.json").read_text(encoding="utf-8"))
        matrix, _ = self._matrix(features, vocabs)
        booster = lgb.Booster(model_file=str(model_artifact))
        scores = np.asarray(booster.predict(matrix), dtype=np.float64)
        if scores.ndim != 1 or len(scores) != features.rows or not np.isfinite(scores).all():
            raise FloatingPointError("TreeRanker produced non-finite or misaligned predictions")
        return scores


def tree_ranker_doctor() -> dict[str, Any]:
    """Run a tiny deterministic grouped fit and bundle round-trip."""

    try:
        import lightgbm as lgb
    except ImportError as error:
        raise ImportError("LightGBM 4.7.0 is not installed; install the 'tree' extra") from error

    rows = 12
    arrays = {
        "row_id": np.arange(rows, dtype=np.int64),
        "date": np.asarray([20220408, 20220409, 20220410] * 4, dtype=np.int32),
        "user_id": np.repeat(np.asarray(["u1", "u2", "u3", "u4"]), 3),
        "video_id": np.asarray(["v1", "v2", "v3"] * 4),
        "author_id": np.asarray(["a1", "a2", "a3"] * 4),
        "tab": np.asarray(["1", "1", "2"] * 4),
        "duration_ms": np.asarray([8000, 15000, 24000] * 4, dtype=np.float32),
    }
    features = FeatureView(Path("synthetic-tree-doctor.npz"), arrays, "0" * 64)
    targets = TargetView(
        Path("synthetic-tree-doctor-targets.npz"),
        np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0], dtype=np.float32),
        "1" * 64,
    )
    config: dict[str, Any] = {
        "n_estimators": 12,
        "learning_rate": 0.1,
        "num_leaves": 7,
        "min_child_samples": 1,
        "reg_lambda": 1.0,
        "n_jobs": 1,
    }
    plugin = TreeRankerPlugin()
    config_hash = "2" * 64
    with tempfile.TemporaryDirectory(prefix="rex-tree-doctor-") as temporary:
        root = Path(temporary)
        first = root / "first"
        primary = plugin.fit(features, targets, config, 17, first)
        bundle_path = create_model_bundle(
            first,
            primary,
            plugin=PLUGIN_PATH,
            seed=17,
            commit_sha="tree-doctor",
            config_sha256=config_hash,
            data_view_sha256=features.sha256,
            features=features,
        )
        bundle = validate_model_bundle(
            bundle_path,
            expected_plugin=PLUGIN_PATH,
            expected_config_sha256=config_hash,
            expected_commit_sha="tree-doctor",
            expected_features=features,
        )
        first_scores = plugin.predict(bundle.primary_path, features, config, root / "predict")

        second = root / "second"
        second_primary = plugin.fit(features, targets, config, 17, second)
        second_scores = plugin.predict(second_primary, features, config, root / "predict-second")
        if not np.array_equal(first_scores, second_scores):
            raise RuntimeError("LightGBM deterministic repeat check failed")
        if not {"model.txt", "vocabs.json", "training.json"}.issubset(
            {member.name for member in bundle.manifest.members}
        ):
            raise RuntimeError("LightGBM bundle is missing a required member")
        return {
            "ok": True,
            "version": lgb.__version__,
            "rows": rows,
            "groups": 4,
            "bundle_members": len(bundle.manifest.members),
            "deterministic": True,
        }
