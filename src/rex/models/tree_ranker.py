"""Optional LightGBM LambdaRank plugin with train-fitted categorical encoders."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from rex.data.views import FeatureView, TargetView
from rex.features.static_metadata import NUMERIC_ARRAYS as STATIC_NUMERIC_ARRAYS
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
CORE_CATEGORICAL_NAMES = BASE_FEATURE_NAMES[:4]
BASE_NUMERIC_NAMES = BASE_FEATURE_NAMES[4:]


class TreeRankerPlugin:
    @staticmethod
    def _group_order(users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return a stable query order and LightGBM group sizes."""

        text_users = users.astype(str)
        order = np.argsort(text_users, kind="stable")
        _, group = np.unique(text_users[order], return_counts=True)
        if int(group.sum()) != len(users) or np.any(group <= 0):
            raise ValueError("LightGBM user groups do not cover the view exactly")
        return order, group

    @staticmethod
    def _inner_temporal_masks(
        dates: np.ndarray,
        *,
        validation_days: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Select a strictly later inner slice without consulting its labels."""

        if validation_days <= 0:
            return None
        unique_dates = np.unique(dates)
        if len(unique_dates) < 2:
            return None
        held_out = unique_dates[-min(validation_days, len(unique_dates) - 1) :]
        valid = np.isin(dates, held_out)
        train = ~valid
        if not np.any(train) or not np.any(valid):
            return None
        return train, valid

    @staticmethod
    def _model_parameters(config: dict[str, Any], seed: int, n_estimators: int) -> dict[str, Any]:
        objective = str(config.get("objective", "lambdarank"))
        if objective not in {"lambdarank", "rank_xendcg"}:
            raise ValueError(f"unsupported LightGBM ranking objective: {objective}")
        bagging_fraction = float(config.get("bagging_fraction", 1.0))
        return {
            "objective": objective,
            "metric": "ndcg",
            "lambdarank_truncation_level": int(config.get("truncation_level", 8)),
            "n_estimators": n_estimators,
            "learning_rate": float(config.get("learning_rate", 0.03)),
            "num_leaves": int(config.get("num_leaves", 23)),
            "max_depth": int(config.get("max_depth", -1)),
            "min_child_samples": int(config.get("min_child_samples", 100)),
            "min_split_gain": float(config.get("min_split_gain", 0.0)),
            "reg_alpha": float(config.get("reg_alpha", 0.0)),
            "reg_lambda": float(config.get("reg_lambda", 2.0)),
            "feature_fraction": float(config.get("feature_fraction", 0.9)),
            "bagging_fraction": bagging_fraction,
            "bagging_freq": 1 if bagging_fraction < 1.0 else 0,
            "cat_smooth": float(config.get("cat_smooth", 20.0)),
            "cat_l2": float(config.get("cat_l2", 10.0)),
            "min_data_per_group": int(config.get("min_data_per_group", 100)),
            "max_cat_threshold": int(config.get("max_cat_threshold", 32)),
            "random_state": seed,
            "n_jobs": int(config.get("n_jobs", 4)),
            "deterministic": True,
            "force_col_wise": True,
            "bagging_seed": seed,
            "feature_fraction_seed": seed,
            "data_random_seed": seed,
            "verbosity": -1,
        }

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
    def _categorical_names(
        view: FeatureView,
        vocabs: dict[str, dict[str, int]] | None = None,
        config: dict[str, Any] | None = None,
    ) -> tuple[str, ...]:
        if vocabs is not None:
            extra = sorted(name for name in vocabs if name not in CORE_CATEGORICAL_NAMES)
            return tuple(name for name in (*CORE_CATEGORICAL_NAMES, *extra) if name in vocabs)
        configured = (config or {}).get("categorical_fields")
        if configured is None:
            return CORE_CATEGORICAL_NAMES
        if not isinstance(configured, (list, tuple)) or not configured:
            raise ValueError("tree categorical_fields must be a non-empty list")
        names = tuple(str(name) for name in configured)
        if len(set(names)) != len(names):
            raise ValueError("tree categorical_fields must be unique")
        missing = [name for name in names if name not in view.arrays]
        if missing:
            raise ValueError(f"tree ranker is missing categorical fields: {missing}")
        return names

    @staticmethod
    def _numeric_names(view: FeatureView, config: dict[str, Any] | None = None) -> tuple[str, ...]:
        configured = (config or {}).get("numeric_fields")
        if configured is None:
            return tuple(
                name
                for name in sorted(view.arrays)
                if name.startswith("fx__") and name not in STATIC_NUMERIC_ARRAYS
            )
        if not isinstance(configured, (list, tuple)):
            raise ValueError("tree numeric_fields must be a list")
        names = tuple(str(name) for name in configured)
        if len(set(names)) != len(names):
            raise ValueError("tree numeric_fields must be unique")
        missing = [name for name in names if name not in view.arrays]
        if missing:
            raise ValueError(f"tree ranker is missing numeric fields: {missing}")
        return names

    @staticmethod
    def _feature_names(
        view: FeatureView,
        categorical_names: tuple[str, ...] | None = None,
        numeric_names: tuple[str, ...] | None = None,
    ) -> list[str]:
        categorical_names = categorical_names or TreeRankerPlugin._categorical_names(view)
        numeric_names = numeric_names or TreeRankerPlugin._numeric_names(view)
        return [
            *categorical_names,
            *BASE_NUMERIC_NAMES,
            *numeric_names,
        ]

    @staticmethod
    def _matrix(
        view: FeatureView,
        vocabs: dict[str, dict[str, int]] | None = None,
        config: dict[str, Any] | None = None,
    ):
        fitted = vocabs is None
        vocabs = {} if vocabs is None else vocabs
        categorical = TreeRankerPlugin._categorical_names(
            view, None if fitted else vocabs, config
        )
        columns: list[np.ndarray] = []
        for name in categorical:
            if name not in view.arrays:
                raise ValueError(f"tree ranker is missing fitted categorical field {name}")
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
        numeric = TreeRankerPlugin._numeric_names(view, config)
        columns.extend(view.arrays[name].astype(np.float32) for name in numeric)
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
        matrix, vocabs = self._matrix(train_features, config=config)
        users = train_features.arrays["user_id"]
        order, group = self._group_order(users)
        categorical_names = self._categorical_names(train_features, config=config)
        numeric_names = self._numeric_names(train_features, config)
        feature_names = self._feature_names(train_features, categorical_names, numeric_names)
        configured_estimators = int(config.get("n_estimators", 500))
        selected_estimators = configured_estimators
        inner_evidence: dict[str, Any] = {"enabled": False}
        masks = self._inner_temporal_masks(
            train_features.arrays["date"],
            validation_days=int(config.get("inner_validation_days", 2)),
        )
        if masks is not None and int(config.get("early_stopping_rounds", 40)) > 0:
            inner_train, inner_valid = masks
            tune_train_order, tune_train_group = self._group_order(users[inner_train])
            tune_valid_order, tune_valid_group = self._group_order(users[inner_valid])
            tune_train_matrix = matrix[inner_train]
            tune_valid_matrix = matrix[inner_valid]
            tune_train_labels = train_targets.labels[inner_train].astype(np.int32)
            tune_valid_labels = train_targets.labels[inner_valid].astype(np.int32)
            tuner = lgb.LGBMRanker(
                **self._model_parameters(config, seed, configured_estimators)
            )
            tuner.fit(
                tune_train_matrix[tune_train_order],
                tune_train_labels[tune_train_order],
                group=tune_train_group.tolist(),
                eval_set=[
                    (
                        tune_valid_matrix[tune_valid_order],
                        tune_valid_labels[tune_valid_order],
                    )
                ],
                eval_group=[tune_valid_group.tolist()],
                eval_at=[5],
                feature_name=feature_names,
                categorical_feature=list(categorical_names),
                callbacks=[
                    lgb.early_stopping(
                        int(config.get("early_stopping_rounds", 40)),
                        verbose=False,
                    )
                ],
            )
            selected_estimators = max(1, int(tuner.best_iteration_ or configured_estimators))
            inner_evidence = {
                "enabled": True,
                "train_rows": int(inner_train.sum()),
                "validation_rows": int(inner_valid.sum()),
                "validation_dates": sorted(
                    {int(value) for value in train_features.arrays["date"][inner_valid]}
                ),
                "selected_estimators": selected_estimators,
                "best_score": tuner.best_score_,
            }

        model = lgb.LGBMRanker(**self._model_parameters(config, seed, selected_estimators))
        model.fit(
            matrix[order],
            train_targets.labels[order].astype(np.int32),
            group=group.tolist(),
            eval_at=[5],
            feature_name=feature_names,
            categorical_feature=list(categorical_names),
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
                    "categorical_features": list(categorical_names),
                    "config": config,
                    "configured_estimators": configured_estimators,
                    "selected_estimators": selected_estimators,
                    "inner_temporal_validation": inner_evidence,
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
        matrix, _ = self._matrix(features, vocabs, config)
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
