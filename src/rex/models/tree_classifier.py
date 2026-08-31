"""Deterministic pointwise LightGBM candidate with a time-ordered tuning slice."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rex.data.views import FeatureView, TargetView


PLUGIN_PATH = "rex.models.tree_classifier:TreeClassifierPlugin"
CORE_CATEGORICAL_FIELDS = ("user_id", "video_id", "author_id", "tab")
BASE_NUMERIC_FEATURES = ("duration", "watch_threshold", "date_offset")


class TreeClassifierPlugin:
    """Regularized binary classifier used as a prediction-diverse tree candidate.

    Static metadata and engineered arrays are never discovered implicitly. A caller
    must name every optional input in ``categorical_fields`` or ``numeric_fields``.
    """

    @staticmethod
    def _inner_temporal_masks(
        dates: np.ndarray,
        *,
        validation_days: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return an earlier tuning train slice and a strictly later validation slice."""

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
        if np.max(dates[train]) >= np.min(dates[valid]):
            raise ValueError("inner validation must be strictly later than inner training")
        return train, valid

    @staticmethod
    def _date_offsets(values: np.ndarray) -> np.ndarray:
        dates: list[np.datetime64] = []
        for value in values:
            text = str(int(value))
            if len(text) != 8:
                raise ValueError(f"invalid YYYYMMDD date for tree classifier: {value}")
            dates.append(np.datetime64(f"{text[:4]}-{text[4:6]}-{text[6:]}", "D"))
        observed = np.asarray(dates, dtype="datetime64[D]")
        return (observed - np.datetime64("2022-04-08", "D")).astype(np.float32) / 30.0

    @staticmethod
    def _configured_fields(
        view: FeatureView,
        config: dict[str, Any],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        configured_categorical = config.get("categorical_fields", CORE_CATEGORICAL_FIELDS)
        configured_numeric = config.get("numeric_fields", ())
        if not isinstance(configured_categorical, (list, tuple)) or not configured_categorical:
            raise ValueError("tree classifier categorical_fields must be a non-empty list")
        if not isinstance(configured_numeric, (list, tuple)):
            raise ValueError("tree classifier numeric_fields must be a list")
        categorical = tuple(str(name) for name in configured_categorical)
        numeric = tuple(str(name) for name in configured_numeric)
        if len(set(categorical)) != len(categorical):
            raise ValueError("tree classifier categorical_fields must be unique")
        if len(set(numeric)) != len(numeric):
            raise ValueError("tree classifier numeric_fields must be unique")
        overlap = set(categorical).intersection(numeric)
        if overlap:
            raise ValueError(
                "tree classifier fields cannot be both categorical and numeric: "
                f"{sorted(overlap)}"
            )
        missing = [name for name in (*categorical, *numeric) if name not in view.arrays]
        if missing:
            raise ValueError(f"tree classifier is missing explicitly configured fields: {missing}")
        for name in (*categorical, *numeric):
            array = view.arrays[name]
            if array.ndim != 1 or len(array) != view.rows:
                raise ValueError(f"tree classifier field must be a row-aligned vector: {name}")
        for name in numeric:
            array = view.arrays[name]
            if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
                raise ValueError(f"tree classifier numeric field must be finite numeric data: {name}")
        return categorical, numeric

    @staticmethod
    def _feature_names(
        categorical_fields: tuple[str, ...],
        numeric_fields: tuple[str, ...],
    ) -> list[str]:
        return [*categorical_fields, *BASE_NUMERIC_FEATURES, *numeric_fields]

    @staticmethod
    def _fit_vocabs(
        view: FeatureView,
        categorical_fields: tuple[str, ...],
    ) -> dict[str, dict[str, int]]:
        vocabs: dict[str, dict[str, int]] = {}
        for name in categorical_fields:
            mapping: dict[str, int] = {}
            for value in view.arrays[name]:
                key = str(value)
                if key not in mapping:
                    mapping[key] = len(mapping)
            vocabs[name] = mapping
        return vocabs

    @classmethod
    def _matrix(
        cls,
        view: FeatureView,
        *,
        categorical_fields: tuple[str, ...],
        numeric_fields: tuple[str, ...],
        vocabs: dict[str, dict[str, int]] | None = None,
    ) -> tuple[np.ndarray, dict[str, dict[str, int]]]:
        fitted_vocabs = (
            cls._fit_vocabs(view, categorical_fields) if vocabs is None else vocabs
        )
        if set(fitted_vocabs) != set(categorical_fields):
            raise ValueError("saved categorical vocabularies do not match the fitted field list")
        columns: list[np.ndarray] = []
        for name in categorical_fields:
            if name not in view.arrays:
                raise ValueError(f"tree classifier is missing fitted categorical field: {name}")
            mapping = fitted_vocabs[name]
            unknown = len(mapping)
            columns.append(
                np.fromiter(
                    (mapping.get(str(value), unknown) for value in view.arrays[name]),
                    dtype=np.float32,
                    count=view.rows,
                )
            )
        duration = view.arrays["duration_ms"].astype(np.float32) / 18_000.0
        threshold = (
            np.minimum(view.arrays["duration_ms"], 18_000).astype(np.float32) / 18_000.0
        )
        columns.extend((duration, threshold, cls._date_offsets(view.arrays["date"])))
        for name in numeric_fields:
            if name not in view.arrays:
                raise ValueError(f"tree classifier is missing fitted numeric field: {name}")
            values = view.arrays[name]
            if values.dtype.kind not in "biuf" or not np.isfinite(values).all():
                raise ValueError(f"tree classifier numeric field must be finite numeric data: {name}")
            columns.append(values.astype(np.float32))
        matrix = np.column_stack(columns).astype(np.float32)
        if matrix.shape != (view.rows, len(cls._feature_names(categorical_fields, numeric_fields))):
            raise ValueError("tree classifier matrix does not match its saved feature names")
        if not np.isfinite(matrix).all():
            raise ValueError("tree classifier matrix contains non-finite values")
        return matrix, fitted_vocabs

    @staticmethod
    def _subset(view: FeatureView, mask: np.ndarray) -> FeatureView:
        return FeatureView(
            view.path,
            {name: values[mask] for name, values in view.arrays.items()},
            view.sha256,
        )

    @staticmethod
    def _model_parameters(
        config: dict[str, Any],
        seed: int,
        n_estimators: int,
    ) -> dict[str, Any]:
        bagging_fraction = float(config.get("bagging_fraction", 0.9))
        return {
            "objective": "binary",
            "metric": "binary_logloss",
            "n_estimators": n_estimators,
            "learning_rate": float(config.get("learning_rate", 0.03)),
            "num_leaves": int(config.get("num_leaves", 23)),
            "max_depth": int(config.get("max_depth", -1)),
            "min_child_samples": int(config.get("min_child_samples", 100)),
            "min_split_gain": float(config.get("min_split_gain", 0.0)),
            "reg_alpha": float(config.get("reg_alpha", 0.5)),
            "reg_lambda": float(config.get("reg_lambda", 5.0)),
            "feature_fraction": float(config.get("feature_fraction", 0.85)),
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
            raise ImportError(
                "install the optional 'tree' dependencies to use TreeClassifierPlugin"
            ) from error
        if train_features.rows != len(train_targets.labels):
            raise ValueError("tree classifier training features and targets are misaligned")
        if not np.isin(train_targets.labels, (0.0, 1.0)).all():
            raise ValueError("tree classifier requires binary training targets")
        output_dir.mkdir(parents=True, exist_ok=True)
        categorical_fields, numeric_fields = self._configured_fields(train_features, config)
        feature_names = self._feature_names(categorical_fields, numeric_fields)
        configured_estimators = int(config.get("n_estimators", 500))
        if configured_estimators <= 0:
            raise ValueError("tree classifier n_estimators must be positive")
        selected_estimators = configured_estimators
        inner_evidence: dict[str, Any] = {"enabled": False}
        masks = self._inner_temporal_masks(
            train_features.arrays["date"],
            validation_days=int(config.get("inner_validation_days", 2)),
        )
        early_stopping_rounds = int(config.get("early_stopping_rounds", 40))
        if masks is not None and early_stopping_rounds > 0:
            inner_train, inner_valid = masks
            tuning_train = self._subset(train_features, inner_train)
            tuning_valid = self._subset(train_features, inner_valid)
            tuning_train_matrix, tuning_vocabs = self._matrix(
                tuning_train,
                categorical_fields=categorical_fields,
                numeric_fields=numeric_fields,
            )
            tuning_valid_matrix, _ = self._matrix(
                tuning_valid,
                categorical_fields=categorical_fields,
                numeric_fields=numeric_fields,
                vocabs=tuning_vocabs,
            )
            tuner = lgb.LGBMClassifier(
                **self._model_parameters(config, seed, configured_estimators)
            )
            tuner.fit(
                tuning_train_matrix,
                train_targets.labels[inner_train].astype(np.int32),
                eval_set=[
                    (
                        tuning_valid_matrix,
                        train_targets.labels[inner_valid].astype(np.int32),
                    )
                ],
                eval_metric="binary_logloss",
                feature_name=feature_names,
                categorical_feature=list(categorical_fields),
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
            selected_estimators = max(1, int(tuner.best_iteration_ or configured_estimators))
            inner_evidence = {
                "enabled": True,
                "train_rows": int(inner_train.sum()),
                "validation_rows": int(inner_valid.sum()),
                "training_max_date": int(np.max(train_features.arrays["date"][inner_train])),
                "validation_min_date": int(np.min(train_features.arrays["date"][inner_valid])),
                "validation_dates": sorted(
                    {int(value) for value in train_features.arrays["date"][inner_valid]}
                ),
                "selected_estimators": selected_estimators,
                "best_score": tuner.best_score_,
                "tuning_vocab_sizes": {
                    name: len(mapping) for name, mapping in sorted(tuning_vocabs.items())
                },
            }

        matrix, vocabs = self._matrix(
            train_features,
            categorical_fields=categorical_fields,
            numeric_fields=numeric_fields,
        )
        model = lgb.LGBMClassifier(
            **self._model_parameters(config, seed, selected_estimators)
        )
        model.fit(
            matrix,
            train_targets.labels.astype(np.int32),
            feature_name=feature_names,
            categorical_feature=list(categorical_fields),
        )
        model_path = output_dir / "model.txt"
        model.booster_.save_model(str(model_path))
        (output_dir / "vocabs.json").write_text(
            json.dumps(vocabs, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output_dir / "feature_spec.json").write_text(
            json.dumps(
                {
                    "categorical_fields": list(categorical_fields),
                    "numeric_fields": list(numeric_fields),
                    "feature_names": feature_names,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (output_dir / "training.json").write_text(
            json.dumps(
                {
                    "seed": seed,
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
            raise ImportError(
                "install the optional 'tree' dependencies to use TreeClassifierPlugin"
            ) from error
        feature_spec = json.loads(
            (model_artifact.parent / "feature_spec.json").read_text(encoding="utf-8")
        )
        categorical_fields = tuple(feature_spec["categorical_fields"])
        numeric_fields = tuple(feature_spec["numeric_fields"])
        expected_names = self._feature_names(categorical_fields, numeric_fields)
        if feature_spec.get("feature_names") != expected_names:
            raise ValueError("saved tree classifier feature names are inconsistent")
        configured_fields = self._configured_fields(features, config)
        if configured_fields != (categorical_fields, numeric_fields):
            raise ValueError("tree classifier prediction fields differ from the fitted fields")
        vocabs = json.loads((model_artifact.parent / "vocabs.json").read_text(encoding="utf-8"))
        matrix, _ = self._matrix(
            features,
            categorical_fields=categorical_fields,
            numeric_fields=numeric_fields,
            vocabs=vocabs,
        )
        booster = lgb.Booster(model_file=str(model_artifact))
        scores = np.asarray(booster.predict(matrix), dtype=np.float64)
        if scores.ndim != 1 or len(scores) != features.rows or not np.isfinite(scores).all():
            raise FloatingPointError("TreeClassifier produced non-finite or misaligned predictions")
        return scores
