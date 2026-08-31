"""Versioned feature recipes with deterministic, provenance-covered caches."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np

from rex.data.manifest import canonical_json_bytes, sha256_bytes, sha256_file
from rex.data.views import load_feature_view, load_target_view
from rex.features.base import FeatureBundle, attach_feature_bundles
from rex.features.history_summaries import (
    apply_candidate_history_state,
    candidate_history_summaries,
    fit_candidate_history_state,
)
from rex.features.candidate_recency import (
    apply_candidate_recency_state,
    candidate_recency_features,
    fit_candidate_recency_state,
)
from rex.features.categorical_crosses import (
    CategoricalCrossSpec,
    apply_categorical_crosses,
    fit_categorical_crosses,
)
from rex.features.multifeedback_history import (
    apply_multifeedback_state,
    fit_multifeedback_state,
    multifeedback_history_features,
)
from rex.features.quantile_buckets import (
    apply_quantile_buckets,
    fit_quantile_buckets,
)
from rex.features.repeat_exposure import (
    apply_repeat_exposure_state,
    fit_repeat_exposure_state,
    repeat_exposure_features,
)
from rex.features.recency import apply_recency_state, fit_recency_state, recency_history_features
from rex.features.temporal_aggregates import (
    apply_entity_statistics,
    expanding_target_rate,
    fit_entity_statistics,
)


RecipeBuilder = Literal[
    "video_statistics",
    "history_length",
    "candidate_affinity",
    "candidate_history",
    "repeat_exposure",
    "recency_history",
    "candidate_recency",
    "candidate_recency_buckets",
    "multifeedback_history",
    "categorical_cross",
]


@dataclass(frozen=True)
class FeatureRecipe:
    name: str
    version: str
    builder: RecipeBuilder
    output_features: tuple[str, ...]
    cutoff: str
    params: dict[str, float | int | str] = field(default_factory=dict)
    control: bool = False

    @property
    def identity_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(asdict(self)))


@dataclass(frozen=True)
class RecipeArtifact:
    root: Path
    train_features: Path
    apply_features: Path
    manifest: Path
    identity_sha256: str


@dataclass(frozen=True)
class CompositeFeatureRecipe:
    """An ordered set of independently controlled feature mechanisms."""

    name: str
    version: str
    components: tuple[FeatureRecipe, ...]

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("composite feature recipe requires at least one component")
        output_names = [name for component in self.components for name in component.output_features]
        duplicates = sorted({name for name in output_names if output_names.count(name) > 1})
        if duplicates:
            raise ValueError(f"composite recipe output collision: {duplicates}")

    @property
    def identity_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(asdict(self)))


RecipeDefinition: TypeAlias = FeatureRecipe | CompositeFeatureRecipe


VIDEO_STATISTICS = FeatureRecipe(
    name="video_statistics",
    version="1.1",
    builder="video_statistics",
    output_features=(
        "video_target_rate",
        "video_prior_impressions",
        "author_target_rate",
        "author_prior_impressions",
    ),
    cutoff="strictly earlier training date; frozen before evaluation",
    params={"prior_strength": 20.0},
)
CANDIDATE_HISTORY = FeatureRecipe(
    name="candidate_history",
    version="1.0",
    builder="candidate_history",
    output_features=("user_author_rate", "user_duration_rate", "user_history_length"),
    cutoff="strictly earlier training date; frozen before evaluation",
    params={"prior_strength": 10.0},
)
HISTORY_LENGTH = FeatureRecipe(
    name="history_length",
    version="1.0",
    builder="history_length",
    output_features=("user_history_length",),
    cutoff="strictly earlier training date; frozen before evaluation",
    params={"prior_strength": 10.0},
)
AUTHOR_DURATION_AFFINITY = FeatureRecipe(
    name="author_duration_affinity",
    version="1.0",
    builder="candidate_affinity",
    output_features=("user_author_rate", "user_duration_rate"),
    cutoff="strictly earlier training date; frozen before evaluation; tag unavailable in safe schema",
    params={"prior_strength": 10.0},
)
REPEAT_EXPOSURE = FeatureRecipe(
    name="repeat_exposure",
    version="1.0",
    builder="repeat_exposure",
    output_features=(
        "repeat_prior_count",
        "repeat_prior_positive",
        "repeat_last_outcome",
        "repeat_days_since",
    ),
    cutoff="strictly earlier training date; frozen before evaluation",
)
RECENCY_HISTORY = FeatureRecipe(
    name="recency_history",
    version="1.0",
    builder="recency_history",
    output_features=("recency_user_rate", "recency_user_count"),
    cutoff="strictly earlier training date; frozen before evaluation",
    params={"half_life_days": 7.0, "prior_strength": 10.0},
)
POINT_IN_TIME_CANDIDATE_RECENCY = FeatureRecipe(
    name="point_in_time_candidate_recency",
    version="1.0",
    builder="candidate_recency",
    output_features=(
        "pt_user_author_rate",
        "pt_user_author_count",
        "pt_user_author_days_since",
        "pt_user_video_count",
        "pt_user_video_positive",
        "pt_user_video_last_outcome",
        "pt_user_video_days_since",
        "pt_user_rate_h1p0",
        "pt_user_count_h1p0",
        "pt_user_rate_h3p0",
        "pt_user_count_h3p0",
        "pt_user_rate_h7p0",
        "pt_user_count_h7p0",
        "pt_user_rate_h14p0",
        "pt_user_count_h14p0",
    ),
    cutoff="strictly earlier timestamp group; frozen outcomes before evaluation",
    params={
        "half_lives_days": "1,3,7,14",
        "candidate_half_life_days": 7.0,
        "prior_strength": 10.0,
    },
)
MULTIFEEDBACK_HISTORY = FeatureRecipe(
    name="multifeedback_history",
    version="1.0",
    builder="multifeedback_history",
    output_features=(
        "pt_feedback_user_count",
        "pt_feedback_user_click_rate",
        "pt_feedback_user_like_rate",
        "pt_feedback_user_follow_rate",
        "pt_feedback_user_hate_rate",
        "pt_feedback_user_long_view_rate",
        "pt_feedback_user_author_count",
        "pt_feedback_user_author_click_rate",
        "pt_feedback_user_author_like_rate",
        "pt_feedback_user_author_long_view_rate",
        "pt_feedback_video_count",
        "pt_feedback_video_click_rate",
        "pt_feedback_video_long_view_rate",
        "pt_feedback_author_count",
        "pt_feedback_author_click_rate",
        "pt_feedback_author_long_view_rate",
    ),
    cutoff="strictly earlier timestamp group; frozen outcomes before evaluation",
    params={"prior_strength": 20.0},
)
RICH_TEMPORAL_HISTORY = CompositeFeatureRecipe(
    name="rich_temporal_history",
    version="1.0",
    components=(POINT_IN_TIME_CANDIDATE_RECENCY, MULTIFEEDBACK_HISTORY),
)
CANDIDATE_RECENCY_BUCKET_FIELDS = (
    "pt_user_author_rate",
    "pt_user_author_count",
    "pt_user_author_days_since",
    "pt_user_video_count",
    "pt_user_video_positive",
    "pt_user_video_days_since",
    "pt_user_rate_h1p0",
    "pt_user_rate_h7p0",
)
CANDIDATE_RECENCY_BUCKETS = FeatureRecipe(
    name="candidate_recency_buckets",
    version="1.0",
    builder="candidate_recency_buckets",
    output_features=tuple(f"bucket__{name}" for name in CANDIDATE_RECENCY_BUCKET_FIELDS),
    cutoff=(
        "strictly earlier timestamp history; categorical boundaries fitted only on the "
        "training partition and frozen before evaluation"
    ),
    params={
        "half_lives_days": "1,3,7,14",
        "candidate_half_life_days": 7.0,
        "prior_strength": 10.0,
        "quantile_bins": 8,
        "count_cap": 64,
    },
)
USER_TAB_CROSS = FeatureRecipe(
    name="user_tab_cross",
    version="1.0",
    builder="categorical_cross",
    output_features=("user_tab_cross",),
    cutoff="train-fitted support vocabulary; no targets",
    params={"left": "user_id", "right": "tab", "min_count": 3},
)
VIDEO_TAB_CROSS = FeatureRecipe(
    name="video_tab_cross",
    version="1.0",
    builder="categorical_cross",
    output_features=("video_tab_cross",),
    cutoff="train-fitted support vocabulary; no targets",
    params={"left": "video_id", "right": "tab", "min_count": 3},
)


def control_recipe(recipe: FeatureRecipe) -> FeatureRecipe:
    return FeatureRecipe(
        name=f"{recipe.name}_control",
        version=recipe.version,
        builder=recipe.builder,
        output_features=recipe.output_features,
        cutoff=recipe.cutoff,
        params=recipe.params,
        control=True,
    )


def _renamed(bundle: FeatureBundle, mapping: dict[str, str]) -> FeatureBundle:
    return FeatureBundle(
        arrays={mapping.get(name, name): value for name, value in bundle.arrays.items()},
        provenance={mapping.get(name, name): value for name, value in bundle.provenance.items()},
    )


def _selected(bundle: FeatureBundle, names: tuple[str, ...]) -> FeatureBundle:
    return FeatureBundle(
        arrays={name: bundle.arrays[name] for name in names},
        provenance={name: bundle.provenance[name] for name in names},
    )


def _recipe_provenance(bundle: FeatureBundle, recipe: FeatureRecipe) -> FeatureBundle:
    return FeatureBundle(
        arrays=bundle.arrays,
        provenance={
            name: {
                **details,
                "recipe": recipe.name,
                "recipe_version": recipe.version,
                "recipe_cutoff": recipe.cutoff,
                "control": recipe.control,
            }
            for name, details in bundle.provenance.items()
        },
    )


def _definition_provenance(
    bundle: FeatureBundle, recipe: RecipeDefinition
) -> FeatureBundle:
    if isinstance(recipe, FeatureRecipe):
        return _recipe_provenance(bundle, recipe)
    return FeatureBundle(
        arrays=bundle.arrays,
        provenance={
            name: {
                **details,
                "composite_recipe": recipe.name,
                "composite_recipe_version": recipe.version,
                "component_identities": [
                    component.identity_sha256 for component in recipe.components
                ],
            }
            for name, details in bundle.provenance.items()
        },
    )


def _zero_bundles(recipe: FeatureRecipe, train_rows: int, apply_rows: int) -> tuple[FeatureBundle, FeatureBundle]:
    provenance = {
        name: {"cutoff": recipe.cutoff, "control": True, "recipe": recipe.name}
        for name in recipe.output_features
    }
    return (
        FeatureBundle(
            {name: np.zeros(train_rows, dtype=np.float32) for name in recipe.output_features},
            provenance,
        ),
        FeatureBundle(
            {name: np.zeros(apply_rows, dtype=np.float32) for name in recipe.output_features},
            provenance,
        ),
    )


def _temporal_array(view, name: str) -> np.ndarray:
    aliases = {
        "source_row_key": ("source_row_key", "source_global_row_key", "fx__source_row_key"),
        "time_ms": ("time_ms", "fx__time_ms"),
    }
    for candidate in aliases[name]:
        if candidate in view.arrays:
            return view.arrays[candidate]
    raise ValueError(f"feature view is missing required temporal column {name}")


def _load_auxiliary_targets(path: Path | None, expected_rows: int) -> dict[str, np.ndarray]:
    if path is None:
        raise ValueError("multifeedback_history requires an auxiliary target view")
    with np.load(path, allow_pickle=False) as saved:
        required = ("is_click", "is_like", "is_follow", "is_hate", "long_view")
        missing = [name for name in required if name not in saved.files]
        if missing:
            raise ValueError(f"auxiliary target view is missing: {missing}")
        result = {name: np.asarray(saved[name], dtype=np.float32) for name in required}
    if any(value.shape != (expected_rows,) for value in result.values()):
        raise ValueError("auxiliary target view rows differ from training features")
    return result


def _build(
    recipe: FeatureRecipe,
    train_path: Path,
    target_path: Path,
    apply_path: Path,
    auxiliary_target_path: Path | None = None,
) -> tuple[FeatureBundle, FeatureBundle]:
    train = load_feature_view(train_path)
    targets = load_target_view(target_path)
    apply = load_feature_view(apply_path)
    if train.rows != len(targets.labels):
        raise ValueError("feature recipe train/target lengths differ")
    if recipe.control:
        return _zero_bundles(recipe, train.rows, apply.rows)
    if recipe.builder == "video_statistics":
        prior = float(recipe.params.get("prior_strength", 20.0))
        train_video = _renamed(
            expanding_target_rate(
                train.arrays["video_id"],
                train.arrays["date"],
                train.arrays["row_id"],
                targets.labels,
                prior_strength=prior,
            ),
            {"target_rate": "video_target_rate", "prior_impressions": "video_prior_impressions"},
        )
        train_author = _renamed(
            expanding_target_rate(
                train.arrays["author_id"],
                train.arrays["date"],
                train.arrays["row_id"],
                targets.labels,
                prior_strength=prior,
            ),
            {
                "target_rate": "author_target_rate",
                "prior_impressions": "author_prior_impressions",
            },
        )
        video_statistics = fit_entity_statistics(
            train.arrays["video_id"], targets.labels, prior_strength=prior
        )
        author_statistics = fit_entity_statistics(
            train.arrays["author_id"], targets.labels, prior_strength=prior
        )
        apply_video = _renamed(
            apply_entity_statistics(apply.arrays["video_id"], video_statistics),
            {"target_rate": "video_target_rate", "prior_impressions": "video_prior_impressions"},
        )
        apply_author = _renamed(
            apply_entity_statistics(apply.arrays["author_id"], author_statistics),
            {
                "target_rate": "author_target_rate",
                "prior_impressions": "author_prior_impressions",
            },
        )
        return (
            FeatureBundle(
                arrays={**train_video.arrays, **train_author.arrays},
                provenance={**train_video.provenance, **train_author.provenance},
            ),
            FeatureBundle(
                arrays={**apply_video.arrays, **apply_author.arrays},
                provenance={**apply_video.provenance, **apply_author.provenance},
            ),
        )
    if recipe.builder in {"history_length", "candidate_affinity", "candidate_history"}:
        prior = float(recipe.params.get("prior_strength", 10.0))
        train_bundle = candidate_history_summaries(
            train.arrays["user_id"],
            train.arrays["author_id"],
            train.arrays["duration_ms"],
            train.arrays["date"],
            train.arrays["row_id"],
            targets.labels,
            prior_strength=prior,
        )
        state = fit_candidate_history_state(
            train.arrays["user_id"],
            train.arrays["author_id"],
            train.arrays["duration_ms"],
            targets.labels,
            prior_strength=prior,
        )
        apply_bundle = apply_candidate_history_state(
            apply.arrays["user_id"],
            apply.arrays["author_id"],
            apply.arrays["duration_ms"],
            state,
        )
        if recipe.builder == "history_length":
            names = ("user_history_length",)
        elif recipe.builder == "candidate_affinity":
            names = ("user_author_rate", "user_duration_rate")
        else:
            names = ("user_author_rate", "user_duration_rate", "user_history_length")
        return _selected(train_bundle, names), _selected(apply_bundle, names)
    if recipe.builder == "repeat_exposure":
        train_bundle = repeat_exposure_features(
            train.arrays["user_id"],
            train.arrays["video_id"],
            train.arrays["date"],
            train.arrays["row_id"],
            targets.labels,
        )
        state = fit_repeat_exposure_state(
            train.arrays["user_id"],
            train.arrays["video_id"],
            train.arrays["date"],
            targets.labels,
        )
        apply_bundle = apply_repeat_exposure_state(
            apply.arrays["user_id"],
            apply.arrays["video_id"],
            apply.arrays["date"],
            state,
        )
        return train_bundle, apply_bundle
    if recipe.builder == "recency_history":
        half_life = float(recipe.params.get("half_life_days", 7.0))
        prior = float(recipe.params.get("prior_strength", 10.0))
        train_bundle = recency_history_features(
            train.arrays["user_id"],
            train.arrays["date"],
            train.arrays["row_id"],
            targets.labels,
            half_life_days=half_life,
            prior_strength=prior,
        )
        state = fit_recency_state(
            train.arrays["user_id"],
            train.arrays["date"],
            targets.labels,
            half_life_days=half_life,
            prior_strength=prior,
        )
        apply_bundle = apply_recency_state(
            apply.arrays["user_id"], apply.arrays["date"], state
        )
        return train_bundle, apply_bundle
    if recipe.builder == "candidate_recency":
        half_lives = tuple(
            float(value)
            for value in str(recipe.params.get("half_lives_days", "1,3,7,14")).split(",")
        )
        prior = float(recipe.params.get("prior_strength", 10.0))
        candidate_half_life = float(recipe.params.get("candidate_half_life_days", 7.0))
        train_bundle = candidate_recency_features(
            train.arrays["user_id"],
            train.arrays["video_id"],
            train.arrays["author_id"],
            _temporal_array(train, "time_ms"),
            _temporal_array(train, "source_row_key"),
            targets.labels,
            half_lives_days=half_lives,
            prior_strength=prior,
            candidate_half_life_days=candidate_half_life,
        )
        state = fit_candidate_recency_state(
            train.arrays["user_id"],
            train.arrays["video_id"],
            train.arrays["author_id"],
            _temporal_array(train, "time_ms"),
            _temporal_array(train, "source_row_key"),
            targets.labels,
            half_lives_days=half_lives,
            prior_strength=prior,
            candidate_half_life_days=candidate_half_life,
        )
        apply_bundle = apply_candidate_recency_state(
            apply.arrays["user_id"],
            apply.arrays["video_id"],
            apply.arrays["author_id"],
            _temporal_array(apply, "time_ms"),
            state,
        )
        return train_bundle, apply_bundle
    if recipe.builder == "candidate_recency_buckets":
        half_lives = tuple(
            float(value)
            for value in str(recipe.params.get("half_lives_days", "1,3,7,14")).split(",")
        )
        prior = float(recipe.params.get("prior_strength", 10.0))
        candidate_half_life = float(recipe.params.get("candidate_half_life_days", 7.0))
        train_history = candidate_recency_features(
            train.arrays["user_id"],
            train.arrays["video_id"],
            train.arrays["author_id"],
            _temporal_array(train, "time_ms"),
            _temporal_array(train, "source_row_key"),
            targets.labels,
            half_lives_days=half_lives,
            prior_strength=prior,
            candidate_half_life_days=candidate_half_life,
        )
        history_state = fit_candidate_recency_state(
            train.arrays["user_id"],
            train.arrays["video_id"],
            train.arrays["author_id"],
            _temporal_array(train, "time_ms"),
            _temporal_array(train, "source_row_key"),
            targets.labels,
            half_lives_days=half_lives,
            prior_strength=prior,
            candidate_half_life_days=candidate_half_life,
        )
        apply_history = apply_candidate_recency_state(
            apply.arrays["user_id"],
            apply.arrays["video_id"],
            apply.arrays["author_id"],
            _temporal_array(apply, "time_ms"),
            history_state,
        )
        bucket_state = fit_quantile_buckets(
            train_history,
            CANDIDATE_RECENCY_BUCKET_FIELDS,
            quantile_bins=int(recipe.params.get("quantile_bins", 8)),
            count_cap=int(recipe.params.get("count_cap", 64)),
        )
        return (
            apply_quantile_buckets(train_history, bucket_state),
            apply_quantile_buckets(apply_history, bucket_state),
        )
    if recipe.builder == "multifeedback_history":
        prior = float(recipe.params.get("prior_strength", 20.0))
        feedback = _load_auxiliary_targets(auxiliary_target_path, train.rows)
        train_bundle = multifeedback_history_features(
            train.arrays["user_id"],
            train.arrays["video_id"],
            train.arrays["author_id"],
            _temporal_array(train, "time_ms"),
            _temporal_array(train, "source_row_key"),
            feedback,
            prior_strength=prior,
        )
        state = fit_multifeedback_state(
            train.arrays["user_id"],
            train.arrays["video_id"],
            train.arrays["author_id"],
            _temporal_array(train, "time_ms"),
            _temporal_array(train, "source_row_key"),
            feedback,
            prior_strength=prior,
        )
        apply_bundle = apply_multifeedback_state(
            apply.arrays["user_id"],
            apply.arrays["video_id"],
            apply.arrays["author_id"],
            state,
        )
        return train_bundle, apply_bundle
    if recipe.builder == "categorical_cross":
        if len(recipe.output_features) != 1:
            raise ValueError("categorical_cross recipes must declare exactly one output")
        spec = CategoricalCrossSpec(
            name=recipe.output_features[0],
            left=str(recipe.params["left"]),
            right=str(recipe.params["right"]),
            min_count=int(recipe.params.get("min_count", 2)),
        )
        state = fit_categorical_crosses(train, (spec,))
        return (
            apply_categorical_crosses(train, state),
            apply_categorical_crosses(apply, state),
        )
    raise ValueError(f"unknown feature recipe builder: {recipe.builder}")


def _merge_bundles(bundles: list[FeatureBundle]) -> FeatureBundle:
    arrays: dict[str, np.ndarray] = {}
    provenance: dict[str, dict[str, object]] = {}
    for bundle in bundles:
        overlap = arrays.keys() & bundle.arrays.keys()
        if overlap:
            raise ValueError(f"composite recipe output collision: {sorted(overlap)}")
        arrays.update(bundle.arrays)
        provenance.update(bundle.provenance)
    return FeatureBundle(arrays, provenance)


def _build_definition(
    recipe: RecipeDefinition,
    train_path: Path,
    target_path: Path,
    apply_path: Path,
    auxiliary_target_path: Path | None,
) -> tuple[FeatureBundle, FeatureBundle]:
    if isinstance(recipe, FeatureRecipe):
        return _build(
            recipe,
            train_path,
            target_path,
            apply_path,
            auxiliary_target_path,
        )
    train_bundles: list[FeatureBundle] = []
    apply_bundles: list[FeatureBundle] = []
    for component in recipe.components:
        train_bundle, apply_bundle = _build(
            component,
            train_path,
            target_path,
            apply_path,
            auxiliary_target_path,
        )
        train_bundles.append(_recipe_provenance(train_bundle, component))
        apply_bundles.append(_recipe_provenance(apply_bundle, component))
    return _merge_bundles(train_bundles), _merge_bundles(apply_bundles)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def materialize_feature_recipe(
    recipe: RecipeDefinition,
    train_feature_path: str | Path,
    train_target_path: str | Path,
    apply_feature_path: str | Path,
    cache_dir: str | Path,
    *,
    auxiliary_target_path: str | Path | None = None,
) -> RecipeArtifact:
    """Build or reuse one hash-addressed train/apply feature recipe."""

    train_path = Path(train_feature_path)
    target_path = Path(train_target_path)
    apply_path = Path(apply_feature_path)
    auxiliary_path = Path(auxiliary_target_path) if auxiliary_target_path is not None else None
    identity_value = {
        "schema_version": "1.0",
        "recipe": asdict(recipe),
        "implementation_sha256": {
            path.name: sha256_file(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("base.py"),
                Path(__file__).with_name("history_summaries.py"),
                Path(__file__).with_name("recency.py"),
                Path(__file__).with_name("repeat_exposure.py"),
                Path(__file__).with_name("temporal_aggregates.py"),
                Path(__file__).with_name("temporal_order.py"),
                Path(__file__).with_name("candidate_recency.py"),
                Path(__file__).with_name("multifeedback_history.py"),
                Path(__file__).with_name("categorical_crosses.py"),
                Path(__file__).with_name("quantile_buckets.py"),
            )
        },
        "train_feature_sha256": sha256_file(train_path),
        "train_target_sha256": sha256_file(target_path),
        "apply_feature_sha256": sha256_file(apply_path),
        "auxiliary_target_sha256": sha256_file(auxiliary_path) if auxiliary_path else None,
    }
    identity = sha256_bytes(canonical_json_bytes(identity_value))
    root = Path(cache_dir) / f"{recipe.name}-{identity[:16]}"
    train_output = root / "train_features.npz"
    apply_output = root / "apply_features.npz"
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and train_output.is_file() and apply_output.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_provenance = train_output.with_suffix(train_output.suffix + ".provenance.json")
        apply_provenance = apply_output.with_suffix(apply_output.suffix + ".provenance.json")
        if (
            existing.get("identity_sha256") == identity
            and existing.get("train_output_sha256") == sha256_file(train_output)
            and existing.get("apply_output_sha256") == sha256_file(apply_output)
            and train_provenance.is_file()
            and apply_provenance.is_file()
            and existing.get("train_provenance_sha256") == sha256_file(train_provenance)
            and existing.get("apply_provenance_sha256") == sha256_file(apply_provenance)
        ):
            return RecipeArtifact(root, train_output, apply_output, manifest_path, identity)
    root.mkdir(parents=True, exist_ok=True)
    train_bundle, apply_bundle = _build_definition(
        recipe, train_path, target_path, apply_path, auxiliary_path
    )
    train_bundle = _definition_provenance(train_bundle, recipe)
    apply_bundle = _definition_provenance(apply_bundle, recipe)
    attach_feature_bundles(train_path, [train_bundle], train_output)
    attach_feature_bundles(apply_path, [apply_bundle], apply_output)
    manifest = {
        **identity_value,
        "identity_sha256": identity,
        "train_output_sha256": sha256_file(train_output),
        "apply_output_sha256": sha256_file(apply_output),
        "train_provenance_sha256": sha256_file(
            train_output.with_suffix(train_output.suffix + ".provenance.json")
        ),
        "apply_provenance_sha256": sha256_file(
            apply_output.with_suffix(apply_output.suffix + ".provenance.json")
        ),
    }
    _atomic_json(manifest_path, manifest)
    return RecipeArtifact(root, train_output, apply_output, manifest_path, identity)
