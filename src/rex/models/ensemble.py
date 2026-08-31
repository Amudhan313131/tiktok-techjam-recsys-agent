"""Leakage-safe score normalization, blending, and train-support shrinkage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rex.data.groups import user_groups
from rex.evaluation.diagnostics import aggregate_user_metrics, per_user_metrics


def _one_dimensional(name: str, values: np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def _canonical_key(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return f"bool:{int(value)}"
    if isinstance(value, int):
        return f"int:{value}"
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("alignment/support keys must be finite")
        return f"float:{value.hex()}"
    if isinstance(value, str):
        return f"str:{value}"
    raise TypeError(f"unsupported alignment/support key type: {type(value).__name__}")


def _array_identity(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    array = _one_dimensional("identity array", values)
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(len(array).to_bytes(8, "big"))
    if array.dtype.kind in {"b", "i", "u", "f", "U", "S"}:
        digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()
    for value in array:
        encoded = _canonical_key(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


@dataclass(frozen=True)
class PredictionVector:
    """Predictions carrying the row and user identities they were made for."""

    name: str
    row_ids: np.ndarray
    user_ids: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("prediction name must not be empty")
        row_ids = _one_dimensional("row_ids", self.row_ids).copy()
        user_ids = _one_dimensional("user_ids", self.user_ids).copy()
        scores = _one_dimensional("scores", self.scores).astype(np.float64, copy=True)
        if not (len(row_ids) == len(user_ids) == len(scores)):
            raise ValueError("prediction row, user, and score lengths differ")
        if not len(row_ids):
            raise ValueError("prediction vector must not be empty")
        row_keys = [_canonical_key(value) for value in row_ids]
        if len(set(row_keys)) != len(row_keys):
            raise ValueError("prediction row IDs must be unique")
        if not np.isfinite(scores).all():
            raise ValueError("prediction scores contain NaN or Inf")
        row_ids.flags.writeable = False
        user_ids.flags.writeable = False
        scores.flags.writeable = False
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "user_ids", user_ids)
        object.__setattr__(self, "scores", scores)

    @property
    def identity_sha256(self) -> str:
        value = {
            "name": self.name,
            "row_ids_sha256": _array_identity(self.row_ids),
            "user_ids_sha256": _array_identity(self.user_ids),
            "scores_sha256": _array_identity(self.scores),
        }
        return _json_sha256(value)


def require_exact_prediction_alignment(
    left: PredictionVector, right: PredictionVector
) -> None:
    """Fail closed unless two score vectors refer to identical rows and users."""

    if len(left.scores) != len(right.scores):
        raise ValueError("prediction lengths differ")
    if not np.array_equal(left.row_ids, right.row_ids):
        raise ValueError("prediction row IDs are not identically aligned")
    if not np.array_equal(left.user_ids, right.user_ids):
        raise ValueError("prediction user IDs are not identically aligned")


@dataclass(frozen=True)
class ShadowBlendFold:
    """One predeclared temporal shadow fold used for blend selection."""

    name: str
    row_ids: np.ndarray
    user_ids: np.ndarray
    labels: np.ndarray
    left: PredictionVector
    right: PredictionVector
    split: str = "shadow"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("shadow fold name must not be empty")
        if self.split != "shadow":
            raise ValueError("blend selection folds must be shadow-only")
        row_ids = _one_dimensional("shadow row_ids", self.row_ids).copy()
        user_ids = _one_dimensional("shadow user_ids", self.user_ids).copy()
        labels = _one_dimensional("shadow labels", self.labels).astype(np.float64, copy=True)
        if not (len(row_ids) == len(user_ids) == len(labels)):
            raise ValueError("shadow row, user, and label lengths differ")
        if not np.isfinite(labels).all() or not np.isin(labels, (0.0, 1.0)).all():
            raise ValueError("shadow labels must be finite binary values")
        require_exact_prediction_alignment(self.left, self.right)
        if not np.array_equal(row_ids, self.left.row_ids):
            raise ValueError("shadow labels and predictions have different row alignment")
        if not np.array_equal(user_ids, self.left.user_ids):
            raise ValueError("shadow labels and predictions have different user alignment")
        row_ids.flags.writeable = False
        user_ids.flags.writeable = False
        labels.flags.writeable = False
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "user_ids", user_ids)
        object.__setattr__(self, "labels", labels)

    @property
    def identity_sha256(self) -> str:
        value = {
            "fold": self.name,
            "split": self.split,
            "row_ids_sha256": _array_identity(self.row_ids),
            "user_ids_sha256": _array_identity(self.user_ids),
            "labels_sha256": _array_identity(self.labels),
            "left_sha256": self.left.identity_sha256,
            "right_sha256": self.right.identity_sha256,
        }
        return _json_sha256(value)


@dataclass(frozen=True)
class BlendSelection:
    """Persistable result of deterministic, shadow-only two-model selection."""

    branch_names: tuple[str, str]
    weights: tuple[float, float]
    stronger_branch: str
    stronger_single_primary: float
    selected_mean_primary: float
    penalized_objective: float
    fold_primaries: Mapping[str, float]
    fold_sha256: Mapping[str, str]
    normalization: str
    grid_size: int
    regularization_strength: float
    selection_split: str = "shadow_only"
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported blend selection schema")
        if len(self.branch_names) != 2 or self.branch_names[0] == self.branch_names[1]:
            raise ValueError("blend selection requires two distinct branch names")
        weights = tuple(float(value) for value in self.weights)
        if len(weights) != 2 or any(value < 0 or value > 1 for value in weights):
            raise ValueError("blend weights must be in [0,1]")
        if abs(sum(weights) - 1.0) > 1e-12:
            raise ValueError("blend weights must sum to one")
        if self.stronger_branch not in self.branch_names:
            raise ValueError("stronger branch is not one of the selected branches")
        if self.selection_split != "shadow_only":
            raise ValueError("blend selection may only be fit on temporal shadow folds")
        if self.normalization not in {"percentile", "standardize"}:
            raise ValueError("unsupported blend normalization")
        if not 2 <= self.grid_size <= 101:
            raise ValueError("blend grid_size must be between 2 and 101")
        if not 0.0 <= self.regularization_strength <= 1.0:
            raise ValueError("blend regularization_strength must be in [0,1]")
        for name, value in {
            "stronger_single_primary": self.stronger_single_primary,
            "selected_mean_primary": self.selected_mean_primary,
            "penalized_objective": self.penalized_objective,
        }.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not self.fold_primaries or set(self.fold_primaries) != set(self.fold_sha256):
            raise ValueError("blend fold metrics and identities must be nonempty and aligned")
        if any(not 0.0 <= float(value) <= 1.0 for value in self.fold_primaries.values()):
            raise ValueError("blend fold primary scores must be in [0,1]")
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in self.fold_sha256.values()
        ):
            raise ValueError("blend fold identities must be SHA-256 digests")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "fold_primaries", dict(sorted(self.fold_primaries.items())))
        object.__setattr__(self, "fold_sha256", dict(sorted(self.fold_sha256.items())))

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "selection_split": self.selection_split,
            "branch_names": list(self.branch_names),
            "weights": list(self.weights),
            "stronger_branch": self.stronger_branch,
            "stronger_single_primary": self.stronger_single_primary,
            "selected_mean_primary": self.selected_mean_primary,
            "penalized_objective": self.penalized_objective,
            "fold_primaries": dict(self.fold_primaries),
            "fold_sha256": dict(self.fold_sha256),
            "normalization": self.normalization,
            "grid_size": self.grid_size,
            "regularization_strength": self.regularization_strength,
        }
        return {**payload, "state_sha256": _json_sha256(payload)}

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "BlendSelection":
        payload = dict(value)
        state_sha256 = str(payload.pop("state_sha256", ""))
        if not state_sha256 or state_sha256 != _json_sha256(payload):
            raise ValueError("blend selection state hash does not match")
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported blend selection schema")
        return cls(
            branch_names=tuple(str(item) for item in payload["branch_names"]),  # type: ignore[arg-type]
            weights=tuple(float(item) for item in payload["weights"]),  # type: ignore[arg-type]
            stronger_branch=str(payload["stronger_branch"]),
            stronger_single_primary=float(payload["stronger_single_primary"]),
            selected_mean_primary=float(payload["selected_mean_primary"]),
            penalized_objective=float(payload["penalized_objective"]),
            fold_primaries={
                str(key): float(item)
                for key, item in dict(payload["fold_primaries"]).items()  # type: ignore[arg-type]
            },
            fold_sha256={
                str(key): str(item)
                for key, item in dict(payload["fold_sha256"]).items()  # type: ignore[arg-type]
            },
            normalization=str(payload["normalization"]),
            grid_size=int(payload["grid_size"]),
            regularization_strength=float(payload["regularization_strength"]),
            selection_split=str(payload["selection_split"]),
            schema_version=str(payload["schema_version"]),
        )


def per_user_percentile_ranks(user_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    user_ids = _one_dimensional("user_ids", user_ids)
    scores = _one_dimensional("scores", scores).astype(np.float64, copy=False)
    if len(user_ids) != len(scores):
        raise ValueError("user and score lengths differ")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or Inf")
    result = np.empty(len(scores), dtype=np.float64)
    for indices in user_groups(user_ids).values():
        values = scores[indices]
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(indices), dtype=np.float64)
        ranks[order] = np.arange(len(indices), dtype=np.float64)
        result[indices] = ranks / max(1, len(indices) - 1)
    return result


def per_user_standardize(user_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    user_ids = _one_dimensional("user_ids", user_ids)
    scores = _one_dimensional("scores", scores).astype(np.float64, copy=False)
    if len(user_ids) != len(scores):
        raise ValueError("user and score lengths differ")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or Inf")
    result = np.empty(len(scores), dtype=np.float64)
    for indices in user_groups(user_ids).values():
        values = scores[indices]
        std = values.std()
        result[indices] = (values - values.mean()) / std if std > 0 else 0.0
    return result


def blend_scores(
    user_ids: np.ndarray,
    predictions: list[np.ndarray],
    weights: np.ndarray,
    *,
    normalization: str = "percentile",
) -> np.ndarray:
    if not predictions:
        raise ValueError("at least one prediction vector is required")
    user_ids = _one_dimensional("user_ids", user_ids)
    weights = _one_dimensional("weights", weights).astype(np.float64, copy=True)
    if len(weights) != len(predictions) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("ensemble weights must be non-negative and match predictions")
    if not np.isfinite(weights).all():
        raise ValueError("ensemble weights contain NaN or Inf")
    values = [_one_dimensional("predictions", item) for item in predictions]
    if any(len(item) != len(user_ids) for item in values):
        raise ValueError("user and prediction lengths differ")
    if normalization == "percentile":
        normalize = per_user_percentile_ranks
    elif normalization == "standardize":
        normalize = per_user_standardize
    else:
        raise ValueError("normalization must be 'percentile' or 'standardize'")
    weights /= weights.sum()
    normalized = np.column_stack([normalize(user_ids, item) for item in values])
    return normalized @ weights


def _primary(user_ids: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> float:
    return float(aggregate_user_metrics(per_user_metrics(user_ids, labels, scores))["primary"])


def select_shadow_blend(
    folds: Sequence[ShadowBlendFold],
    *,
    normalization: str = "percentile",
    grid_size: int = 21,
    regularization_strength: float = 0.00025,
) -> BlendSelection:
    """Select a two-model blend on temporal shadow folds, never official validation.

    The regularized objective is the equal-fold mean primary score minus a
    quadratic penalty for moving away from the stronger single branch.  This
    makes a blend earn its additional complexity rather than winning on a tiny,
    unstable grid fluctuation.
    """

    if not folds:
        raise ValueError("at least one temporal shadow fold is required")
    if not 2 <= grid_size <= 101:
        raise ValueError("grid_size must be between 2 and 101")
    if not 0.0 <= regularization_strength <= 1.0:
        raise ValueError("regularization_strength must be in [0,1]")
    fold_names = [fold.name for fold in folds]
    if len(set(fold_names)) != len(fold_names):
        raise ValueError("temporal shadow fold names must be unique")
    branch_names = (folds[0].left.name, folds[0].right.name)
    if branch_names[0] == branch_names[1]:
        raise ValueError("shadow blend branches must have distinct names")
    for fold in folds:
        if (fold.left.name, fold.right.name) != branch_names:
            raise ValueError("shadow folds must use the same ordered branch names")

    single_scores: list[list[float]] = [[], []]
    for fold in folds:
        single_scores[0].append(_primary(fold.user_ids, fold.labels, fold.left.scores))
        single_scores[1].append(_primary(fold.user_ids, fold.labels, fold.right.scores))
    mean_singles = [float(np.mean(values)) for values in single_scores]
    stronger_index = 0 if mean_singles[0] >= mean_singles[1] else 1
    anchor_left_weight = 1.0 if stronger_index == 0 else 0.0

    best_key: tuple[float, float, float, float] | None = None
    best_weight = anchor_left_weight
    best_fold_primaries: dict[str, float] = {}
    best_mean = mean_singles[stronger_index]
    best_objective = best_mean
    for left_weight in np.linspace(0.0, 1.0, grid_size):
        fold_primaries: dict[str, float] = {}
        for fold in folds:
            scores = blend_scores(
                fold.user_ids,
                [fold.left.scores, fold.right.scores],
                np.asarray([left_weight, 1.0 - left_weight], dtype=np.float64),
                normalization=normalization,
            )
            fold_primaries[fold.name] = _primary(fold.user_ids, fold.labels, scores)
        mean_primary = float(np.mean(list(fold_primaries.values())))
        distance = abs(float(left_weight) - anchor_left_weight)
        objective = mean_primary - regularization_strength * distance**2
        # Rounded score components remove platform-level floating noise. The
        # remaining tie-breaks prefer the stronger branch, then lower left weight.
        key = (
            round(objective, 15),
            round(mean_primary, 15),
            -round(distance, 15),
            -round(float(left_weight), 15),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_weight = float(left_weight)
            best_fold_primaries = fold_primaries
            best_mean = mean_primary
            best_objective = objective

    return BlendSelection(
        branch_names=branch_names,
        weights=(best_weight, 1.0 - best_weight),
        stronger_branch=branch_names[stronger_index],
        stronger_single_primary=mean_singles[stronger_index],
        selected_mean_primary=best_mean,
        penalized_objective=best_objective,
        fold_primaries=best_fold_primaries,
        fold_sha256={fold.name: fold.identity_sha256 for fold in folds},
        normalization=normalization,
        grid_size=grid_size,
        regularization_strength=regularization_strength,
    )


def save_blend_selection(path: Path, state: BlendSelection) -> Path:
    return _atomic_json(path, state.to_json())


def load_blend_selection(path: Path) -> BlendSelection:
    return BlendSelection.from_json(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class SupportShrinkageSpec:
    """Predeclared transform parameters; these must not be tuned on validation."""

    key_name: str
    prior_strength: float
    minimum_candidate_weight: float = 0.0
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported support shrinkage spec schema")
        if not self.key_name.strip():
            raise ValueError("support key_name must not be empty")
        if not np.isfinite(self.prior_strength) or self.prior_strength <= 0:
            raise ValueError("support prior_strength must be positive and finite")
        if not 0.0 <= self.minimum_candidate_weight <= 1.0:
            raise ValueError("minimum_candidate_weight must be in [0,1]")


@dataclass(frozen=True)
class SupportShrinkageState:
    """Counts fitted from training identities only; contains no outcomes."""

    spec: SupportShrinkageSpec
    counts: Mapping[str, int]
    train_rows: int
    train_keys_sha256: str
    fitted_on: str = "train_support_only"
    uses_labels: bool = False
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported support shrinkage state schema")
        counts = dict(sorted((str(key), int(value)) for key, value in self.counts.items()))
        if any(value <= 0 for value in counts.values()):
            raise ValueError("support counts must be positive")
        if self.train_rows != sum(counts.values()) or self.train_rows <= 0:
            raise ValueError("train_rows must equal the sum of support counts")
        if self.fitted_on != "train_support_only" or self.uses_labels:
            raise ValueError("support shrinkage state must be label-free and train-only")
        if len(self.train_keys_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.train_keys_sha256
        ):
            raise ValueError("train support identity must be a SHA-256 digest")
        object.__setattr__(self, "counts", counts)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "spec": {
                "schema_version": self.spec.schema_version,
                "key_name": self.spec.key_name,
                "prior_strength": self.spec.prior_strength,
                "minimum_candidate_weight": self.spec.minimum_candidate_weight,
            },
            "counts": dict(self.counts),
            "train_rows": self.train_rows,
            "train_keys_sha256": self.train_keys_sha256,
            "fitted_on": self.fitted_on,
            "uses_labels": self.uses_labels,
        }
        return {**payload, "state_sha256": _json_sha256(payload)}

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "SupportShrinkageState":
        payload = dict(value)
        state_sha256 = str(payload.pop("state_sha256", ""))
        if not state_sha256 or state_sha256 != _json_sha256(payload):
            raise ValueError("support shrinkage state hash does not match")
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported support shrinkage state schema")
        raw_spec = dict(payload["spec"])  # type: ignore[arg-type]
        if raw_spec.get("schema_version") != "1.0":
            raise ValueError("unsupported support shrinkage spec schema")
        return cls(
            spec=SupportShrinkageSpec(
                key_name=str(raw_spec["key_name"]),
                prior_strength=float(raw_spec["prior_strength"]),
                minimum_candidate_weight=float(raw_spec["minimum_candidate_weight"]),
                schema_version=str(raw_spec["schema_version"]),
            ),
            counts={
                str(key): int(item)
                for key, item in dict(payload["counts"]).items()  # type: ignore[arg-type]
            },
            train_rows=int(payload["train_rows"]),
            train_keys_sha256=str(payload["train_keys_sha256"]),
            fitted_on=str(payload["fitted_on"]),
            uses_labels=bool(payload["uses_labels"]),
            schema_version=str(payload["schema_version"]),
        )


def fit_train_support_shrinkage(
    train_keys: np.ndarray, spec: SupportShrinkageSpec
) -> SupportShrinkageState:
    """Fit support counts from training keys; outcomes are absent by construction."""

    values = _one_dimensional("train support keys", train_keys)
    if not len(values):
        raise ValueError("train support keys must not be empty")
    counts: dict[str, int] = {}
    for value in values:
        key = _canonical_key(value)
        counts[key] = counts.get(key, 0) + 1
    return SupportShrinkageState(
        spec=spec,
        counts=counts,
        train_rows=len(values),
        train_keys_sha256=_array_identity(values),
    )


def support_candidate_weights(
    state: SupportShrinkageState, apply_keys: np.ndarray
) -> np.ndarray:
    """Return fixed candidate weights using only the persisted train supports."""

    keys = _one_dimensional("apply support keys", apply_keys)
    support = np.asarray(
        [state.counts.get(_canonical_key(value), 0) for value in keys], dtype=np.float64
    )
    raw = support / (support + state.spec.prior_strength)
    floor = state.spec.minimum_candidate_weight
    return floor + (1.0 - floor) * raw


def apply_train_support_shrinkage(
    state: SupportShrinkageState,
    apply_keys: np.ndarray,
    candidate: PredictionVector,
    anchor: PredictionVector,
) -> np.ndarray:
    """Shrink candidate predictions toward an anchor without accepting labels."""

    require_exact_prediction_alignment(candidate, anchor)
    keys = _one_dimensional("apply support keys", apply_keys)
    if len(keys) != len(candidate.scores):
        raise ValueError("support keys and aligned predictions have different lengths")
    weights = support_candidate_weights(state, keys)
    return weights * candidate.scores + (1.0 - weights) * anchor.scores


def save_support_shrinkage(path: Path, state: SupportShrinkageState) -> Path:
    return _atomic_json(path, state.to_json())


def load_support_shrinkage(path: Path) -> SupportShrinkageState:
    return SupportShrinkageState.from_json(json.loads(path.read_text(encoding="utf-8")))


def prediction_vectors(
    names: Iterable[str], row_ids: np.ndarray, user_ids: np.ndarray, scores: Iterable[np.ndarray]
) -> tuple[PredictionVector, ...]:
    """Convenience constructor that preserves alignment metadata for every branch."""

    return tuple(
        PredictionVector(name=name, row_ids=row_ids, user_ids=user_ids, scores=values)
        for name, values in zip(names, scores, strict=True)
    )
