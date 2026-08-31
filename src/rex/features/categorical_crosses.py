"""Train-fitted categorical crosses with explicit rare and unknown backoff.

This module deliberately has no recipe-registry dependency.  A caller fits the
state on one training view, serializes that state with the model/recipe
artifact, and applies it unchanged to later views.  No targets are accepted or
read, so a cross cannot accidentally encode evaluation outcomes.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from rex.data.views import FeatureView
from rex.features.base import FeatureBundle


RARE_INDEX = 0
UNKNOWN_INDEX = 1
FIRST_VALUE_INDEX = 2
STATE_SCHEMA_VERSION = "1.0"


def _canonical_value(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "__MISSING__"
    return str(value)


def _cross_key(left: object, right: object) -> str:
    return json.dumps(
        [_canonical_value(left), _canonical_value(right)],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _column(view: FeatureView, name: str) -> np.ndarray:
    key = name if name in view.arrays else f"fx__{name}"
    if key not in view.arrays:
        raise ValueError(f"categorical cross requires feature column {key}")
    values = np.asarray(view.arrays[key])
    if values.ndim != 1 or len(values) != view.rows:
        raise ValueError(f"categorical cross column {key} is misaligned")
    if values.dtype.kind in "biufc" and not np.isfinite(values).all():
        raise ValueError(f"categorical cross column {key} contains NaN or Inf")
    return values


@dataclass(frozen=True)
class CategoricalCrossSpec:
    """One declared two-column cross and its train support threshold."""

    name: str
    left: str
    right: str
    min_count: int = 2

    def __post_init__(self) -> None:
        for attribute in ("name", "left", "right"):
            if not str(getattr(self, attribute)).strip():
                raise ValueError(f"categorical cross {attribute} cannot be empty")
        if self.left == self.right:
            raise ValueError("categorical cross inputs must be different columns")
        if self.min_count < 1:
            raise ValueError("categorical cross min_count must be positive")


@dataclass(frozen=True)
class FittedCategoricalCross:
    spec: CategoricalCrossSpec
    vocabulary: dict[str, int]
    rare_keys: tuple[str, ...]
    training_rows: int

    def __post_init__(self) -> None:
        expected = list(range(FIRST_VALUE_INDEX, FIRST_VALUE_INDEX + len(self.vocabulary)))
        if sorted(self.vocabulary.values()) != expected:
            raise ValueError("categorical cross vocabulary indices are not canonical")
        if set(self.vocabulary) & set(self.rare_keys):
            raise ValueError("categorical cross key cannot be both frequent and rare")

    def to_json(self) -> dict[str, Any]:
        return {
            "spec": asdict(self.spec),
            "vocabulary": self.vocabulary,
            "rare_keys": list(self.rare_keys),
            "training_rows": self.training_rows,
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "FittedCategoricalCross":
        return cls(
            spec=CategoricalCrossSpec(**value["spec"]),
            vocabulary={str(key): int(index) for key, index in value["vocabulary"].items()},
            rare_keys=tuple(str(item) for item in value["rare_keys"]),
            training_rows=int(value["training_rows"]),
        )


@dataclass(frozen=True)
class CategoricalCrossState:
    crosses: tuple[FittedCategoricalCross, ...]

    def __post_init__(self) -> None:
        names = [item.spec.name for item in self.crosses]
        if not names:
            raise ValueError("at least one categorical cross is required")
        if len(set(names)) != len(names):
            raise ValueError("categorical cross names must be unique")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "rare_index": RARE_INDEX,
            "unknown_index": UNKNOWN_INDEX,
            "crosses": [item.to_json() for item in self.crosses],
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "CategoricalCrossState":
        if value.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ValueError("unsupported categorical-cross state schema")
        if value.get("rare_index") != RARE_INDEX or value.get("unknown_index") != UNKNOWN_INDEX:
            raise ValueError("categorical-cross reserved indices drifted")
        return cls(
            tuple(FittedCategoricalCross.from_json(item) for item in value["crosses"])
        )


def fit_categorical_crosses(
    train: FeatureView,
    specs: Iterable[CategoricalCrossSpec],
) -> CategoricalCrossState:
    """Fit deterministic vocabularies solely from a training feature view."""

    declared = tuple(specs)
    if not declared:
        raise ValueError("at least one categorical cross specification is required")
    if len({spec.name for spec in declared}) != len(declared):
        raise ValueError("categorical cross names must be unique")
    fitted: list[FittedCategoricalCross] = []
    for spec in declared:
        left = _column(train, spec.left)
        right = _column(train, spec.right)
        counts = Counter(_cross_key(a, b) for a, b in zip(left, right, strict=True))
        frequent = sorted(key for key, count in counts.items() if count >= spec.min_count)
        rare = tuple(sorted(key for key, count in counts.items() if count < spec.min_count))
        vocabulary = {
            key: FIRST_VALUE_INDEX + index for index, key in enumerate(frequent)
        }
        fitted.append(FittedCategoricalCross(spec, vocabulary, rare, train.rows))
    return CategoricalCrossState(tuple(fitted))


def apply_categorical_crosses(
    view: FeatureView,
    state: CategoricalCrossState,
) -> FeatureBundle:
    """Apply frozen cross vocabularies, distinguishing train-rare from unseen."""

    arrays: dict[str, np.ndarray] = {}
    provenance: dict[str, dict[str, object]] = {}
    for fitted in state.crosses:
        spec = fitted.spec
        left = _column(view, spec.left)
        right = _column(view, spec.right)
        rare = set(fitted.rare_keys)
        encoded = np.fromiter(
            (
                fitted.vocabulary.get(
                    key,
                    RARE_INDEX if key in rare else UNKNOWN_INDEX,
                )
                for key in (
                    _cross_key(a, b) for a, b in zip(left, right, strict=True)
                )
            ),
            dtype=np.int32,
            count=view.rows,
        )
        arrays[spec.name] = encoded
        provenance[spec.name] = {
            "cutoff": "train-fitted categorical vocabulary; no targets",
            "left": spec.left,
            "right": spec.right,
            "min_count": spec.min_count,
            "training_rows": fitted.training_rows,
            "vocabulary_size": len(fitted.vocabulary),
            "rare_key_count": len(fitted.rare_keys),
            "rare_index": RARE_INDEX,
            "unknown_index": UNKNOWN_INDEX,
        }
    bundle = FeatureBundle(arrays, provenance)
    bundle.validate(view.rows)
    return bundle


def fit_transform_categorical_crosses(
    train: FeatureView,
    specs: Iterable[CategoricalCrossSpec],
) -> tuple[CategoricalCrossState, FeatureBundle]:
    state = fit_categorical_crosses(train, specs)
    return state, apply_categorical_crosses(train, state)
