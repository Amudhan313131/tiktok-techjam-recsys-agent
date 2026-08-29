"""Rolling temporal fold definitions and point-in-time ordering helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShadowFold:
    name: str
    train_min: int
    train_max: int
    valid_min: int
    valid_max: int

    def masks(self, dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        train = (dates >= self.train_min) & (dates <= self.train_max)
        valid = (dates >= self.valid_min) & (dates <= self.valid_max)
        if np.any(train & valid):
            raise ValueError(f"shadow fold {self.name} overlaps")
        return train, valid


DEFAULT_SHADOW_FOLDS = (
    ShadowFold("A", 20220408, 20220414, 20220415, 20220416),
    ShadowFold("B", 20220408, 20220416, 20220417, 20220418),
    ShadowFold("C", 20220408, 20220418, 20220419, 20220421),
)


def validate_shadow_folds(dates: np.ndarray, folds=DEFAULT_SHADOW_FOLDS) -> None:
    for fold in folds:
        train, valid = fold.masks(dates)
        if not train.any() or not valid.any():
            raise ValueError(f"shadow fold {fold.name} has an empty partition")


def date_batches(dates: np.ndarray, row_ids: np.ndarray) -> list[np.ndarray]:
    """Return stable date batches so same-day labels never influence same-day features."""
    order = np.lexsort((row_ids, dates))
    ordered_dates = dates[order]
    if not len(order):
        return []
    boundaries = np.flatnonzero(np.diff(ordered_dates)) + 1
    return [part for part in np.split(order, boundaries) if len(part)]
