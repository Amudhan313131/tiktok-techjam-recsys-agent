"""Complete-user sampling and same-user positive/negative pair generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairSample:
    positive_indices: np.ndarray
    negative_indices: np.ndarray
    users_with_pairs: int
    pairs_per_user: dict[str, int]


def user_groups(user_ids: np.ndarray) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for index, user in enumerate(user_ids):
        groups.setdefault(str(user), []).append(index)
    return {user: np.asarray(indices, dtype=np.int64) for user, indices in groups.items()}


def sample_complete_users(
    user_ids: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    groups = user_groups(user_ids)
    users = np.asarray(sorted(groups), dtype=str)
    rng = np.random.default_rng(seed)
    count = max(1, int(np.ceil(len(users) * fraction)))
    selected = set(rng.choice(users, size=count, replace=False).tolist())
    indices = np.concatenate([groups[user] for user in users if user in selected])
    return np.sort(indices)


def sample_same_user_pairs(
    user_ids: np.ndarray,
    labels: np.ndarray,
    *,
    negatives_per_positive: int,
    seed: int,
    max_pairs: int | None = None,
) -> PairSample:
    if negatives_per_positive <= 0:
        raise ValueError("negatives_per_positive must be positive")
    if len(user_ids) != len(labels):
        raise ValueError("user/label lengths differ")
    rng = np.random.default_rng(seed)
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    pairs_per_user: dict[str, int] = {}
    for user, indices in user_groups(np.asarray(user_ids)).items():
        user_labels = np.asarray(labels)[indices]
        pos = indices[user_labels == 1]
        neg = indices[user_labels == 0]
        if not len(pos) or not len(neg):
            continue
        repeated_pos = np.repeat(pos, negatives_per_positive)
        sampled_neg = rng.choice(neg, size=len(repeated_pos), replace=True)
        positives.append(repeated_pos)
        negatives.append(sampled_neg)
        pairs_per_user[user] = len(repeated_pos)
    if not positives:
        return PairSample(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            0,
            {},
        )
    pos_all = np.concatenate(positives)
    neg_all = np.concatenate(negatives)
    order = rng.permutation(len(pos_all))
    if max_pairs is not None:
        order = order[:max_pairs]
    return PairSample(
        positive_indices=pos_all[order],
        negative_indices=neg_all[order],
        users_with_pairs=len(pairs_per_user),
        pairs_per_user=pairs_per_user,
    )
