"""Metric-aligned pairwise and hybrid loss utilities."""

from __future__ import annotations

import numpy as np


def pair_logistic_loss(score_difference: np.ndarray, weights: np.ndarray | None = None) -> float:
    differences = np.asarray(score_difference, dtype=np.float64)
    losses = np.logaddexp(0.0, -differences)
    if weights is None:
        return float(np.mean(losses)) if len(losses) else 0.0
    normalized = np.asarray(weights, dtype=np.float64)
    if normalized.shape != losses.shape or np.any(normalized < 0):
        raise ValueError("pair weights must be non-negative and match differences")
    denominator = normalized.sum()
    return float(np.sum(losses * normalized) / denominator) if denominator else 0.0


def pair_logistic_gradient(score_difference: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    differences = np.asarray(score_difference, dtype=np.float64)
    gradient = -1.0 / (1.0 + np.exp(np.clip(differences, -30, 30)))
    if weights is not None:
        normalized = np.asarray(weights, dtype=np.float64)
        denominator = normalized.sum()
        if denominator:
            gradient *= normalized / denominator * len(gradient)
        else:
            gradient.fill(0.0)
    return gradient.astype(np.float32)


def delta_ndcg_weights(
    positive_ranks: np.ndarray,
    negative_ranks: np.ndarray,
    positives_in_group: np.ndarray,
    *,
    cutoff: int = 5,
) -> np.ndarray:
    """Absolute nDCG change for swapping a positive and negative rank."""
    positive_ranks = np.asarray(positive_ranks, dtype=np.int64)
    negative_ranks = np.asarray(negative_ranks, dtype=np.int64)
    positives_in_group = np.asarray(positives_in_group, dtype=np.int64)
    ideal_lengths = np.minimum(positives_in_group, cutoff)
    discounts = 1.0 / np.log2(np.arange(cutoff, dtype=np.float64) + 2.0)
    prefix = np.concatenate(([0.0], np.cumsum(discounts)))
    idcg = prefix[ideal_lengths]
    positive_discount = np.where(
        positive_ranks < cutoff, 1.0 / np.log2(positive_ranks + 2.0), 0.0
    )
    negative_discount = np.where(
        negative_ranks < cutoff, 1.0 / np.log2(negative_ranks + 2.0), 0.0
    )
    return np.divide(
        np.abs(positive_discount - negative_discount),
        idcg,
        out=np.zeros_like(idcg),
        where=idcg > 0,
    ).astype(np.float32)


def binary_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def normalized_hybrid_loss(
    pair_loss: float,
    bce_loss: float,
    *,
    pair_reference: float,
    bce_reference: float,
    bce_weight: float,
) -> float:
    if pair_reference <= 0 or bce_reference <= 0 or not 0 <= bce_weight <= 1:
        raise ValueError("invalid hybrid loss normalization")
    return (1 - bce_weight) * pair_loss / pair_reference + bce_weight * bce_loss / bce_reference
