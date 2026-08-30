"""Allowlisted production adapter for agent-authored pairwise-FM changes.

The protected implementation and artifact contract live in ``rex.models.rank_fm``.
Autonomous patches may evolve this thin adapter (and its bound experiment config)
without gaining write access to the trusted model, evaluator, or control plane.
"""

from __future__ import annotations

from rex.models.rank_fm import RankFMPlugin


class ExperimentalPairRankFMPlugin(RankFMPlugin):
    """Current pairwise implementation; safe extension point for one-change patches."""

