"""Allowlisted production adapter for agent-authored tree/history changes."""

from __future__ import annotations

from rex.models.tree_ranker import TreeRankerPlugin


class ExperimentalTreeHistoryPlugin(TreeRankerPlugin):
    """Current LambdaRank implementation; safe extension point for one-change patches."""

