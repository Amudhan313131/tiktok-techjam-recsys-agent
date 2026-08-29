"""Evidence-gated experiment queue from the scientific implementation plan."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentCard:
    card_id: str
    mechanism: str
    prerequisite: str | None = None


DEFAULT_QUEUE = (
    ExperimentCard("E00", "reproduce five-seed official FM and set the incumbent"),
    ExperimentCard("E01", "replace pointwise BCE with same-user fixed-K PairLogit"),
    ExperimentCard("E02", "tree ranker with point-in-time item/author rates and no-stat control"),
    ExperimentCard("E03", "simple candidate/history affinity summaries"),
    ExperimentCard("E04", "delta-nDCG@5 weighting on best pairwise model", "E01_supported"),
    ExperimentCard("E05", "small BCE stabilizer for pairwise variance", "E01_supported"),
    ExperimentCard("E06", "repeat-exposure count, prior outcome and elapsed time", "tree_supported"),
    ExperimentCard("E07", "user-author, tag and duration affinity", "tree_supported"),
    ExperimentCard("E08", "recency-decayed point-in-time summaries", "history_supported"),
    ExperimentCard("E09", "candidate/history dot-product gate for neural history", "history_supported"),
    ExperimentCard("E10", "per-user blend of pairwise FM and tree/history branch", "two_branches_supported"),
    ExperimentCard("E11", "small DIN-like attention", "E09_supported"),
    ExperimentCard("E12", "tab-conditioned click auxiliary", "E11_supported"),
    ExperimentCard("E13", "threshold-aware watch-time auxiliary", "E12_decided"),
    ExperimentCard("E14", "three-seed confirmation of best single and blend", "finalist_ready"),
)


class SearchPolicy:
    def __init__(self, cards: tuple[ExperimentCard, ...] = DEFAULT_QUEUE):
        self.cards = cards

    def next_card(self, attempted: set[str], evidence_flags: set[str]) -> ExperimentCard | None:
        for card in self.cards:
            if card.card_id in attempted:
                continue
            if card.prerequisite is None or card.prerequisite in evidence_flags:
                return card
        return None
