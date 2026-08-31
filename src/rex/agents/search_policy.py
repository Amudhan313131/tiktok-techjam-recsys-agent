"""Evidence-gated experiment queue from the scientific implementation plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


METHOD_CARD_VERSION = "1.0"

METHOD_SOURCES: dict[str, dict[str, str]] = {
    "kuairand": {
        "title": "KuaiRand: An Unbiased Sequential Recommendation Dataset",
        "doi": "10.1145/3511808.3557624",
        "url": "https://doi.org/10.1145/3511808.3557624",
    },
    "bagging": {
        "title": "Bagging Predictors",
        "doi": "10.1007/BF00058655",
        "url": "https://doi.org/10.1007/BF00058655",
    },
    "ranknet": {
        "title": "Learning to Rank using Gradient Descent",
        "url": "https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/",
    },
    "lambdamart": {
        "title": "From RankNet to LambdaRank to LambdaMART: An Overview",
        "url": "https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/",
    },
    "lightgbm": {
        "title": "LightGBM: A Highly Efficient Gradient Boosting Decision Tree",
        "url": "https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html",
    },
    "factorization_machines": {
        "title": "Factorization Machines",
        "doi": "10.1109/ICDM.2010.127",
        "url": "https://doi.org/10.1109/ICDM.2010.127",
    },
    "ordered_target_statistics": {
        "title": "CatBoost: unbiased boosting with categorical features",
        "url": "https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html",
    },
    "stacked_generalization": {
        "title": "Stacked Generalization",
        "doi": "10.1016/S0893-6080(05)80023-1",
        "url": "https://doi.org/10.1016/S0893-6080(05)80023-1",
    },
}

METHOD_CARD_REFERENCES: dict[str, dict[str, Any]] = {
    "E15": {
        "primary_change": (
            "average five context-aware FM initializations instead of relying on one "
            "stochastic model"
        ),
        "falsifier": (
            "the five-member mean ensemble fails to beat its matched one-member control"
        ),
        "implementation_hint": (
            "In the bound YAML only, change aggregation from median to mean. Preserve "
            "ensemble_members=5, epochs=7, the plugin, and every other setting. The plugin "
            "already consumes sanitized fx__hour and fx__is_rand fields; do not reimplement it."
        ),
        "citation_ids": ["kuairand", "bagging", "factorization_machines"],
    },
    "E01": {
        "primary_change": "replace pointwise BCE with same-user fixed-K PairLogit",
        "falsifier": "cheap primary delta is below 0.001",
        "citation_ids": ["ranknet", "factorization_machines"],
    },
    "E02": {
        "primary_change": "use grouped LightGBM LambdaRank with a no-stat control",
        "falsifier": "the target-stat branch fails to beat its no-stat control",
        "citation_ids": ["lambdamart", "lightgbm", "ordered_target_statistics"],
    },
    "E03": {
        "primary_change": "add simple candidate-history summaries",
        "falsifier": "history summaries fail the cheap or two-of-three temporal gate",
        "citation_ids": ["ordered_target_statistics", "lightgbm"],
    },
    "E04": {
        "primary_change": "add delta-nDCG@5 weighting to the pairwise loss",
        "falsifier": "metric-aligned weighting fails to improve the pairwise parent",
        "citation_ids": ["ranknet", "lambdamart"],
    },
    "E05": {
        "primary_change": "add a small BCE stabilizer to PairLogit",
        "falsifier": "variance does not fall or either protected component regresses",
        "citation_ids": ["ranknet", "factorization_machines"],
    },
    "E06": {
        "primary_change": "add repeated-exposure count, prior outcome, and elapsed time",
        "falsifier": "repeat-exposure features fail the temporal evidence gates",
        "citation_ids": ["ordered_target_statistics", "lightgbm"],
    },
    "E07": {
        "primary_change": "add user-author and duration affinity",
        "falsifier": "affinity features fail the temporal evidence gates",
        "citation_ids": ["factorization_machines", "ordered_target_statistics"],
    },
    "E08": {
        "primary_change": "add recency-decayed point-in-time summaries",
        "falsifier": "recency weighting fails to beat the non-decayed parent",
        "citation_ids": ["ordered_target_statistics", "lightgbm"],
    },
    "E09": {
        "primary_change": "add candidate-history dot products",
        "falsifier": "dot products fail to improve at least two temporal folds",
        "citation_ids": ["factorization_machines"],
    },
    "E10": {
        "primary_change": "blend two prior branch predictions using shadow-only weights",
        "falsifier": "the blend fails to beat the stronger component on shadow folds",
        "citation_ids": ["stacked_generalization"],
    },
}


@dataclass(frozen=True)
class ExperimentCard:
    card_id: str
    mechanism: str
    prerequisite: str | None = None
    stage: str = "search"


DEFAULT_QUEUE = (
    ExperimentCard("E00", "reproduce five-seed official FM and set the incumbent", stage="baseline"),
    ExperimentCard(
        "E15",
        "mean ensemble of context-aware FMs using inference-safe hour and exposure policy",
    ),
    ExperimentCard("E01", "replace pointwise BCE with same-user fixed-K PairLogit"),
    ExperimentCard("E02", "tree ranker with point-in-time item/author rates and no-stat control"),
    ExperimentCard("E03", "simple candidate/history affinity summaries"),
    ExperimentCard("E04", "delta-nDCG@5 weighting on best pairwise model", "E01_supported"),
    ExperimentCard("E05", "small BCE stabilizer for pairwise variance", "E01_supported"),
    ExperimentCard("E06", "repeat-exposure count, prior outcome and elapsed time", "tree_supported"),
    ExperimentCard("E07", "user-author and duration affinity", "tree_supported"),
    ExperimentCard("E08", "recency-decayed point-in-time summaries", "history_supported"),
    ExperimentCard("E09", "candidate/history dot-product gate for neural history", "history_supported"),
    ExperimentCard("E10", "per-user blend of pairwise FM and tree/history branch", "two_branches_supported"),
    ExperimentCard("E11", "small DIN-like attention", "E09_supported"),
    ExperimentCard("E12", "tab-conditioned click auxiliary", "E11_supported"),
    ExperimentCard("E13", "threshold-aware watch-time auxiliary", "E12_decided"),
    ExperimentCard(
        "E14",
        "three-seed confirmation of best single and blend",
        "finalist_ready",
        stage="deferred_confirmation",
    ),
)


SUPPORTED_FLAGS = {
    "E15": {"context_ensemble_supported", "finalist_ready"},
    "E01": {"E01_supported"},
    "E02": {"tree_supported"},
    "E03": {"history_supported"},
    "E06": {"history_supported"},
    "E07": {"history_supported"},
    "E08": {"history_supported"},
    "E09": {"E09_supported"},
    "E10": {"finalist_ready"},
    "E11": {"E11_supported"},
    "E12": {"E12_decided"},
}


class SearchPolicy:
    def __init__(self, cards: tuple[ExperimentCard, ...] = DEFAULT_QUEUE):
        self.cards = cards

    def next_card(self, attempted: set[str], evidence_flags: set[str]) -> ExperimentCard | None:
        for card in self.cards:
            if card.stage != "search":
                continue
            if card.card_id in attempted:
                continue
            if card.prerequisite is None or card.prerequisite in evidence_flags:
                return card
        return None

    @staticmethod
    def evidence_flags(promoted_cards: set[str]) -> set[str]:
        flags: set[str] = set()
        for card_id in promoted_cards:
            flags.update(SUPPORTED_FLAGS.get(card_id, set()))
        tree_or_history = {"E02", "E03", "E06", "E07", "E08"}
        if "E01" in promoted_cards and promoted_cards.intersection(tree_or_history):
            flags.add("two_branches_supported")
        return flags

    @staticmethod
    def proposal_context(
        card: ExperimentCard,
        *,
        evidence_artifact_ids: list[str],
        incumbent_experiment_id: str | None,
        incumbent_primary_units: int | None,
        hypotheses_remaining: int,
        seconds_remaining: float,
    ) -> dict[str, object]:
        """Build a label-free proposal context whose claims must cite durable evidence."""

        detail = METHOD_CARD_REFERENCES.get(card.card_id, {})
        citation_ids = list(detail.get("citation_ids", []))
        return {
            "method_card": {
                "card_id": card.card_id,
                "version": METHOD_CARD_VERSION,
                "citation_id": f"method-card:{METHOD_CARD_VERSION}:{card.card_id}",
                "mechanism": card.mechanism,
                "prerequisite": card.prerequisite,
                "stage": card.stage,
                **detail,
            },
            "method_sources": [
                {"source_id": source_id, **METHOD_SOURCES[source_id]}
                for source_id in citation_ids
            ],
            "evidence_artifact_ids": sorted(set(evidence_artifact_ids)),
            "artifact_ids": sorted(set(evidence_artifact_ids)),
            "incumbent": {
                "experiment_id": incumbent_experiment_id,
                "primary_units": incumbent_primary_units,
            },
            "budget": {
                "hypotheses_remaining": max(0, hypotheses_remaining),
                "seconds_remaining": max(0.0, seconds_remaining),
            },
            "constraints": {
                "one_scientific_change": True,
                "validation_and_test_labels_forbidden": True,
                "confirmation_deferred": True,
                "test_submission_deferred": True,
            },
        }
