"""Evidence-gated experiment queue from the scientific implementation plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


METHOD_CARD_VERSION = "2.1"

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
        "falsifier": ("the five-member mean ensemble fails to beat its matched one-member control"),
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
    "E16": {
        "primary_change": "increase ensemble membership from one to five with mean aggregation fixed",
        "falsifier": "five-member mean does not beat the matched one-member mean",
        "implementation_hint": (
            "In the bound YAML only, change ensemble_members from 1 to 5. Preserve mean "
            "aggregation, epochs=7, the plugin, and every other setting. Do not edit the "
            "shared experimental wrapper because the matched one-member control executes "
            "from the same candidate snapshot."
        ),
        "citation_ids": ["bagging", "factorization_machines"],
    },
    "E17": {
        "primary_change": "add one train-fitted user-by-tab categorical cross",
        "falsifier": "the cross fails to improve two temporal folds by a safe margin",
        "citation_ids": ["factorization_machines"],
    },
    "E18": {
        "primary_change": "add one train-fitted video-by-tab categorical cross",
        "falsifier": "the cross fails to improve two temporal folds by a safe margin",
        "citation_ids": ["factorization_machines"],
    },
    "E19": {
        "primary_change": "add the predeclared inference-safe item metadata block",
        "falsifier": "item metadata fails to transfer on two temporal folds",
        "citation_ids": ["kuairand", "factorization_machines"],
    },
    "E20": {
        "primary_change": "add the predeclared coarse user metadata block",
        "falsifier": "user metadata fails to transfer on two temporal folds",
        "citation_ids": ["kuairand", "factorization_machines"],
    },
    "E21": {
        "primary_change": "replace uniform FM interactions with field-pair weights",
        "falsifier": "FwFM fails to beat the exact same-field FM control",
        "citation_ids": ["factorization_machines"],
    },
    "E22": {
        "primary_change": "add candidate-conditioned point-in-time recency summaries",
        "falsifier": "candidate recency fails the pooled temporal uncertainty gate",
        "citation_ids": ["ordered_target_statistics", "lightgbm"],
    },
    "E23": {
        "primary_change": "add strictly historical multi-feedback summaries",
        "falsifier": "prior feedback fails to improve long-view ranking safely",
        "citation_ids": ["ordered_target_statistics", "lightgbm"],
    },
    "E24": {
        "primary_change": "regularize the no-stat tree ranker before adding supported blocks",
        "falsifier": "the regularized tree branch fails temporal transfer",
        "citation_ids": ["lambdamart", "lightgbm"],
    },
    "E25": {
        "primary_change": "fit one nonnegative blend weight using shadow OOF predictions only",
        "falsifier": "the blend fails to beat its strongest component on shadow folds",
        "citation_ids": ["stacked_generalization"],
    },
    "E26": {
        "primary_change": (
            "replace the ranking-tree objective with pointwise binary classification and "
            "add only the inference-safe item metadata block"
        ),
        "falsifier": "the item-aware pointwise tree fails to beat its matched core-field control",
        "citation_ids": ["kuairand", "lightgbm"],
    },
    "E27": {
        "primary_change": (
            "add the complete predeclared static user and item metadata block to the core "
            "pointwise tree"
        ),
        "falsifier": "the complete metadata tree fails to beat the core tree on two temporal folds",
        "citation_ids": ["kuairand", "lightgbm"],
    },
    "E28": {
        "primary_change": (
            "categorically bucket candidate-conditioned prior-history signals and add them to "
            "the supported item-metadata FM"
        ),
        "falsifier": "bucketed prior history fails to improve two temporal folds",
        "citation_ids": ["factorization_machines", "ordered_target_statistics"],
    },
    "E29": {
        "primary_change": (
            "add only coarse user metadata and numeric profile fields to the core pointwise tree"
        ),
        "falsifier": "the user-only metadata tree fails to improve two temporal folds",
        "citation_ids": ["kuairand", "lightgbm"],
    },
    "E30": {
        "primary_change": (
            "increase item-metadata FM embedding dimension from 16 to 32 while holding every "
            "field and optimizer setting fixed"
        ),
        "falsifier": "the larger embedding fails to improve two temporal folds without instability",
        "citation_ids": ["kuairand", "factorization_machines"],
    },
}


@dataclass(frozen=True)
class ExperimentCard:
    card_id: str
    mechanism: str
    prerequisite: str | None = None
    stage: str = "search"
    family: str = "legacy"
    expected_gain: float = 0.0
    estimated_cost_seconds: int = 600
    memory_mb: int = 1024
    dependencies: tuple[str, ...] = ()
    target_segments: tuple[str, ...] = ()
    parent: str | None = None
    falsifier: str = ""
    diversity_target: str = ""

    def __post_init__(self) -> None:
        detail = METHOD_CARD_REFERENCES.get(self.card_id, {})
        if not self.falsifier:
            object.__setattr__(
                self,
                "falsifier",
                str(detail.get("falsifier") or f"{self.mechanism} fails its temporal gate"),
            )
        if self.family == "legacy":
            object.__setattr__(self, "family", f"legacy_{self.card_id.lower()}")
        if self.prerequisite and not self.dependencies:
            object.__setattr__(self, "dependencies", (self.prerequisite,))


DEFAULT_QUEUE = (
    ExperimentCard(
        "E00",
        "reproduce five-seed official FM and set the incumbent",
        stage="baseline",
        family="baseline_fm",
    ),
    ExperimentCard(
        "E15",
        "mean ensemble of context-aware FMs using inference-safe hour and exposure policy",
        family="context_fm_ensemble",
        expected_gain=0.0015,
        estimated_cost_seconds=900,
        memory_mb=1536,
        target_segments=("history:1-4", "history:5-19", "repeat:seen"),
        parent="baseline",
        falsifier=METHOD_CARD_REFERENCES["E15"]["falsifier"],
        diversity_target="reduce initialization variance",
    ),
    ExperimentCard(
        "E16",
        "isolation of ensemble member count with aggregation held fixed",
        family="ensemble_isolation",
        expected_gain=0.0006,
        estimated_cost_seconds=750,
        memory_mb=1536,
        parent="E15",
        falsifier=METHOD_CARD_REFERENCES["E16"]["falsifier"],
        diversity_target="verify the ensemble-size mechanism before extending it",
    ),
    ExperimentCard(
        "E17",
        "rare-backed-off user-by-tab categorical cross",
        family="context_cross",
        expected_gain=0.0003,
        estimated_cost_seconds=600,
        memory_mb=1536,
        target_segments=("tab:*", "user:warm"),
        parent="E16",
        falsifier=METHOD_CARD_REFERENCES["E17"]["falsifier"],
        diversity_target="model context-specific user preference",
    ),
    ExperimentCard(
        "E18",
        "rare-backed-off video-by-tab categorical cross",
        family="context_cross",
        expected_gain=0.0001,
        estimated_cost_seconds=600,
        memory_mb=1536,
        target_segments=("tab:*", "video:warm"),
        parent="E17",
        falsifier=METHOD_CARD_REFERENCES["E18"]["falsifier"],
        diversity_target="model context-specific item response",
    ),
    ExperimentCard(
        "E19",
        "inference-safe static item metadata block",
        family="static_metadata",
        expected_gain=0.0030,
        estimated_cost_seconds=750,
        memory_mb=1536,
        target_segments=("video:cold", "video:warm"),
        parent="E15",
        falsifier=METHOD_CARD_REFERENCES["E19"]["falsifier"],
        diversity_target="generalize beyond exact video identifiers",
    ),
    ExperimentCard(
        "E20",
        "inference-safe coarse user metadata block",
        family="static_metadata",
        expected_gain=0.0001,
        estimated_cost_seconds=750,
        memory_mb=1536,
        target_segments=("user:cold", "history:0"),
        parent="E19",
        falsifier=METHOD_CARD_REFERENCES["E20"]["falsifier"],
        diversity_target="generalize beyond exact user identifiers",
    ),
    ExperimentCard(
        "E21",
        "field-weighted FM on the strongest supported field set",
        family="field_weighted_fm",
        expected_gain=0.0001,
        estimated_cost_seconds=900,
        memory_mb=1792,
        target_segments=("all",),
        parent="best_supported_fields",
        falsifier=METHOD_CARD_REFERENCES["E21"]["falsifier"],
        diversity_target="learn field-pair importance without a deep model",
    ),
    ExperimentCard(
        "E22",
        "candidate-conditioned recency and support features",
        family="temporal_history",
        expected_gain=0.0001,
        estimated_cost_seconds=900,
        memory_mb=1792,
        target_segments=("repeat:seen", "author_affinity:seen", "history:20+"),
        parent="best_supported_fields",
        falsifier=METHOD_CARD_REFERENCES["E22"]["falsifier"],
        diversity_target="capture time-varying candidate affinity",
    ),
    ExperimentCard(
        "E23",
        "strictly historical multi-feedback summaries",
        prerequisite="candidate_recency_supported",
        family="temporal_multifeedback",
        expected_gain=0.0001,
        estimated_cost_seconds=1100,
        memory_mb=1792,
        dependencies=("candidate_recency_supported",),
        target_segments=("repeat:seen", "history:20+"),
        parent="E22",
        falsifier=METHOD_CARD_REFERENCES["E23"]["falsifier"],
        diversity_target="use prior actions unavailable to the incumbent",
    ),
    ExperimentCard(
        "E24",
        "regularized no-stat LightGBM ranking branch",
        family="boosted_tree",
        expected_gain=0.0015,
        estimated_cost_seconds=900,
        memory_mb=2048,
        target_segments=("all",),
        parent="baseline",
        falsifier=METHOD_CARD_REFERENCES["E24"]["falsifier"],
        diversity_target="produce errors distinct from factorization models",
    ),
    ExperimentCard(
        "E25",
        "shadow-only temporal out-of-fold blend",
        prerequisite="two_diverse_families_supported",
        family="oof_blend",
        expected_gain=0.0020,
        estimated_cost_seconds=450,
        memory_mb=1024,
        dependencies=("two_diverse_families_supported",),
        target_segments=("all", "user:cold", "repeat:seen"),
        parent="two_supported_branches",
        falsifier=METHOD_CARD_REFERENCES["E25"]["falsifier"],
        diversity_target="combine complementary same-row temporal errors",
    ),
    ExperimentCard(
        "E26",
        "pointwise LightGBM classifier with inference-safe item metadata",
        family="boosted_tree_pointwise",
        expected_gain=0.0025,
        estimated_cost_seconds=800,
        memory_mb=2048,
        target_segments=("all", "video:cold"),
        parent="baseline",
        falsifier=METHOD_CARD_REFERENCES["E26"]["falsifier"],
        diversity_target="produce calibrated nonlinear errors distinct from FM and LambdaRank",
    ),
    ExperimentCard(
        "E27",
        "complete static metadata block on the core pointwise tree",
        family="boosted_tree_pointwise",
        expected_gain=0.0006,
        estimated_cost_seconds=850,
        memory_mb=2048,
        target_segments=("user:cold", "history:0"),
        parent="pointwise_core",
        falsifier=METHOD_CARD_REFERENCES["E27"]["falsifier"],
        diversity_target="test whether user-item metadata interactions create a distinct branch",
    ),
    ExperimentCard(
        "E28",
        "categorical candidate-recency buckets on the item-metadata FM",
        prerequisite="metadata_supported",
        family="temporal_bucket_fm",
        expected_gain=0.0001,
        estimated_cost_seconds=1050,
        memory_mb=1792,
        dependencies=("metadata_supported",),
        target_segments=("repeat:seen", "author_affinity:seen", "history:20+"),
        parent="E19",
        falsifier=METHOD_CARD_REFERENCES["E28"]["falsifier"],
        diversity_target="let FM interact compact prior-history states with user, item, and tab",
    ),
    ExperimentCard(
        "E29",
        "coarse user metadata on the core pointwise classifier",
        family="boosted_tree_pointwise_user",
        expected_gain=0.0012,
        estimated_cost_seconds=850,
        memory_mb=2048,
        target_segments=("user:cold", "history:0"),
        parent="pointwise_core",
        falsifier=METHOD_CARD_REFERENCES["E29"]["falsifier"],
        diversity_target="isolate the useful part of the full-metadata tree",
    ),
    ExperimentCard(
        "E30",
        "larger latent dimension on the item-metadata FM",
        prerequisite="metadata_supported",
        family="fm_capacity",
        expected_gain=0.0015,
        estimated_cost_seconds=1200,
        memory_mb=2048,
        dependencies=("metadata_supported",),
        target_segments=("all", "video:cold", "video:warm"),
        parent="E19",
        falsifier=METHOD_CARD_REFERENCES["E30"]["falsifier"],
        diversity_target="increase interaction capacity only after fields are supported",
    ),
    ExperimentCard(
        "E03",
        "simple candidate/history affinity summaries",
        family="history_tree",
        expected_gain=0.0010,
        estimated_cost_seconds=750,
        memory_mb=1536,
        falsifier=METHOD_CARD_REFERENCES["E03"]["falsifier"],
        diversity_target="temporal affinity beyond exact IDs",
    ),
    ExperimentCard(
        "E02",
        "tree ranker with point-in-time item/author rates and no-stat control",
        family="boosted_tree_statistics",
        expected_gain=0.0005,
        estimated_cost_seconds=900,
        memory_mb=2048,
        falsifier=METHOD_CARD_REFERENCES["E02"]["falsifier"],
        diversity_target="nonlinear ranking residuals",
    ),
    ExperimentCard(
        "E01",
        "replace pointwise BCE with same-user fixed-K PairLogit",
        family="pairwise_fm",
        expected_gain=0.0001,
        estimated_cost_seconds=900,
        memory_mb=1536,
        falsifier=METHOD_CARD_REFERENCES["E01"]["falsifier"],
        diversity_target="metric-aligned pairwise gradients",
    ),
    ExperimentCard("E04", "delta-nDCG@5 weighting on best pairwise model", "E01_supported"),
    ExperimentCard("E05", "small BCE stabilizer for pairwise variance", "E01_supported"),
    ExperimentCard(
        "E06", "repeat-exposure count, prior outcome and elapsed time", "tree_supported"
    ),
    ExperimentCard("E07", "user-author and duration affinity", "tree_supported"),
    ExperimentCard("E08", "recency-decayed point-in-time summaries", "history_supported"),
    ExperimentCard(
        "E09", "candidate/history dot-product gate for neural history", "history_supported"
    ),
    ExperimentCard(
        "E10", "per-user blend of pairwise FM and tree/history branch", "two_branches_supported"
    ),
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
    "E16": {"ensemble_isolation_supported"},
    "E17": {"context_cross_supported"},
    "E18": {"context_cross_supported"},
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
    "E19": {"metadata_supported"},
    "E20": {"metadata_supported"},
    "E21": {"fwfm_supported", "finalist_ready"},
    "E22": {"candidate_recency_supported", "history_supported"},
    "E23": {"history_supported"},
    "E24": {"tree_supported"},
    "E25": {"finalist_ready"},
    "E26": {"tree_supported", "pointwise_tree_supported"},
    "E27": {"tree_supported", "pointwise_tree_supported"},
    "E28": {"candidate_recency_supported", "history_supported", "finalist_ready"},
    "E29": {"tree_supported", "pointwise_tree_supported"},
    "E30": {"finalist_ready"},
}


class SearchPolicy:
    def __init__(self, cards: tuple[ExperimentCard, ...] = DEFAULT_QUEUE):
        self.cards = cards

    def next_card(self, attempted: set[str], evidence_flags: set[str]) -> ExperimentCard | None:
        attempted_families = {card.family for card in self.cards if card.card_id in attempted}
        eligible = [
            card
            for card in self.cards
            if card.stage == "search"
            and card.card_id not in attempted
            and (card.prerequisite is None or card.prerequisite in evidence_flags)
            and set(card.dependencies).issubset(evidence_flags)
        ]
        if not eligible:
            return None
        isolation = next((card for card in eligible if card.card_id == "E16"), None)
        if isolation is not None:
            return isolation

        def utility(card: ExperimentCard) -> tuple[float, str]:
            novelty = 1.25 if card.family not in attempted_families else 1.0
            cost = max(1.0, card.estimated_cost_seconds / 600.0)
            memory_risk = 1.0 + max(0, card.memory_mb - 1536) / 8192.0
            return (-(card.expected_gain * novelty / cost / memory_risk), card.card_id)

        return min(eligible, key=utility)

    @staticmethod
    def evidence_flags(promoted_cards: set[str]) -> set[str]:
        flags: set[str] = set()
        for card_id in promoted_cards:
            flags.update(SUPPORTED_FLAGS.get(card_id, set()))
        tree_or_history = {"E02", "E03", "E06", "E07", "E08"}
        if "E01" in promoted_cards and promoted_cards.intersection(tree_or_history):
            flags.add("two_branches_supported")
        families = {card.family for card in DEFAULT_QUEUE if card.card_id in promoted_cards}
        if len(families) >= 2:
            flags.add("two_diverse_families_supported")
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
                "family": card.family,
                "expected_gain": card.expected_gain,
                "estimated_cost_seconds": card.estimated_cost_seconds,
                "memory_mb": card.memory_mb,
                "dependencies": list(card.dependencies),
                "target_segments": list(card.target_segments),
                "parent": card.parent,
                "falsifier": card.falsifier or detail.get("falsifier", ""),
                "diversity_target": card.diversity_target,
                **detail,
            },
            "method_sources": [
                {"source_id": source_id, **METHOD_SOURCES[source_id]} for source_id in citation_ids
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
