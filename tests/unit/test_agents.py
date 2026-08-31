from __future__ import annotations

import pytest

from rex.agents.provider import FakeProvider
from rex.agents.recovery import decide_repair
from rex.agents.search_policy import SearchPolicy
from rex.agents.services import DiagnosisService, ProposalService
from rex.contracts import AttemptStatus


def proposal_payload() -> dict:
    return {
        "experiment_id": "E01",
        "parent_id": None,
        "operator": "LOSS",
        "hypothesis": "Pairwise ranking will improve within-user impression ordering.",
        "mechanism": "Same-user positive and negative impressions create aligned gradients.",
        "primary_change": "pairwise loss",
        "files_to_change": ["src/rex/losses/experimental/pair.py"],
        "expected_metric_effects": {"primary": "increase"},
        "expected_segment_effects": {},
        "falsifier": "Cheap primary delta is below 0.001.",
        "leakage_analysis": "Training labels only and same-user complete groups.",
        "estimated_seconds": 60,
        "cheap_rung": {"fold": "A"},
        "full_rung": {"folds": ["A", "B", "C"]},
    }


def test_fake_provider_yields_validated_proposal() -> None:
    provider = FakeProvider([proposal_payload()])
    decision = ProposalService(provider).propose({"artifact_ids": []})
    assert decision.parsed.experiment_id == "E01"
    assert provider.calls[0]["role"] == "proposal"


def test_diagnosis_cannot_invent_artifact_ids() -> None:
    payload = {
        "experiment_id": "E01",
        "outcome": "supported",
        "evidence_artifact_ids": ["invented"],
        "metric_deltas": {"primary": 0.002},
        "uncertainty": "one fold",
        "next_operator": "FEATURE",
        "reusable_lesson": "Pairwise gradients helped ordering.",
    }
    with pytest.raises(ValueError):
        DiagnosisService(FakeProvider([payload])).diagnose("E01", {"artifact_ids": ["real"]})


def test_repair_is_bounded_to_two_attempts() -> None:
    assert decide_repair(AttemptStatus.SYNTAX, 0).repair
    assert decide_repair(AttemptStatus.SYNTAX, 1).repair
    assert not decide_repair(AttemptStatus.SYNTAX, 2).repair
    assert decide_repair(AttemptStatus.INTERRUPTED, 0).repair
    assert not decide_repair(AttemptStatus.CRASH, 0).repair


def test_search_policy_enforces_evidence_gate() -> None:
    policy = SearchPolicy()
    attempted = {"E00", "E01", "E02", "E03", "E15"}
    assert policy.next_card(attempted, set()) is None
    assert policy.next_card(attempted, {"E01_supported"}).card_id == "E04"
