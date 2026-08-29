"""Evidence-scoped proposal, coding, and diagnosis services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rex.agents.provider import ProviderResponse, StructuredProvider
from rex.contracts import ExperimentProposal, Reflection


class PatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patch: str = Field(min_length=10)
    rationale: str = Field(min_length=8)
    tests: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class AgentDecision:
    parsed: BaseModel
    response: ProviderResponse


PROPOSER_SYSTEM = """You are the proposal component of an autonomous recommender-system
researcher. Change exactly one scientific variable. Use only evidence and artifact IDs in
the supplied context. Never request labels from validation or test inference views. Return
only the required structured object."""

CODER_SYSTEM = """You are a constrained code-evolution component. Return one unified diff
that implements the accepted proposal. Modify only its declared files. Do not touch the
evaluator, data manifests, budget logic, event store, submission validator, or coordinator."""

DIAGNOSER_SYSTEM = """You are the diagnosis component of an autonomous recommender-system
researcher. Every conclusion must cite supplied artifact IDs. Separate supported findings
from uncertainty and select one next operator. Return only the required structured object."""

FIXTURE_PROPOSER_SYSTEM = """You are proving experiment-orchestration plumbing on generated
fixture data, not conducting scientific model research. Use the exact fixture iteration from
the context to set experiment_id to fixture-NNN. Propose one harmless change to DEFAULT_BIAS
in src/rex/models/experimental/fixture.py and declare only that file. Make no model-quality,
competition, confirmation, or submission claim. Return only the required structured object."""

FIXTURE_CODER_SYSTEM = """You are proving constrained patch plumbing on generated fixture
data. Return one unified diff that changes only DEFAULT_BIAS in
src/rex/models/experimental/fixture.py. The parent fixture source contains DEFAULT_BIAS = 0.0.
Do not edit, create, rename, or delete any other file. Return only the required structured
object."""

FIXTURE_DIAGNOSER_SYSTEM = """Diagnose generated fixture evidence only. Cite supplied artifact
IDs, state explicitly that the result is not scientific model evidence, and choose ABANDON as
the next operator. Return only the required structured object."""


class ProposalService:
    def __init__(self, provider: StructuredProvider):
        self.provider = provider

    def propose(self, context: dict[str, Any]) -> AgentDecision:
        response = self.provider.generate(
            role="proposal",
            system=(FIXTURE_PROPOSER_SYSTEM if context.get("fixture_only") else PROPOSER_SYSTEM),
            prompt=_context_prompt(context),
            schema=ExperimentProposal.model_json_schema(),
        )
        return AgentDecision(ExperimentProposal.model_validate(response.value), response)


class CodingService:
    def __init__(self, provider: StructuredProvider):
        self.provider = provider

    def create_patch(self, proposal: ExperimentProposal, context: dict[str, Any]) -> AgentDecision:
        response = self.provider.generate(
            role="patch",
            system=(FIXTURE_CODER_SYSTEM if context.get("fixture_only") else CODER_SYSTEM),
            prompt=_context_prompt({"proposal": proposal.model_dump(mode="json"), **context}),
            schema=PatchResponse.model_json_schema(),
        )
        return AgentDecision(PatchResponse.model_validate(response.value), response)


class DiagnosisService:
    def __init__(self, provider: StructuredProvider):
        self.provider = provider

    def diagnose(self, experiment_id: str, context: dict[str, Any]) -> AgentDecision:
        response = self.provider.generate(
            role="diagnosis",
            system=(
                FIXTURE_DIAGNOSER_SYSTEM if context.get("fixture_only") else DIAGNOSER_SYSTEM
            ),
            prompt=_context_prompt({"experiment_id": experiment_id, **context}),
            schema=Reflection.model_json_schema(),
        )
        reflection = Reflection.model_validate(response.value)
        if reflection.experiment_id != experiment_id:
            raise ValueError("diagnosis returned a different experiment_id")
        known = set(context.get("artifact_ids", []))
        unknown = set(reflection.evidence_artifact_ids) - known
        if unknown:
            raise ValueError(f"diagnosis cites unknown artifact IDs: {sorted(unknown)}")
        return AgentDecision(reflection, response)


def _context_prompt(context: dict[str, Any]) -> str:
    import json

    return json.dumps(context, indent=2, sort_keys=True)
