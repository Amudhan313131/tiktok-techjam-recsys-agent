"""Bounded, typed repair policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rex.contracts import AttemptStatus


REPAIRABLE = {
    AttemptStatus.SYNTAX,
    AttemptStatus.IMPORT,
    AttemptStatus.CONTRACT,
    AttemptStatus.TIMEOUT,
    AttemptStatus.OOM,
    AttemptStatus.NAN,
    AttemptStatus.INVALID_ARTIFACT,
    AttemptStatus.INTERRUPTED,
}


@dataclass(frozen=True)
class RepairDecision:
    repair: bool
    repair_number: int
    reason: str


class RepairAction(StrEnum):
    RESUME = "resume_same_action"
    PATCH = "request_constrained_patch"
    REDUCE_WORKLOAD = "reduce_workload"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class TypedRepairPlan:
    repair: bool
    consumes_budget: bool
    repair_number: int
    action: RepairAction
    reason: str
    overrides: dict[str, Any]


def decide_repair(status: AttemptStatus, repairs_already_used: int, *, maximum: int = 2) -> RepairDecision:
    if status not in REPAIRABLE:
        return RepairDecision(False, repairs_already_used, f"{status} is not safely repairable")
    if repairs_already_used >= maximum:
        return RepairDecision(False, repairs_already_used, f"repair limit {maximum} reached")
    return RepairDecision(True, repairs_already_used + 1, f"bounded repair for {status}")


def plan_repair(
    status: AttemptStatus,
    repairs_already_used: int,
    *,
    phase: str,
    maximum: int = 2,
) -> TypedRepairPlan:
    """Choose a typed, evidence-recordable repair across the entire experiment.

    Process interruption is resumed idempotently and does not consume a model
    repair. All model/config repairs share one experiment-wide allowance.
    """

    if status == AttemptStatus.INTERRUPTED:
        return TypedRepairPlan(
            True,
            False,
            repairs_already_used,
            RepairAction.RESUME,
            f"resume interrupted {phase} action from its durable reservation",
            {},
        )
    if status == AttemptStatus.CRASH:
        return TypedRepairPlan(
            False,
            False,
            repairs_already_used,
            RepairAction.TERMINAL,
            "untyped runtime crash is not safe to repair automatically",
            {},
        )
    if status not in REPAIRABLE or repairs_already_used >= maximum:
        return TypedRepairPlan(
            False,
            False,
            repairs_already_used,
            RepairAction.TERMINAL,
            f"experiment repair limit {maximum} reached or status is not repairable",
            {},
        )
    number = repairs_already_used + 1
    if status == AttemptStatus.TIMEOUT:
        return TypedRepairPlan(
            True,
            True,
            number,
            RepairAction.REDUCE_WORKLOAD,
            f"reduce bounded {phase} workload after timeout",
            {"batch_size_scale": 0.5, "workers": 1},
        )
    if status == AttemptStatus.OOM:
        return TypedRepairPlan(
            True,
            True,
            number,
            RepairAction.REDUCE_WORKLOAD,
            f"reduce memory pressure for {phase}",
            {"batch_size_scale": 0.5, "workers": 1, "threads": 1},
        )
    return TypedRepairPlan(
        True,
        True,
        number,
        RepairAction.PATCH,
        f"request a constrained {status} repair for {phase}",
        {"allowed_scope": "declared_model_files_only"},
    )
