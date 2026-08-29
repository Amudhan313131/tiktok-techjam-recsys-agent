"""Bounded, typed repair policy."""

from __future__ import annotations

from dataclasses import dataclass

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


def decide_repair(status: AttemptStatus, repairs_already_used: int, *, maximum: int = 2) -> RepairDecision:
    if status not in REPAIRABLE:
        return RepairDecision(False, repairs_already_used, f"{status} is not safely repairable")
    if repairs_already_used >= maximum:
        return RepairDecision(False, repairs_already_used, f"repair limit {maximum} reached")
    return RepairDecision(True, repairs_already_used + 1, f"bounded repair for {status}")
