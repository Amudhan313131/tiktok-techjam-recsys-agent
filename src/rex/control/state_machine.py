"""Explicit legal transitions for run and experiment lifecycles."""

from __future__ import annotations

from rex.contracts import ExperimentState, RunState


class IllegalTransition(RuntimeError):
    pass


RUN_TRANSITIONS: dict[RunState, set[RunState]] = {
    RunState.INITIALIZING: {RunState.BASELINE_VERIFYING, RunState.BASELINE_BLOCKED, RunState.FATAL},
    RunState.BASELINE_VERIFYING: {RunState.SEARCHING, RunState.BASELINE_BLOCKED, RunState.FATAL},
    RunState.SEARCHING: {
        RunState.FINALIZING,
        RunState.BUDGET_EXHAUSTED,
        RunState.FATAL,
    },
    RunState.FINALIZING: {RunState.COMPLETE, RunState.FATAL, RunState.BUDGET_EXHAUSTED},
    RunState.COMPLETE: set(),
    RunState.BASELINE_BLOCKED: set(),
    RunState.BUDGET_EXHAUSTED: {RunState.FINALIZING},
    RunState.FATAL: set(),
}


EXPERIMENT_TRANSITIONS: dict[ExperimentState, set[ExperimentState]] = {
    ExperimentState.PROPOSED: {
        ExperimentState.WORKTREE_READY,
        ExperimentState.REJECTED,
        ExperimentState.ABANDONED,
    },
    ExperimentState.WORKTREE_READY: {
        ExperimentState.PATCHED,
        ExperimentState.FAILED_REPAIRABLE,
        ExperimentState.REJECTED,
    },
    ExperimentState.PATCHED: {
        ExperimentState.STATIC_VALID,
        ExperimentState.FAILED_REPAIRABLE,
        ExperimentState.REJECTED,
    },
    ExperimentState.STATIC_VALID: {
        ExperimentState.FIXTURE_VALID,
        ExperimentState.FAILED_REPAIRABLE,
        ExperimentState.REJECTED,
    },
    ExperimentState.FIXTURE_VALID: {
        ExperimentState.CHEAP_RUNNING,
        ExperimentState.FAILED_REPAIRABLE,
        ExperimentState.REJECTED,
    },
    ExperimentState.CHEAP_RUNNING: {
        ExperimentState.CHEAP_COMPLETE,
        ExperimentState.FAILED_REPAIRABLE,
        ExperimentState.FAILED_FINAL,
    },
    ExperimentState.CHEAP_COMPLETE: {
        ExperimentState.FULL_RESERVED,
        ExperimentState.REJECTED,
        ExperimentState.ABANDONED,
    },
    ExperimentState.FULL_RESERVED: {
        ExperimentState.FULL_RUNNING,
        ExperimentState.FAILED_FINAL,
    },
    ExperimentState.FULL_RUNNING: {
        ExperimentState.FULL_COMPLETE,
        ExperimentState.FAILED_REPAIRABLE,
        ExperimentState.FAILED_FINAL,
    },
    ExperimentState.FULL_COMPLETE: {
        ExperimentState.DIAGNOSED,
        ExperimentState.FAILED_FINAL,
    },
    ExperimentState.DIAGNOSED: {
        ExperimentState.OFFICIAL_VALID_RUNNING,
        ExperimentState.CONFIRMING,
        ExperimentState.REJECTED,
        ExperimentState.ABANDONED,
    },
    ExperimentState.OFFICIAL_VALID_RUNNING: {
        ExperimentState.OFFICIAL_VALID_COMPLETE,
        ExperimentState.FAILED_REPAIRABLE,
        ExperimentState.FAILED_FINAL,
        ExperimentState.ABANDONED,
    },
    ExperimentState.OFFICIAL_VALID_COMPLETE: {
        ExperimentState.PROMOTED,
        ExperimentState.REJECTED,
        ExperimentState.ABANDONED,
    },
    ExperimentState.CONFIRMING: {
        ExperimentState.CONFIRMED,
        ExperimentState.REJECTED,
        ExperimentState.FAILED_FINAL,
    },
    ExperimentState.CONFIRMED: {
        ExperimentState.SUBMISSION_BUILDING,
        ExperimentState.REJECTED,
    },
    ExperimentState.SUBMISSION_BUILDING: {
        ExperimentState.SUBMISSION_VALID,
        ExperimentState.FAILED_FINAL,
    },
    ExperimentState.SUBMISSION_VALID: {
        ExperimentState.PROMOTED,
        ExperimentState.REJECTED,
    },
    ExperimentState.FAILED_REPAIRABLE: {
        ExperimentState.REPAIRING,
        ExperimentState.FAILED_FINAL,
        ExperimentState.ABANDONED,
    },
    ExperimentState.REPAIRING: {
        ExperimentState.PATCHED,
        ExperimentState.CHEAP_RUNNING,
        ExperimentState.FULL_RUNNING,
        ExperimentState.OFFICIAL_VALID_RUNNING,
        ExperimentState.FAILED_FINAL,
    },
    ExperimentState.PROMOTED: set(),
    ExperimentState.REJECTED: set(),
    ExperimentState.ABANDONED: set(),
    ExperimentState.FAILED_FINAL: set(),
}


def require_run_transition(current: RunState, next_state: RunState) -> None:
    if next_state not in RUN_TRANSITIONS[current]:
        raise IllegalTransition(f"illegal run transition: {current} -> {next_state}")


def require_experiment_transition(current: ExperimentState, next_state: ExperimentState) -> None:
    if next_state not in EXPERIMENT_TRANSITIONS[current]:
        raise IllegalTransition(f"illegal experiment transition: {current} -> {next_state}")
