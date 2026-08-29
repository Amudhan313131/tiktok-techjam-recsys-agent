"""
Persisted agent state — the backbone of both the "provable autonomy" story
and the convergence/scoring correctness rules locked in docs/spec.md section 3.

Two independent trackers live here, on purpose:
  1. plateau_window   — last N validation primary scores, used ONLY to decide
                         WHEN to stop (epsilon/N rule).
  2. best_ever_*       — the single highest validation primary score seen at
                         ANY point so far, and the checkpoint that produced it.
                         This is WHAT gets submitted, regardless of which
                         condition actually stopped the run.
Do not merge these into one concept — see spec.md section 3 for why.

Also holds the zero-intervention proof machinery:
  - human_override_count, bumped either manually or automatically via the
    heartbeat-gap detector in start_process()
  - a heartbeat trail so a restart is structurally visible even if nobody
    remembered to log it
"""

import json
import os
from datetime import datetime, timezone

STATE_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "agent_state.json")

# If the gap between the last heartbeat and process start exceeds this,
# flag it as a likely-unlogged restart even if override_count wasn't bumped.
RESTART_GAP_THRESHOLD_SECONDS = 60 * 5  # 5 minutes — tune once real iteration time is known


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _default_state():
    return {
        "human_override_count": 0,
        "iteration_count": 0,
        # --- plateau tracker: only for deciding WHEN to stop ---
        "plateau_window": [],          # last N=3 validation primary scores
        "converged": False,
        "convergence_reason": None,    # "epsilon_plateau" | "iteration_cap" | "wall_clock_cap"
        # --- best-ever tracker: this is WHAT gets submitted ---
        "best_ever_primary_score": None,
        "best_ever_gauc": None,
        "best_ever_ndcg5": None,
        "best_ever_iteration": None,
        "best_ever_checkpoint_path": None,
        # --- resource usage, for Feasibility reporting ---
        "tokens_used_total": 0,
        "wall_seconds_training_total": 0,
        # --- autonomy proof ---
        "started_at": _now_iso(),
        "last_heartbeat": _now_iso(),
        "restart_events": [],
        "status": "running",   # running | converged | budget_exhausted | crashed
    }


def load_state():
    if not os.path.exists(STATE_PATH):
        return _default_state()
    with open(STATE_PATH, "r") as f:
        return json.load(f)


def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp_path = STATE_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, STATE_PATH)  # atomic write, avoids a truncated file on crash


def start_process(manual_restart: bool = False):
    """Call once at the top of the orchestrator's __main__."""
    state = load_state()
    now = datetime.now(timezone.utc)
    last_heartbeat = datetime.fromisoformat(state["last_heartbeat"])
    gap_seconds = (now - last_heartbeat).total_seconds()

    is_first_run = state["iteration_count"] == 0
    likely_unlogged_restart = (
        not is_first_run and not manual_restart and gap_seconds > RESTART_GAP_THRESHOLD_SECONDS
    )

    if manual_restart or likely_unlogged_restart:
        state["human_override_count"] += 1
        state["restart_events"].append({
            "timestamp": _now_iso(),
            "gap_seconds": gap_seconds,
            "manually_logged": manual_restart,
        })

    state["last_heartbeat"] = _now_iso()
    save_state(state)
    return state


def heartbeat(state):
    state["last_heartbeat"] = _now_iso()
    save_state(state)
    return state


def record_iteration(state, primary_score, gauc, ndcg5, checkpoint_path, budget_cfg):
    """
    Updates both trackers after a completed iteration, and evaluates the
    epsilon/N convergence rule. Call this once per iteration, after Stage 5.
    """
    state["iteration_count"] += 1

    # --- best-ever tracker: unconditional, independent of plateau logic ---
    is_new_best = (
        primary_score is not None
        and (state["best_ever_primary_score"] is None or primary_score > state["best_ever_primary_score"])
    )
    if is_new_best:
        state["best_ever_primary_score"] = primary_score
        state["best_ever_gauc"] = gauc
        state["best_ever_ndcg5"] = ndcg5
        state["best_ever_iteration"] = state["iteration_count"]
        state["best_ever_checkpoint_path"] = checkpoint_path

    # --- plateau tracker: only for the stop decision ---
    if primary_score is not None:
        state["plateau_window"].append(primary_score)
        n = budget_cfg["convergence"]["patience_n"]
        state["plateau_window"] = state["plateau_window"][-n:]

        if len(state["plateau_window"]) == n:
            window = state["plateau_window"]
            # Tiny tolerance guards against float rounding putting a
            # genuine plateau (e.g. exactly epsilon=0.002 apart) just over
            # the threshold — confirmed by a real test case where
            # 0.603 - 0.601 computed as 0.0020000000000000018 in float64.
            spread = max(window) - min(window)
            if spread <= budget_cfg["convergence"]["epsilon"] + 1e-9:
                state["converged"] = True
                state["convergence_reason"] = "epsilon_plateau"

    heartbeat(state)
    return state, is_new_best


def add_resource_usage(state, tokens=0, wall_seconds_training=0):
    state["tokens_used_total"] += tokens
    state["wall_seconds_training_total"] += wall_seconds_training
    save_state(state)
    return state
