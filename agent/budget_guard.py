"""
Dead man's switch + the official hard caps. Call check_budget(state, config)
at the top of every orchestrator loop iteration. Raises BudgetExceeded if any
cap is hit, independent of whether the score is still improving — protects
against a runaway overnight loop, and separately enforces the exact caps from
the problem statement (50 iterations / 6h wall-clock) so our own stopping
logic can never accidentally run past what's actually allowed.
"""

import os
import time

import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "budget.yaml")


class BudgetExceeded(Exception):
    pass


def load_budget_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def check_budget(state, config=None):
    config = config or load_budget_config()

    if state["tokens_used_total"] >= config["max_tokens_total"]:
        raise BudgetExceeded(
            f"Token budget exhausted: {state['tokens_used_total']} >= {config['max_tokens_total']}"
        )

    if state["iteration_count"] >= config["max_iterations"]:
        raise BudgetExceeded(
            f"Iteration cap reached: {state['iteration_count']} >= {config['max_iterations']} (official spec cap)"
        )

    started_at = time.mktime(time.strptime(state["started_at"][:19], "%Y-%m-%dT%H:%M:%S"))
    wall_elapsed = time.time() - started_at
    if wall_elapsed >= config["max_wall_clock_seconds"]:
        raise BudgetExceeded(
            f"Wall-clock cap reached: {wall_elapsed:.0f}s >= {config['max_wall_clock_seconds']}s (official 6h spec cap)"
        )

    return True
