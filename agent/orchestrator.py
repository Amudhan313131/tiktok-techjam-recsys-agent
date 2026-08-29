"""
Main agent loop. Run via: python agent/orchestrator.py

Each iteration:
  Stage 3: LLM picks the next move from the trick menu, given EDA + run log + budget
  Stage 4: cheap dev_mode test of that move — abandon early if it doesn't show promise
  Stage 5: full run via the bulletproof subprocess wrapper
  Stage 6: structured diagnosis of the result, forced JSON schema, anchored
           against the real reference numbers (random/popularity/baseline/ceiling)

After every iteration: update the dual tracker (plateau window for WHEN to
stop, best-ever for WHAT to submit — see docs/spec.md section 3). If a new
best-ever candidate appears, gate it through the submission validator (F8)
before accepting it.

Loop ends when: epsilon/N plateau triggers, OR the 50-iteration cap hits, OR
the 6h wall-clock cap hits — whichever comes first (agent/budget_guard.py).
"""

import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
import state as state_mod  # noqa: E402
from budget_guard import check_budget, load_budget_config, BudgetExceeded  # noqa: E402
from run_training_subprocess import run_training  # noqa: E402
import submission_validator  # noqa: E402
import llm_client  # noqa: E402

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
ITER_LOG_DIR = os.path.join(LOGS_DIR, "iterations")
TRICK_MENU_PATH = os.path.join(os.path.dirname(__file__), "trick_menu.yaml")

STAGE3_SYSTEM_PROMPT = """You are the reasoning module of an autonomous ML research
agent specializing in recommender systems (KuaiRand-Pure benchmark, target label
long_view, scored on GAUC and nDCG@5). Given the current EDA findings, the trick menu
with diagnosis rules, and the run log so far, decide the single next move to try.
Balance exploiting the current best-performing branch against exploring untested
moves. Respond with the move id from the trick menu and a one-paragraph justification
referencing specific evidence."""

STAGE6_SYSTEM_PROMPT = """You are the reflection module of an autonomous ML research
agent. Given the metrics from the last training run (primary, GAUC, nDCG@5) compared
to the previous best and to the known reference anchors (random=0.4753,
popularity-only=0.5715, official baseline=0.5946, ceiling=0.8645), diagnose what
happened using the recsys_reflection tool. Ground every claim in the actual numbers
provided — do not speculate beyond the evidence."""


def load_trick_menu():
    with open(TRICK_MENU_PATH) as f:
        return yaml.safe_load(f)


def write_iteration_log(iteration_num, record):
    os.makedirs(ITER_LOG_DIR, exist_ok=True)
    path = os.path.join(ITER_LOG_DIR, f"iter_{iteration_num:03d}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)


def build_stage3_context(run_log_summary, eda_findings, trick_menu, budget_cfg, agent_state):
    return f"""
EDA findings:
{json.dumps(eda_findings, indent=2)}

Trick menu + diagnosis rules:
{json.dumps(trick_menu, indent=2)}

Run log so far (most recent iterations):
{json.dumps(run_log_summary, indent=2)}

Budget remaining:
tokens_used={agent_state['tokens_used_total']}/{budget_cfg['max_tokens_total']}
iterations={agent_state['iteration_count']}/{budget_cfg['max_iterations']}
best_ever_primary={agent_state['best_ever_primary_score']}
"""


def build_stage6_context(move, metrics_result, agent_state):
    metrics = metrics_result.get("metrics") or {}
    return f"""
Move attempted: {json.dumps(move)}
Result status: {metrics_result['status']}
This run's metrics: primary={metrics.get('primary')}, gauc={metrics.get('gauc')}, ndcg5={metrics.get('ndcg5')}
Best-ever so far: primary={agent_state['best_ever_primary_score']}, gauc={agent_state['best_ever_gauc']}, ndcg5={agent_state['best_ever_ndcg5']}
Reference anchors: random=0.4753, popularity_only=0.5715, official_baseline=0.5946, ceiling=0.8645
Error (if any): {metrics_result.get('error')}
"""


def run_loop(dry_run: bool = False):
    """
    dry_run=True skips actual LLM/subprocess calls and just exercises the
    state/logging/convergence/validator plumbing with fake numbers — useful
    for testing the harness before hooking up real data/models.
    """
    agent_state = state_mod.start_process()
    budget_cfg = load_budget_config()
    trick_menu = load_trick_menu()
    run_log_summary = []
    eda_findings = {}  # TODO: populate from training/data/loader.run_recsys_eda() once, at startup

    print(f"[orchestrator] starting. iteration={agent_state['iteration_count']} "
          f"best_ever_primary={agent_state['best_ever_primary_score']}")

    dry_run_scores = [0.55, 0.58, 0.60]  # simulate an improving-then-flattening trend

    while True:
        try:
            check_budget(agent_state, budget_cfg)
        except BudgetExceeded as e:
            print(f"[orchestrator] stopping: {e}")
            agent_state["status"] = "budget_exhausted"
            agent_state["convergence_reason"] = agent_state["convergence_reason"] or str(e)
            state_mod.save_state(agent_state)
            break

        if agent_state.get("converged"):
            print(f"[orchestrator] stopping: converged ({agent_state['convergence_reason']})")
            agent_state["status"] = "converged"
            state_mod.save_state(agent_state)
            break

        agent_state = state_mod.heartbeat(agent_state)
        iter_num = agent_state["iteration_count"] + 1

        # ---- Stage 3: pick next move -----------------------------------
        if dry_run:
            move_text, usage3 = "feature_crossing (dry run placeholder)", {"input_tokens": 0, "output_tokens": 0}
        else:
            ctx3 = build_stage3_context(run_log_summary[-5:], eda_findings, trick_menu, budget_cfg, agent_state)
            move_text, usage3 = llm_client.reason_next_move(STAGE3_SYSTEM_PROMPT, ctx3)

        agent_state = state_mod.add_resource_usage(
            agent_state, tokens=usage3["input_tokens"] + usage3["output_tokens"]
        )
        move = {"raw_decision": move_text}  # TODO: parse move_text into a structured move id + params

        # ---- Stage 4: cheap test ----------------------------------------
        if dry_run:
            cheap_result = {"status": "success", "metrics": {"primary": 0.5, "gauc": 0.5, "ndcg5": 0.5}, "error": None, "wall_seconds": 0}
        else:
            cheap_result = run_training(
                config=move, dev_mode=True, seed=42, checkpoint_dir=None,
                timeout_seconds=budget_cfg["training_timeout_seconds"],
            )

        promising = cheap_result["status"] == "success"  # TODO: real promise threshold vs. current best

        full_result = None
        ckpt_dir = os.path.join(LOGS_DIR, "checkpoints", f"iter_{iter_num:03d}")
        if promising and not dry_run:
            # ---- Stage 5: full run, checkpointed every iteration -----------
            full_result = run_training(
                config=move, dev_mode=False, seed=42, checkpoint_dir=ckpt_dir,
                timeout_seconds=budget_cfg["training_timeout_seconds"],
            )
        elif dry_run:
            score = dry_run_scores[min(iter_num - 1, len(dry_run_scores) - 1)]
            full_result = {"status": "success", "metrics": {"primary": score, "gauc": score, "ndcg5": score}, "error": None, "wall_seconds": 0}
            os.makedirs(ckpt_dir, exist_ok=True)

        result_for_reflection = full_result or cheap_result
        wall_seconds = result_for_reflection.get("wall_seconds", 0)
        agent_state = state_mod.add_resource_usage(agent_state, wall_seconds_training=wall_seconds)

        metrics = result_for_reflection.get("metrics") or {}
        primary_score = metrics.get("primary") if result_for_reflection["status"] == "success" else None
        gauc = metrics.get("gauc")
        ndcg5 = metrics.get("ndcg5")

        agent_state, is_new_best = state_mod.record_iteration(
            agent_state, primary_score=primary_score, gauc=gauc, ndcg5=ndcg5,
            checkpoint_path=ckpt_dir, budget_cfg=budget_cfg,
        )

        # ---- F8: gate any new best-ever candidate through the submission
        # validator BEFORE it's trusted as the run's current answer -------
        validation_result = None
        if is_new_best and not dry_run:
            candidate_csv = os.path.join(ckpt_dir, "submission.csv")  # TODO: write real predictions here
            if os.path.exists(candidate_csv):
                validation_result = submission_validator.validate(candidate_csv)
                if not validation_result["valid"]:
                    print(f"[orchestrator] WARNING: new best-ever candidate failed submission validation: "
                          f"{validation_result['output']}")

        # ---- Stage 6: structured reflection ------------------------------
        if dry_run:
            reflection, usage6 = {
                "recsys_phenomenon_identified": "no_clear_phenomenon",
                "evidence": "dry run",
                "confidence": "low",
                "notes": "dry run placeholder",
                "next_action": "n/a",
                "next_action_category": "feature",
            }, {"input_tokens": 0, "output_tokens": 0}
        else:
            ctx6 = build_stage6_context(move, result_for_reflection, agent_state)
            reflection, usage6 = llm_client.structured_reflect(STAGE6_SYSTEM_PROMPT, ctx6)

        agent_state = state_mod.add_resource_usage(
            agent_state, tokens=usage6["input_tokens"] + usage6["output_tokens"]
        )

        iter_record = {
            "iteration": iter_num,
            "move": move,
            "cheap_test_result": cheap_result,
            "full_run_result": full_result,
            "primary_score": primary_score,
            "gauc": gauc,
            "ndcg5": ndcg5,
            "is_new_best": is_new_best,
            "submission_validation": validation_result,
            "reflection": reflection,
        }
        write_iteration_log(iter_num, iter_record)
        run_log_summary.append({
            "iteration": iter_num, "move": move, "primary_score": primary_score,
            "phenomenon": reflection.get("recsys_phenomenon_identified"),
        })

        print(f"[orchestrator] iter={iter_num} primary={primary_score} "
              f"best_ever={agent_state['best_ever_primary_score']} "
              f"phenomenon={reflection.get('recsys_phenomenon_identified')} "
              f"converged={agent_state['converged']}")

        if dry_run and iter_num >= 3:
            print("[orchestrator] dry run complete.")
            break

    print(f"[orchestrator] loop ended. status={agent_state['status']} "
          f"best_ever_primary={agent_state['best_ever_primary_score']} "
          f"at iteration {agent_state['best_ever_iteration']} "
          f"(checkpoint: {agent_state['best_ever_checkpoint_path']})")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    run_loop(dry_run=dry_run)
