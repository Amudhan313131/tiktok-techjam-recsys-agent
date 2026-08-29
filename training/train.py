"""
Subprocess-callable training entrypoint. Always launched via
agent/run_training_subprocess.py, never imported directly into the agent
process — this keeps a hung/crashed training run from ever taking down the
orchestrator itself.

Matches the real Starter Kit: numpy-only, CPU-only, target label is
long_view, scored on GAUC + nDCG@5 (primary = mean of the two). No GPU,
no torch — the organizer baseline (FM, k=16) runs in ~40s on one core.

Contract with the orchestrator:
- Reads config (move id, hyperparams) from --config_json (a JSON string or path)
- Writes final metrics to --output_path as JSON:
    {"primary": ..., "gauc": ..., "ndcg5": ..., "seed": ...}
- Checkpoints on completion (every iteration gets a checkpoint — see
  docs/spec.md section 3; compute is cheap here so there's no reason to skip)
- Exits nonzero on failure (crash vs timeout vs nan_loss is distinguished by
  the subprocess wrapper, not this script)

Fill in the actual model in training/models/ (start from the organizer's
baseline.py --model fm as your reference implementation for Stage 2) — this
file wires plumbing (dev_mode sampling, seeding, checkpointing, structured
output) that should stay stable regardless of which trick-menu move is being
tested. Wire in the organizer's evaluate.py for real GAUC/nDCG@5 numbers
instead of the None placeholders below.
"""

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.data.loader import load_splits  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def train(config: dict, dev_mode: bool, seed: int, checkpoint_dir: str):
    set_seed(seed)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    train_df, val_df = load_splits(dev_mode=dev_mode, seed=seed)

    # ---- placeholder model -----------------------------------------------
    # Replace with the real model from training/models/ (start from the
    # organizer's FM baseline for Stage 2, then layer in trick-menu moves).
    # This stub exists purely to exercise the harness end-to-end.
    # TODO: wire in organizer's evaluate.py for real GAUC / nDCG@5 scoring
    # against val_df, using long_view as the label.
    # -----------------------------------------------------------------------

    if checkpoint_dir:
        ckpt_path = os.path.join(checkpoint_dir, "model.ckpt")
        with open(ckpt_path, "w") as f:
            json.dump({"seed": seed, "config": config}, f)

    metrics = {
        "primary": None,   # TODO: mean(gauc, ndcg5) from evaluate.py
        "gauc": None,
        "ndcg5": None,
        "seed": seed,
        "dev_mode": dev_mode,
        "config": config,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", required=True, help="JSON string or path to a JSON file")
    parser.add_argument("--dev_mode", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    if os.path.isfile(args.config_json):
        with open(args.config_json) as f:
            config = json.load(f)
    else:
        config = json.loads(args.config_json)

    try:
        metrics = train(
            config=config,
            dev_mode=args.dev_mode,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
        )
    except Exception as e:
        # Clean failure signal for the subprocess wrapper to type, rather
        # than a raw traceback it has to parse.
        print(json.dumps({"error": str(e), "error_type": type(e).__name__}), file=sys.stderr)
        sys.exit(1)

    with open(args.output_path, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
