"""
The bulletproof loop's core piece. Training NEVER runs in the orchestrator's
own process — always via subprocess.run() with a hard timeout, so a hung or
crashed training script can never freeze the whole agent.

Returns a result dict with a typed status so Stage 6 (and Stage 3's next
pick) can react differently to a timeout vs. a crash vs. diverging (NaN
loss) vs. success. No OOM category — this benchmark is numpy/CPU-only, so
GPU memory exhaustion isn't a failure mode here; NaN divergence is the
numpy-equivalent thing worth distinguishing.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "training", "train.py")


def run_training(config: dict, dev_mode: bool, seed: int, checkpoint_dir: str,
                  timeout_seconds: int) -> dict:
    """
    Returns:
      {
        "status": "success" | "timeout" | "crash" | "nan_loss",
        "metrics": {...} | None,
        "error": str | None,
        "wall_seconds": float,
      }
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
        output_path = out_f.name

    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--config_json", json.dumps(config),
        "--seed", str(seed),
        "--output_path", output_path,
    ]
    if dev_mode:
        cmd.append("--dev_mode")
    if checkpoint_dir:
        cmd += ["--checkpoint_dir", checkpoint_dir]

    start = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_seconds
        )
        wall_seconds = time.time() - start

        if proc.returncode == 0 and os.path.exists(output_path):
            with open(output_path) as f:
                metrics = json.load(f)
            return {"status": "success", "metrics": metrics, "error": None, "wall_seconds": wall_seconds}

        stderr = proc.stderr or ""
        if "nan" in stderr.lower() or "NaN" in stderr:
            status = "nan_loss"
        else:
            status = "crash"
        return {"status": status, "metrics": None, "error": stderr[-2000:], "wall_seconds": wall_seconds}

    except subprocess.TimeoutExpired:
        wall_seconds = time.time() - start
        return {
            "status": "timeout",
            "metrics": None,
            "error": f"training exceeded {timeout_seconds}s timeout",
            "wall_seconds": wall_seconds,
        }
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)
