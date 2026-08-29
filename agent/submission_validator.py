"""
F8 — Submission validator as a gate, not an afterthought.

The official Starter Kit ships submit.py --check, which validates the
required CSV format (row_id,user_id,video_id,score) against the evaluation
split. Wire THIS validator into the orchestrator's promotion step — every
time a branch is proposed as the new "best/final" candidate, run this check
BEFORE accepting it, not just once at the very end. A formatting rejection
at actual submission time is a completely avoidable way to lose points after
doing all the hard work.

Prefer calling the organizer's real submit.py --check if it's available in
the Starter Kit (most faithful to the actual grading check). This module
provides a standalone fallback implementing the same rules from the spec,
for use before the Starter Kit script is wired in or in case it's ever
unavailable in the run environment.
"""

import csv
import math
import os
import subprocess
import sys

REQUIRED_HEADER = ["row_id", "user_id", "video_id", "score"]

# Point this at the real Starter Kit's submit.py once it's downloaded, to
# use the organizer's own check instead of (or in addition to) the fallback.
OFFICIAL_SUBMIT_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "kuairand_starter_kit", "submit.py"
)


class SubmissionInvalid(Exception):
    pass


def validate_with_official_script(csv_path: str) -> dict:
    """Preferred path: shells out to the organizer's own submit.py --check."""
    if not os.path.exists(OFFICIAL_SUBMIT_SCRIPT):
        raise FileNotFoundError(
            f"Official submit.py not found at {OFFICIAL_SUBMIT_SCRIPT} — "
            f"download kuairand-starter-kit.zip and place it there, or use "
            f"validate_fallback() instead."
        )
    proc = subprocess.run(
        [sys.executable, OFFICIAL_SUBMIT_SCRIPT, "--check", csv_path],
        capture_output=True, text=True,
    )
    return {"valid": proc.returncode == 0, "output": proc.stdout + proc.stderr}


def validate_fallback(csv_path: str, expected_row_count: int = None) -> dict:
    """
    Standalone check mirroring the spec's stated validation rules:
    - correct header
    - row_id: 0-based, strictly increasing, no gaps
    - score: numeric, no NaN/Inf
    - optional: row count matches the expected evaluation split size
    """
    errors = []

    if not os.path.exists(csv_path):
        return {"valid": False, "output": f"File not found: {csv_path}"}

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != REQUIRED_HEADER:
            errors.append(f"Bad header: expected {REQUIRED_HEADER}, got {header}")

        row_count = 0
        expected_row_id = 0
        for row in reader:
            row_count += 1
            if len(row) != 4:
                errors.append(f"Row {row_count}: expected 4 columns, got {len(row)}")
                continue

            row_id_str, _user_id, _video_id, score_str = row

            try:
                row_id = int(row_id_str)
                if row_id != expected_row_id:
                    errors.append(
                        f"Row {row_count}: row_id gap or misordering — "
                        f"expected {expected_row_id}, got {row_id}"
                    )
                expected_row_id += 1
            except ValueError:
                errors.append(f"Row {row_count}: row_id '{row_id_str}' is not an integer")

            try:
                score = float(score_str)
                if math.isnan(score) or math.isinf(score):
                    errors.append(f"Row {row_count}: score is NaN/Inf")
            except ValueError:
                errors.append(f"Row {row_count}: score '{score_str}' is not numeric")

            if len(errors) > 20:
                errors.append("... too many errors, stopping early")
                break

    if expected_row_count is not None and row_count != expected_row_count:
        errors.append(f"Row count mismatch: expected {expected_row_count}, got {row_count}")

    return {"valid": len(errors) == 0, "output": "\n".join(errors) if errors else "OK"}


def validate(csv_path: str, expected_row_count: int = None) -> dict:
    """Entry point: try the official script first, fall back to the local check."""
    try:
        return validate_with_official_script(csv_path)
    except FileNotFoundError:
        return validate_fallback(csv_path, expected_row_count=expected_row_count)
