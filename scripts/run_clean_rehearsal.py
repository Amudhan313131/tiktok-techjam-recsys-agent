#!/usr/bin/env python3
"""Start or inspect the clean R3 rehearsal without requiring REX to be installed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rex.rehearsal_clean import (  # noqa: E402
    MAX_R3_SECONDS,
    R3Envelope,
    R3EnvelopeError,
    R3Options,
    compact_status_snapshot,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a clean, validation-only R3 rehearsal or take a compact status snapshot."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="run the fresh-clone R3 envelope")
    start.add_argument("--source-root", type=Path, default=ROOT)
    start.add_argument("--source-ref", default="HEAD")
    start.add_argument("--data-dir", type=Path, required=True)
    start.add_argument("--output-dir", type=Path, required=True)
    start.add_argument("--run-id")
    start.add_argument(
        "--llm",
        choices=["codex_cli", "claude_cli", "openai_api", "auto"],
        required=True,
    )
    start.add_argument(
        "--authorize-paid-api",
        action="store_true",
        help="explicitly authorize OpenAI API use (and auto-mode paid fallback)",
    )
    start.add_argument("--wall-clock-seconds", type=int, default=MAX_R3_SECONDS)
    start.add_argument("--finalization-reserve-seconds", type=int, default=1200)
    start.add_argument("--snapshot-interval-seconds", type=int, default=3600)
    start.add_argument("--lease-wait-seconds", type=int, default=10800)
    start.add_argument("--pre-injection-restart-limit", type=int, default=2)
    status = commands.add_parser("status", help="write and print one read-only compact snapshot")
    status.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "status":
            result = compact_status_snapshot(args.output_dir, reason="operator-hourly-check")
        else:
            result = R3Envelope(
                R3Options(
                    source_root=args.source_root,
                    source_ref=args.source_ref,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    llm=args.llm,
                    run_id=args.run_id,
                    wall_clock_seconds=args.wall_clock_seconds,
                    finalization_reserve_seconds=args.finalization_reserve_seconds,
                    snapshot_interval_seconds=args.snapshot_interval_seconds,
                    lease_wait_seconds=args.lease_wait_seconds,
                    pre_injection_restart_limit=args.pre_injection_restart_limit,
                    authorize_paid_api=args.authorize_paid_api,
                )
            ).run()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (R3EnvelopeError, OSError, ValueError) as error:
        print(f"r3: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
