"""Compatibility CLI for the frozen RunRequest/RunResult worker contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rex.contracts import RunRequest  # noqa: E402
from rex.execution.artifacts import atomic_write_json  # noqa: E402
from rex.execution.worker import execute  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_json", "--request", dest="request", required=True)
    parser.add_argument("--split", choices=["train", "shadow", "valid", "test"])
    parser.add_argument("--output_dir")
    parser.add_argument("--result")
    args = parser.parse_args(argv)
    request = RunRequest.model_validate_json(Path(args.request).read_text(encoding="utf-8"))
    if args.split and args.split != request.split:
        parser.error(f"--split={args.split} disagrees with request split={request.split}")
    if args.output_dir:
        request = request.model_copy(update={"output_dir": str(Path(args.output_dir).resolve())})
    result_path = Path(args.result or Path(request.output_dir) / "result.json")
    result = execute(request)
    atomic_write_json(result_path, result.model_dump(mode="json", by_alias=True))
    return 0 if result.status.value == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
