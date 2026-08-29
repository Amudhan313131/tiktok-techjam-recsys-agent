"""Compatibility entrypoint; the maintained coordinator lives under ``rex``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rex.cli import main as rex_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--dry-run"]:
        arguments = ["rehearse", "--level", "R0"]
    if not arguments:
        print("Use `python -m rex.cli --help` for explicit run, rehearsal, and report commands.")
        return 2
    return rex_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
