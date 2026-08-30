"""Minimal subprocess entry point for the immutable organizer evaluator."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    path = Path(args.evaluator).resolve(strict=True)
    spec = importlib.util.spec_from_file_location("rex_frozen_evaluator_child", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with np.load(args.input, allow_pickle=False) as payload:
        raw = module.evaluate(
            payload["user_id"].tolist(),
            payload["long_view"].tolist(),
            payload["score"].tolist(),
        )
    print(json.dumps(raw, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
