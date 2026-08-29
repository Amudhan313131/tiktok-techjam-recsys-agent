#!/usr/bin/env bash
set -euo pipefail

# Entrypoint for an unattended run. Use nohup/tmux/screen on your cloud
# instance so this survives you closing the laptop lid.
#
# Usage:
#   bash scripts/run_agent.sh            # real run
#   bash scripts/run_agent.sh --dry-run  # exercise the harness without real LLM/training calls

cd "$(dirname "$0")/.."

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "ERROR: ANTHROPIC_API_KEY is not set." >&2
  exit 1
fi

mkdir -p logs/iterations logs/checkpoints

python agent/orchestrator.py "$@" 2>&1 | tee -a logs/run_console.log
