#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [ "$#" -eq 0 ]; then
  exec python3 -m rex.cli doctor
fi
exec python3 -m rex.cli "$@"
