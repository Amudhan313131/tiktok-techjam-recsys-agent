#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to regenerate the Linux dependency locks." >&2
  exit 1
fi

compile_lock() {
  local platform="$1"
  local output="$2"
  uv pip compile requirements-linux.in \
    --generate-hashes \
    --python-version 3.13 \
    --python-platform "$platform" \
    --only-binary :all: \
    --custom-compile-command "scripts/lock_linux.sh" \
    --output-file "$output"
}

compile_lock x86_64-manylinux_2_28 requirements-lock-linux-amd64.txt
compile_lock aarch64-manylinux_2_28 requirements-lock-linux-arm64.txt

echo "Linux dependency locks regenerated successfully."
