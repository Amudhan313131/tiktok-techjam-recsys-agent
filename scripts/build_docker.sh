#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

platform=""
tag="${REX_WORKER_IMAGE:-rex:local}"
push=false

usage() {
  cat <<'EOF'
Usage: scripts/build_docker.sh [--platform linux/amd64|linux/arm64] [--tag IMAGE] [--push]

Builds one immutable REX architecture at a time so the image records the exact
architecture-specific dependency-lock hash. --push requires a registry tag.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --platform) platform="$2"; shift 2 ;;
    --tag) tag="$2"; shift 2 ;;
    --push) push=true; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! docker buildx version >/dev/null 2>&1; then
  echo "Docker Buildx is required." >&2
  exit 1
fi
if [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
  echo "Refusing a production image build from a dirty Git checkout." >&2
  exit 1
fi

if [ -z "$platform" ]; then
  case "$(docker info --format '{{.Architecture}}')" in
    x86_64 | amd64) platform="linux/amd64" ;;
    aarch64 | arm64) platform="linux/arm64" ;;
    *) echo "Unsupported Docker architecture." >&2; exit 1 ;;
  esac
fi
case "$platform" in
  linux/amd64) lock="requirements-lock-linux-amd64.txt" ;;
  linux/arm64) lock="requirements-lock-linux-arm64.txt" ;;
  *) echo "Only linux/amd64 and linux/arm64 are supported." >&2; exit 2 ;;
esac

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

output=(--load)
if [ "$push" = true ]; then
  output=(--push)
fi

docker buildx build \
  --platform "$platform" \
  --tag "$tag" \
  --build-arg "REX_SOURCE_COMMIT=$(git rev-parse HEAD)" \
  --build-arg "REX_DEPENDENCY_LOCK_SHA256=$(sha256_file "$lock")" \
  --build-arg "REX_PYPROJECT_SHA256=$(sha256_file pyproject.toml)" \
  --build-arg "REX_STARTER_MANIFEST_SHA256=$(sha256_file configs/frozen/starter_manifest.json)" \
  "${output[@]}" \
  .
