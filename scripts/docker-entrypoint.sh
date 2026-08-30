#!/bin/sh
set -eu

umask 077
home_root="${HOME:-/tmp/rex-home}"

copy_controller_file() {
  source_path="$1"
  target_path="$2"
  if [ -f "$source_path" ] && [ ! -L "$source_path" ]; then
    cp "$source_path" "$target_path"
    chmod 600 "$target_path"
  fi
}

# Authentication enters through read-only mounts. The CLIs receive private,
# ephemeral writable copies because both may refresh local state during a call.
mkdir -p "$home_root/.codex" "$home_root/.claude"
copy_controller_file /run/rex-auth/codex/auth.json "$home_root/.codex/auth.json"
copy_controller_file /run/rex-auth/codex/config.toml "$home_root/.codex/config.toml"
copy_controller_file /run/rex-auth/claude/settings.json "$home_root/.claude/settings.json"
copy_controller_file /run/rex-auth/claude/.credentials.json "$home_root/.claude/.credentials.json"

exec /usr/bin/tini -- python -m rex.execution.docker_controller "$@"
