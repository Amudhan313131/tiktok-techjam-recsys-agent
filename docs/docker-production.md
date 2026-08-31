# Docker production guide

Docker is the supported production runtime for REX on macOS, Windows, and
Linux. It keeps generated model code away from your normal computer
environment and makes every experiment reproducible.

If you only want to run REX, start with the shorter
[four-step setup guide](how-to-run.md). This page explains the available modes,
the files Docker can access, and the production safety rules.

## The simple version

1. Pull a clean, committed copy of the repository.
2. Put KuaiRand-Pure under `data/KuaiRand-Pure/data/`.
3. Log in with Codex or Claude, or explicitly provide an OpenAI API key.
4. Build `rex:local` and start the Docker rehearsal command.

The exact copy-and-paste commands are in [How to run REX with
Docker](how-to-run.md).

## Which command should I use?

Use `scripts/run_docker_rehearsal.py` for the complete benchmark-ready dress
rehearsal. It performs setup checks, runs the validation autopilot, injects one
controlled failure, proves recovery, and seals the final evidence.

Use `scripts/rex` only when you need an individual low-level command such as
`doctor`, `run`, `status`, or `report`. Windows PowerShell users can use
`scripts/rex.ps1`. The complete rehearsal is easiest from macOS/Linux or
Windows WSL2.

## Choose the researcher

| Mode | What you need | Use for the clean rehearsal? |
|---|---|---|
| `codex_cli` | A completed Codex login in `~/.codex` | Yes |
| `claude_cli` | A completed Claude login in `~/.claude` | Yes |
| `openai_api` | `OPENAI_API_KEY` and `OPENAI_MODEL`; paid calls are explicitly authorized | Yes |
| `auto` | One or both local CLI logins; optional authorized API fallback | Yes |
| `fixed` | No LLM login | No; it is for the deterministic method queue, not live R3 research |

Explicit modes never silently switch to another provider. The LLM proposes and
diagnoses experiments; the FM or LightGBM model is what learns ranking patterns
and produces predictions.

The Docker image already contains the pinned Codex and Claude CLI programs.
Your host login directory is mounted read-only so the trusted controller can
authenticate. It is never given to candidate workers.

## Build the image

The easiest build command is:

```bash
scripts/build_docker.sh --tag rex:local
```

The script detects the current Docker architecture. You may also choose it
explicitly:

```bash
scripts/build_docker.sh --platform linux/amd64 --tag rex:local
scripts/build_docker.sh --platform linux/arm64 --tag rex:local
```

The build refuses a dirty checkout. The resulting image records the exact Git
commit, platform, Python base image, dependency lock, project configuration,
Starter Kit identity, and bundled CLI versions. Production workers use the
immutable image ID, even if you supplied a convenient local tag.

## What Docker can access

| Container path | Access | Contains |
|---|---|---|
| `/source` | Read-only | The clean, committed repository |
| `/data` | Read-only | Verified KuaiRand-Pure input files |
| `/runs` | Read/write | Experiment worktrees, logs, models, reports, and caches |

Always place the run output outside the source repository. Use a new run ID and
new output directory for every clean rehearsal. Never overwrite or resume a
sealed failed run.

## What Docker protects

REX uses two kinds of containers:

- The trusted controller selects ideas, calls the authorized LLM, records
  evidence, and starts workers.
- A fresh worker runs each candidate model. It has no network, LLM credentials,
  API key, Docker socket, privilege escalation, or writable source/data mount.

Workers are non-root, resource-limited, and read-only except for their exact
experiment result directory. A timeout, out-of-memory result, invalid metric,
corrupt artifact, or row-alignment failure is recorded and cannot replace the
previous validation champion.

The Docker socket is available only to the trusted controller because it can
control host containers. Never mount it into a worker, enable privileged mode,
or run candidate code in the controller.

## Secrets and login files

Never put a real API key in the Dockerfile, Compose file, source tree, report,
or model artifact. API mode passes `OPENAI_API_KEY` and `OPENAI_MODEL` only to
the trusted controller.

Codex and Claude login directories are mounted read-only. Required files are
copied into the controller's temporary private filesystem so normal CLI state
updates do not change the host login directory.

The optional `.env` workflow is intended for the lower-level `scripts/rex`
launcher. Copy `.env.example` to an untracked `.env` and use absolute paths:

```text
REX_SOURCE_DIR=/absolute/path/to/repository
REX_DATA_DIR=/absolute/path/to/KuaiRand-Pure/data
REX_RUNS_DIR=/absolute/path/to/rex-runs
REX_WORKER_IMAGE=rex:local
REX_CODEX_HOME=/absolute/path/to/.codex
REX_CLAUDE_HOME=/absolute/path/to/.claude
```

The four-step guide does not require an `.env` file.

## Low-level commands

The complete Docker rehearsal runs these checks automatically. When diagnosing
setup separately, macOS, Linux, and WSL users can run:

```bash
scripts/rex build
scripts/rex doctor --config configs/run/production.yaml --tree --llm fixed
scripts/rex run --config configs/run/production.yaml --llm fixed
```

Windows PowerShell equivalents are:

```powershell
scripts/rex.ps1 build
scripts/rex.ps1 doctor --config configs/run/production.yaml --tree --llm fixed
scripts/rex.ps1 run --config configs/run/production.yaml --llm fixed
```

For a local CLI, replace `fixed` with `codex_cli` or `claude_cli` and configure
the matching home directory. For direct API mode, use `--llm openai_api` with
`--authorize-paid-api` after setting the two OpenAI environment variables.

Run `doctor` after a Docker upgrade, host upgrade, mount change, or image
change. It checks the image identity, non-root user, read-only filesystem,
network isolation, credential isolation, write boundaries, resource limits,
and interrupt/recovery behavior.

## Cross-platform notes

- macOS: use Docker Desktop and the normal terminal commands.
- Linux: use Docker Engine with the Buildx and Compose v2 plugins.
- Windows: use Docker Desktop with WSL2 for the simplest complete-rehearsal
  setup. Run the shell commands from the WSL repository checkout.
- Docker Desktop users must allow Docker to access the source, data, and output
  directories when prompted.

## Advanced release notes

REX locks Linux dependencies separately for amd64 and arm64. Regenerate the
locks only when `requirements-linux.in` or its pinned inputs change:

```bash
scripts/lock_linux.sh
```

A released image should be referenced by an immutable registry digest. CI
verifies lock regeneration, Compose rendering, both architecture builds,
metadata, non-root execution, absence of secrets, and worker policy. Before a
release, maintainers should also run the Docker doctor and recovery rehearsal
on Windows Docker Desktop, Apple Silicon Docker Desktop, and native Linux.

Native Python remains useful for development and for reading old evidence, but
it is not a production isolation guarantee. Docker failures stop production;
REX does not fall back to an unsandboxed host process.
