# Docker production runtime

Docker is the only supported production runtime for REX on Windows, macOS, and
Linux. Native Python remains useful for development and for reading historical
evidence, but it is not a production-isolation promise. A Docker failure stops
the run; REX never falls back to an unsandboxed host process.

## Trust boundary

The versioned image is used in two roles:

- The trusted controller selects ideas, asks an authorized LLM for constrained
  patches, records evidence, and controls Docker. It alone receives the Docker
  socket and optional LLM credentials.
- Every operation that runs candidate code happens in a fresh worker. A worker
  has no network, API key, LLM authentication, Docker socket, added Linux
  capabilities, or privilege escalation. Its root filesystem and exact inputs
  are read-only. Only its experiment-specific result directory is writable.

The Docker socket grants host-level control and is therefore restricted to the
trusted controller. Never add it to the `worker` service, run a worker as
privileged, or weaken its security options. Candidate code must never run in
the controller. The launchers map the native-Linux socket group and account for
Docker Desktop's root-group socket presentation without making the controller
process root.

The controller presents three stable paths to REX:

| Container path | Access | Purpose |
|---|---|---|
| `/source` | read-only | clean, committed source revision |
| `/data` | read-only | verified KuaiRand inputs |
| `/runs` | read/write | worktrees, leases, logs, models, and reports |

When it creates a sibling worker, REX resolves these paths through the Docker
daemon's controller-container inspection. Requests outside an approved root,
including traversal and symlink escapes, fail closed.

## Build the image

Requirements are separately resolved for Linux amd64 and arm64 and every wheel
hash is locked. Regenerate them only when `requirements-linux.in` or its pinned
inputs change:

```bash
scripts/lock_linux.sh
```

The production build helper accepts only one architecture per invocation so
the image can record the exact architecture-specific lock hash:

```bash
scripts/build_docker.sh --platform linux/amd64 --tag rex:local
# Apple Silicon or ARM Linux:
scripts/build_docker.sh --platform linux/arm64 --tag rex:local
```

It refuses a dirty checkout. The image records the Git revision, dependency
lock, `pyproject.toml`, Starter Kit manifest, architecture, pinned Python base
digest, fixed Debian snapshot, and pinned Codex/Claude CLI versions. The Docker
CLI archive itself is also checked against an architecture-specific SHA-256
before installation.

Release automation builds both architectures. Publish each architecture by
digest, then create a multi-platform manifest in the registry; production runs
must use the immutable digest, not only a mutable tag.

## Configure mounts and authorization

Copy `.env.example` to the untracked `.env` file and set absolute host paths:

```text
REX_SOURCE_DIR=/absolute/path/to/repository
REX_DATA_DIR=/absolute/path/to/KuaiRand-Pure/data
REX_RUNS_DIR=/absolute/path/to/rex-runs
REX_WORKER_IMAGE=rex:local
REX_EXPECTED_IMAGE_DIGEST=sha256:...
```

The launchers resolve `REX_EXPECTED_IMAGE_DIGEST` from the locally present
image ID when it is omitted. Supplying it explicitly is recommended for a
released registry image; either way the controller refuses to start unless it
has an immutable `sha256:` identity. They also replace a mutable local tag with
that image ID before worker creation. When using Compose directly, set
`REX_WORKER_IMAGE` itself to an immutable image ID or `name@sha256:` reference.

On Windows, use forward-slash paths such as
`C:/Users/me/project/data/KuaiRand-Pure/data`. All three directories must be
shared with Docker Desktop. The supported container paths remain `/source`,
`/data`, and `/runs` on every host.

Three researcher routes are available:

- `fixed` needs no credentials and consumes no LLM tokens.
- `codex_cli` or `claude_cli` uses the pinned CLI in the image. Set
  `REX_CODEX_HOME` or `REX_CLAUDE_HOME` to the authenticated host directory.
  It is mounted read-only into the controller only.
- `openai_api` receives `OPENAI_API_KEY` and `OPENAI_MODEL` in the controller
  only and still requires REX's explicit paid-API authorization flag.

Never put a real key in the Dockerfile, Compose file, image, source tree, run
report, or model artifact. The checked-in `.env.example` intentionally contains
only empty placeholders.

## Cross-platform commands

macOS and Linux:

```bash
scripts/rex build
scripts/rex doctor --config configs/run/production.yaml --tree --llm fixed
scripts/rex run --config configs/run/production.yaml --llm fixed
```

Windows PowerShell:

```powershell
scripts/rex.ps1 build
scripts/rex.ps1 doctor --config configs/run/production.yaml --tree --llm fixed
scripts/rex.ps1 run --config configs/run/production.yaml --llm fixed
```

The launchers only prepare portable metadata and wrap the same Compose
commands; Compose remains the runtime on every platform. If invoking Compose
directly, export the Git and file-hash build arguments shown in `compose.yaml`,
build from a clean checkout, and set both `REX_WORKER_IMAGE` and
`REX_EXPECTED_IMAGE_DIGEST` to the immutable built identity before `compose
run`. The launchers perform those error-prone metadata steps automatically.

For an API-backed run, append `--llm openai_api --authorize-paid-api`. For an
authenticated local CLI, select `codex_cli` or `claude_cli`. Explicit modes do
not silently switch providers.

## Doctor and production guarantees

Run `doctor` before the first paid call and after any Docker, image, mount, or
host upgrade. The Docker doctor creates a disposable worker and proves the
expected image and labels, non-root user, read-only root, approved reads,
protected write denial, exact output write, absent socket and credentials,
disabled network, resource limits, non-executable temporary storage, and
interrupt/recovery lifecycle. Any mismatch is fatal.

Each worker follows a create, inspect, persist-lease, start, monitor, collect,
and remove handshake. Leases bind the exact container ID, daemon, image digest,
labels, run, experiment, attempt, and request hashes. Recovery never targets a
container by human-readable name alone. Timeout, out-of-memory, malformed
results, corrupted artifacts, and alignment failures are recorded without
replacing the previous champion.

The Compose `worker` service is under the `internal` profile and exists only as
a visible policy/smoke-test fixture. Normal users do not start it. The
controller creates more narrowly mounted, experiment-specific workers.

## Cross-platform release checklist

CI verifies lock regeneration, Compose rendering, amd64 and arm64 image builds,
metadata, non-root execution, absence of secrets, and worker-policy structure.
Before a release, manually run build, Docker doctor, and the fixture crash
rehearsal on Windows Docker Desktop, Apple Silicon Docker Desktop, and native
Linux Docker Engine. At least one full scientific queue must reproduce the
known validation baseline and canonical row identities inside Docker before
the release image digest is approved.

The older macOS sandbox can remain for one transition release as an explicit
rollback backend. It is not an automatic fallback and does not change the
Docker-only production contract.
