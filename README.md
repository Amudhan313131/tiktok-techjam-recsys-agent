# REX — evidence-audited recommendation research autopilot

REX is a guarded experiment runner for the KuaiRand-Pure `long_view` task. In
the current phase it proves the autonomous control loop on generated fixture
data only. Production model experimentation, finalist confirmation, the
six-hour dress rehearsal, and final submission creation are deliberately
disabled.

The authoritative benchmark definition is in
[`docs/task_contract.md`](docs/task_contract.md). The active scope is in
[`docs/current_phase_plan.md`](docs/current_phase_plan.md). The broader plan is
archived in [`docs/implementation_plan.md`](docs/implementation_plan.md) and is
not the plan being executed now.

## The system in plain language

There are two different kinds of optional intelligence:

- An LLM is the research manager. It proposes one idea, returns a constrained
  code patch, and diagnoses the recorded result. It never makes recommendation
  predictions and it cannot edit protected control-plane files directly.
- LightGBM is a prediction model. It learns ranking patterns from tabular data.
  It does not choose ideas, write code, or manage experiments.

The connected fixture flow is:

```text
choose one idea
  -> return a patch for an allowed model file
  -> reject unsafe paths or capabilities
  -> compile and run a fixture gate
  -> run a cheap generated-data attempt
  -> run a fuller generated-data attempt only if the cheap gate passes
  -> record metrics and artifacts
  -> diagnose the result
  -> reject or close the fixture candidate
  -> repeat within the configured budget
```

Every candidate runs from an isolated, clean Git worktree at an exact commit.
The main checkout is not modified by an LLM patch. Attempts, LLM calls,
resources, hashes, state transitions, process ownership, and recovery events
are written transactionally to SQLite and a hash-chained event ledger.

## Current safety boundary

`configs/run/fixture.yaml` is the only configuration accepted by the autopilot.
It generates tiny local arrays and explicitly reports:

```text
production_science_enabled = false
confirmation_enabled = false
final_submission_enabled = false
```

`configs/run/default.yaml` is marked `production_science_disabled`. The current
autopilot therefore cannot launch the deferred scientific queue, confirm a
winner, run for six hours, or create a competition submission.

## Setup

Use a project-local environment so LightGBM and the provider SDK do not change
your system Python:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[tree,dev]'
.venv/bin/python -m rex.cli doctor --tree --llm fixed
```

The `tree` extra installs the pinned LightGBM 4.7.0 and scikit-learn 1.9.0
versions. `doctor --tree` trains and reloads a tiny synthetic LambdaRank bundle;
it does not access competition data.

## LLM option 1: your locally authenticated Codex CLI

This uses `codex exec` as a subprocess. It uses your local Codex CLI login and
configuration; it does not attach to or control this particular Codex desktop
task. Calls are ephemeral, read-only, non-interactive, and JSON-schema
constrained. They start in an empty temporary directory so competition data is
not exposed as Codex's working directory. Codex returns a proposed diff, and
REX applies it only after its allowlist and fixture semantic checks prove that
the sole change is the approved numeric model-bias constant.

```bash
# First install/login to the Codex CLI if needed.
codex login

# Check installation/auth without making a live model request.
.venv/bin/python -m rex.cli doctor --llm codex_cli

# Make one explicit live structured health request.
.venv/bin/python -m rex.cli doctor --llm codex_cli --live

# Run the generated-fixture autopilot with Codex as the researcher.
.venv/bin/python -m rex.cli run --llm codex_cli
```

## LLM option 2: your locally authenticated Claude CLI

This uses `claude --print` through the installed Claude Code CLI. REX disables
Claude tools, slash commands, Chrome, MCP servers, settings sources, and session
persistence. Calls run from an empty temporary directory and require Claude's
JSON Schema output.

```bash
# First log in if needed.
claude auth login

# Check installation/auth without making a live model request.
.venv/bin/python -m rex.cli doctor --llm claude_cli

# Make one explicit live structured health request.
.venv/bin/python -m rex.cli doctor --llm claude_cli --live

# Run the generated-fixture autopilot with Claude as the researcher.
.venv/bin/python -m rex.cli run --llm claude_cli
```

## LLM option 3: OpenAI API key

Secrets are read from environment variables only. An API key in YAML is
rejected, error messages are redacted, API-side response storage is disabled,
SDK retries are disabled in favor of REX's bounded retry policy, and configured
call/token ceilings are enforced. Successful API usage is rehydrated from the
durable evidence store when a fixture run resumes.

```bash
export OPENAI_API_KEY='your-key'
export OPENAI_MODEL='your-available-model-id'

.venv/bin/python -m rex.cli doctor --llm openai_api
.venv/bin/python -m rex.cli doctor --llm openai_api --live
.venv/bin/python -m rex.cli run --llm openai_api
```

Automatic routing tries Codex and then local Claude before the fixed provider.
Paid OpenAI API fallback is off by default and must be explicitly enabled:

```bash
.venv/bin/python -m rex.cli run \
  --llm auto \
  --allow-paid-api-fallback
```

For completely offline deterministic testing, use `--llm fixed` (the default in
the fixture configuration).

## Run and inspect the connected autopilot

```bash
# Normal fixture-only autonomous run
.venv/bin/python -m rex.cli run --config configs/run/fixture.yaml

# Resume a previously interrupted fixture run
.venv/bin/python -m rex.cli run \
  --config configs/run/fixture.yaml \
  --resume FIXTURE_RUN_ID

# Inspect durable run, experiment, and process-session state
.venv/bin/python -m rex.cli status \
  --config configs/run/fixture.yaml \
  --run-id FIXTURE_RUN_ID

# Short connected rehearsal with provider, worker, and protected-patch faults
.venv/bin/python -m rex.cli rehearse --level FIXTURE
```

The connected rehearsal is normally a few seconds on this machine and has a
15-minute ceiling. It is not the deferred six-hour dress rehearsal.

## Crash and recovery verification

The connected rehearsal proves provider-interruption recovery, a transient NaN
worker recovery, persistent NaN exhaustion after exactly two repairs, continued
run completion, and protected-patch rejection. Integration tests also prove
resume from a pre-commit candidate and from finalization.

The wider component fault suite covers worker crashes, timeouts with descendant
cleanup, simulated OOM, missing or corrupt artifacts, dirty or wrong-commit
worktrees, provider retry/fallback, secret redaction, stale process takeover,
interrupted event export, incumbent invariance, and malformed/misaligned
organizer CSV checks. These are production-style tests of the infrastructure;
they are not a production model run.

```bash
.venv/bin/python -m pytest -q tests/fault_injection
.venv/bin/python -m pytest -q
```

## Explicitly deferred

The following work is not part of the current implementation run:

- scientific comparisons of losses, history features, LightGBM, or blends;
- three-fold and three-seed confirmation of a winning model;
- the six-hour clean-room dress rehearsal;
- final test prediction, submission CSV creation, or submission packaging.
- a general-purpose sandbox for arbitrary model-code changes; the active
  fixture runner instead enforces the narrower bias-only semantic patch.

Those phases can later consume the control plane built here, but none has been
started or claimed by the fixture evidence.

`training/train.py`, `agent/orchestrator.py`, and `scripts/run_agent.sh` remain
compatibility entrypoints. Maintained implementation code lives in `src/rex/`.
