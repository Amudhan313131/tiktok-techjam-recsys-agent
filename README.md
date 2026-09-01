# REX — evidence-audited recommendation research autopilot

REX is a guarded autonomous experiment runner for the KuaiRand-Pure
`long_view` ranking task. It includes the production validation search, a clean
one-command dress-rehearsal envelope, and a separately authorized final
submission path. The small generated-fixture path remains available for fast
tests. Docker is the supported production runtime on Windows, macOS, and Linux.
New users should start with the copy-and-paste
[`docs/how-to-run.md`](docs/how-to-run.md) guide. Security design and advanced
runtime details are in [`docs/docker-production.md`](docs/docker-production.md).
The final technical results, iteration log, and artifact identities are under
[`submission/`](submission/).

The benchmark contract is in [`docs/task_contract.md`](docs/task_contract.md).
The exact implemented scope and exclusions are in
[`docs/current_phase_plan.md`](docs/current_phase_plan.md).

## What the production command does

One command connects the complete validation research loop:

```text
select one eligible method card
  -> use its versioned config, or ask an authorized LLM for one constrained patch
  -> reject protected paths and unsafe capabilities
  -> compile and test the candidate in its exact Git worktree
  -> run a cheap temporal shadow experiment
  -> run all three shadow folds only if the cheap gate passes
  -> compare uncertainty and temporal support across discovery candidates
  -> obtain an evidence-bound diagnosis
  -> continue with the next eligible method card or lock one finalist
  -> consume one atomic official-validation evaluation for that finalist
  -> record metrics, resources, hashes, diagnostics, and failures
  -> seal the validation champion or preserve the prior champion
```

The fixed queue can run without an LLM. In addition to the earlier FM, history,
and tree controls, its larger-margin block covers ensemble isolation,
`user×tab` and `video×tab` crosses, inference-safe item and user metadata,
field-weighted FM, candidate-conditioned recency, strictly historical feedback,
a regularized tree ranker, and a shadow-OOF blend. Each comparison is bound to
a control so the intended scientific change is isolated. The complete method
cards and current direct evidence are documented in
[`docs/larger_margin_research.md`](docs/larger_margin_research.md).

The loop stops at the first applicable condition: the method queue is
exhausted, 50 hypotheses/evaluations are consumed, or the configured six-hour
upper bound reaches its finalization reserve. The three-transaction `0.002`
epsilon plateau is also available, but cannot stop discovery until at least
six valid research families have been evaluated. The upper bound is a safety
budget, not a requirement to keep running for six hours.

Every run is durable in SQLite plus a hash-chained event ledger. Candidate
code executes from an exact clean Git worktree. Model workers receive only
explicit filesystem capabilities, a sanitized environment, no network, and
CPU/memory/time limits. A failed candidate gets at most two typed repairs and
cannot replace the previous validation champion.

## Verified real-data baseline

The local KuaiRand-Pure archive has been independently checked against the
frozen file hashes and temporal split contract:

| Split | Rows | Dates used |
|---|---:|---|
| Train | 1,141,112 | observed 2022-04-09 through 2022-04-21 |
| Validation | 124,909 | 2022-04-22 through 2022-04-28 |
| Hidden test features | 170,588 | 2022-04-29 through 2022-05-08 |

The five-seed FM reproduction on validation produced mean primary
`0.6015720538` with standard deviation `0.0003161765`, matching the frozen
`0.6016` reference. Random and popularity controls produced `0.4826598382` and
`0.5807219293`. The evidence is written under
`runs/baseline-independent-v2/summary.json` in this workspace.

No hidden-test target was loaded or scored. The generated test feature view has
no target path, and no hidden-test score is claimed.

## Historical production validation search

The earlier fixed-provider production loop was executed from an isolated clean source
snapshot as run `production-20260830-165853-15087c`. It stopped automatically
after three non-improving transactions (`epsilon_plateau`) and preserved the
best FM seed as the validation champion:

| Evidence | Primary result | Outcome |
|---|---:|---|
| Selected FM baseline | `0.6020372016` | retained |
| E01 pairwise FM, mean A/B/C delta | `-0.0175319` | rejected before official validation |
| E02 LightGBM + point-in-time statistics, official valid | `0.5978742331` | rejected (`-0.0041630`) |
| E03 history length, cheap delta | `+0.0005442` | rejected below the `+0.001` gate |

The audit bundle is under
`runs/production-20260830-165853-15087c/`. That three-card result is historical;
the current queue includes the larger-margin E16-E30 block and family-aware
plateau guard described above. No hidden-test prediction or scoring was run.

## Final verified submission

The clean immutable Docker run `r3-docker-20260831-codex-v15` completed and
selected E15 as its validation champion. The five-member mean context-FM
ensemble reached GAUC `0.6702043`, nDCG@5 `0.5371096`, and primary `0.6036570`.
That is `+0.0016198` over the strongest reproduced baseline seed (`0.6020372`)
and `+0.0020570` over the supplied `0.6016` reference. The run stopped through
its recorded epsilon-plateau rule after 3 of 50 iterations, used 310,811 LLM
tokens and 1,484.2 seconds of agent wall-clock time, and recorded 0 manual
interventions and 0 GPU-hours.

The Docker envelope deliberately terminated one active controller and recovered
the same run automatically. The separately authorized finalizer then predicted
exactly 170,588 test rows, passed the organizer's format/alignment checker
twice, sealed the model, predictions, CSV, reports, and hashes, and completed a
one-time filesystem handoff. The seal records `test_scored: false`; no hidden
test metric is claimed. The final technical evidence is under [`submission/`](submission/).

Later larger-margin shadow research remains documented in
[`docs/larger_margin_research.md`](docs/larger_margin_research.md), but it did
not produce a completed eligible run that supersedes V15 and is not part of the
final submission.

## Development setup

Use a project-local environment. The `tree` extra installs the pinned
LightGBM 4.7.0 and scikit-learn 1.9.0 runtime.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[tree,dev]'
```

Place the three KuaiRand-Pure CSV files under
`data/KuaiRand-Pure/data/`, then build the hashed feature views and isolated
target vault:

```bash
.venv/bin/python -m rex.cli bootstrap \
  --data-dir data/KuaiRand-Pure/data \
  --output-dir runs/data

.venv/bin/python -m rex.cli baseline \
  --data-dir data/KuaiRand-Pure/data \
  --view-dir runs/data \
  --seeds 0,1,2,3,4
```

The bootstrap verifies the frozen raw file identities, split dates, row
alignment, forbidden inference columns, and the absence of a test target. Safe
views contain prediction-time context, deterministic temporal keys, and
allow-listed basic user/video metadata. Outcome-derived monthly video
statistics are explicitly forbidden. Train/validation-only auxiliary feedback
is stored in a separate vault for strictly historical recipes; no test feedback
vault is created. Production startup repeats the baseline evidence gate before
search begins.

## Production prerequisites

Production execution uses Docker on Windows, macOS, and Linux. Native Python is
not a production-isolation promise, and Docker failures fail closed without an
unsandboxed fallback. The beginner guide covers pulling the code, placing the
dataset, authenticating Codex/Claude/OpenAI, building the image, and starting a
real run:

[`docs/how-to-run.md`](docs/how-to-run.md)

The trusted controller alone receives Docker and optional LLM credentials;
every generated code path runs in a disposable, networkless, non-root worker
with read-only inputs and bounded resources. Full image provenance, mount, and
security details are documented in
[`docs/docker-production.md`](docs/docker-production.md).

The project root must also be a **clean committed Git checkout**. This is a
provenance and recovery requirement: every experiment records its parent and
candidate commit, and resumption verifies the same root commit. The production
command refuses to start while tracked or untracked implementation files are
present. Commit the implementation intentionally before launching; do not use
a cleanup command that discards work.

## Choose the researcher

The LLM manages research; it does not make video predictions. LightGBM is one
of the prediction models; it does not propose or diagnose experiments.

### Fixed queue, no LLM

This deterministic mode executes the versioned method-card configs:

```bash
scripts/rex run \
  --config configs/run/production.yaml \
  --llm fixed
```

### Locally authenticated Codex CLI

REX invokes `codex exec` in an empty temporary directory with read-only,
non-interactive, ephemeral, JSON-schema-constrained settings. It uses the local
CLI login; it does not attach to this Codex desktop task.

```bash
codex
export REX_CODEX_HOME="$HOME/.codex"
scripts/rex doctor \
  --config configs/run/production.yaml --llm codex_cli
scripts/rex doctor \
  --config configs/run/production.yaml --llm codex_cli --live
scripts/rex run \
  --config configs/run/production.yaml --llm codex_cli
```

### Locally authenticated Claude CLI

REX invokes `claude --print` in an empty temporary directory. Tools, slash
commands, Chrome, MCP, settings sources, and session persistence are disabled;
the response must match the requested JSON Schema.

```bash
claude
export REX_CLAUDE_HOME="$HOME/.claude"
scripts/rex doctor \
  --config configs/run/production.yaml --llm claude_cli
scripts/rex doctor \
  --config configs/run/production.yaml --llm claude_cli --live
scripts/rex run \
  --config configs/run/production.yaml --llm claude_cli
```

### OpenAI API key

Secrets and the model name come from environment variables only. Responses are
strictly structured, API storage and SDK retries are disabled, and durable call
and token ceilings survive resumption.

```bash
export OPENAI_API_KEY='your-key'
export OPENAI_MODEL='your-available-model-id'

scripts/rex doctor \
  --config configs/run/production.yaml --llm openai_api
scripts/rex doctor \
  --config configs/run/production.yaml --llm openai_api --live
scripts/rex run \
  --config configs/run/production.yaml --llm openai_api \
  --authorize-paid-api
```

`auto` tries the local Codex CLI and then the local Claude CLI. Paid OpenAI API
fallback is disabled unless explicitly authorized:

```bash
scripts/rex run \
  --config configs/run/production.yaml \
  --llm auto \
  --allow-paid-api-fallback
```

Explicit modes do not silently switch providers.

## Resume, inspect, and export evidence

```bash
# Resume the exact transaction recorded for an interrupted run.
scripts/rex run \
  --config configs/run/production.yaml \
  --llm fixed \
  --resume PRODUCTION_RUN_ID

# Inspect durable state, repairs, sessions, baseline, and search promotions.
scripts/rex status \
  --config configs/run/production.yaml \
  --run-id PRODUCTION_RUN_ID

# Export the full evidence index and verify the event chain.
scripts/rex report \
  --database /runs/PRODUCTION_RUN_ID/state.sqlite3 \
  --run-id PRODUCTION_RUN_ID \
  --output-dir /runs/PRODUCTION_RUN_ID/report
```

The evidence export includes baseline artifacts and pre-experiment LLM failure
artifacts even though those records do not yet have an experiment ID.

## Clean one-command Docker R3 rehearsal

The Docker supervisor starts trusted controller containers, performs bootstrap
and the fail-closed Docker doctor, launches the validation-only autopilot,
force-kills the first controller after it has durably leased a worker, and
resumes the exact run under its original deadline. It seals the scientific
iteration log, metrics, recovery evidence, manual-intervention count, resource
usage, environment identity, and validation champion without predicting or
scoring hidden test rows:

```bash
python3 scripts/run_docker_rehearsal.py start \
  --source-root "$PWD" \
  --data-dir "$PWD/data/KuaiRand-Pure/data" \
  --output-dir /absolute/path/to/rex-docker-r3 \
  --run-id rex-docker-r3 \
  --image rex:local \
  --llm codex_cli \
  --codex-home "$HOME/.codex"
```

Use `--llm claude_cli --claude-home "$HOME/.claude"` for Claude, or `--llm
openai_api` after setting `OPENAI_API_KEY` and `OPENAI_MODEL`. The clean R3
rehearsal requires a live researcher; `fixed` is only for the deterministic
method queue. The source must be clean and committed, and the image label must
record that exact commit. Status is read-only:

```bash
python3 scripts/run_docker_rehearsal.py status \
  --output-dir /absolute/path/to/rex-docker-r3
```

## Legacy native R3 rollback rehearsal

The previous macOS R3 envelope remains for one transition release as an
explicit rollback rehearsal; it is not an automatic production fallback. It
starts its global clock before setup, clones the exact committed source into
an isolated directory, creates a fresh environment from the fully hashed lock,
verifies the Starter Kit and raw data, checks the live LLM route, runs the
validation-only autopilot, and injects exactly one coordinator `SIGKILL` after a
durable worker lease exists. It then resumes the same run and proves the attempt
was recorded once, the prior champion survived, and no test prediction or test
metric was created.

Agent-authored diffs are checked against byte-exact allowlisted snapshots. An
inapplicable diff, Python syntax error, static-gate failure, or fixture-test
failure is preserved as evidence and may receive at most two new,
attempt-scoped coding repairs from one shared budget. Every rejected patch is
reversed and the parent worktree is verified before another attempt. Protected
path and sandbox-policy failures are never retried. If the coordinator itself
exits before training, the envelope may relaunch the same run at most twice,
only when the database proves preparation made durable progress.

Full-shadow folds use a bounded deterministic executor. The shipped 8-core,
16-GB rehearsal profile permits six 2-GB, one-thread model pipelines, allowing
candidate and control for folds A, B, and C to run together. Fit still precedes
prediction within each pipeline, and evidence is always combined in A/B/C
order regardless of completion order.

Two optional shared caches avoid scientifically identical work across clean
runs. The baseline cache is keyed by data, baseline code/config, Python
environment, and evaluator hashes; every hit replays the official validation
evaluator. The control cache includes the exact config, data/fold, environment,
seed, feature provenance, and source commit. Both caches are immutable,
validation-only, atomically published, copied into run-local evidence, and
quarantine corrupt entries. They never contain test targets or test scoring.

The output directory must be outside the repository and must not already
contain another run:

```bash
python3 scripts/run_clean_rehearsal.py start \
  --data-dir "$PWD/data/KuaiRand-Pure/data" \
  --output-dir /absolute/path/to/rex-r3-output \
  --baseline-cache-dir /absolute/path/to/rex-shared-cache/baseline \
  --control-cache-dir /absolute/path/to/rex-shared-cache/controls \
  --llm codex_cli
```

The cache flags are optional. Their defaults are sibling directories under
`.rex-shared-cache`, outside both the source checkout and rehearsal output.

Use `--llm claude_cli` for the locally authenticated Claude CLI. Direct API
mode additionally requires `--authorize-paid-api`; `auto` uses that flag only
to authorize paid fallback after the two local CLIs.

The launcher writes compact status snapshots itself once per hour, so no LLM
call is needed merely to monitor the run. A human or scheduled read-only check
can inspect the latest durable state without restarting or modifying it:

```bash
python3 scripts/run_clean_rehearsal.py status \
  --output-dir /absolute/path/to/rex-r3-output
```

A successful R3 seal includes the source commit and tree identity, dependency
lock and installed environment, data and evaluator hashes, actual provider
calls, injected-fault and recovery evidence, all hourly snapshots, the complete
report, and every member of the winning validation bundle. Success is written
only if evidence sealing itself finishes before the six-hour ceiling.

## Final submission generation and handoff

R3 is validation-only. Test prediction is a separate operation and requires an
explicit authorization flag. It accepts only a completed production run and
its immutable best-valid bundle:

```bash
scripts/rex finalize-submission \
  --run-id PRODUCTION_RUN_ID \
  --config configs/run/production.yaml \
  --data-dir /data \
  --output-dir /runs/PRODUCTION_RUN_ID/final-submission \
  --authorize-test-prediction
```

The durable job checks the source database and report without changing them,
checks out the winner's exact commit, predicts exactly 170,588 canonical test
rows with `target_view_path=null`, rejects non-finite or misaligned output,
builds `row_id,user_id,video_id,score`, and invokes only
`submit.py --check --split test`. The copied CSV is checked a second time. The
sealed result hashes the submission, predictions, config, model manifest and
checkpoint members, source report, checker transcripts, commit, and evidence;
it explicitly records `test_scored: false`.

Copying the sealed bundle is a separate one-time handoff bound to its exact seal
hash:

```bash
scripts/rex handoff-submission \
  --run-id PRODUCTION_RUN_ID \
  --job-id SUBMISSION_JOB_ID \
  --seal-sha256 SEALED_MANIFEST_SHA256 \
  --target-dir /runs/final-handoff \
  --authorize-once
```

No upload API is invented. Organizer submission remains the single deliberate
human action after inspecting the sealed handoff.

## Recovery rehearsals

The rehearsals are bounded, offline production-control exercises. They use a
generated committed control repository; they do not open KuaiRand data, call a
live LLM, train a scientific model, confirm a winner, or create a submission.

```bash
# R1: supervisor restart and typed worker/evaluator/artifact/database faults.
.venv/bin/python -m rex.cli rehearse --level R1 \
  --output-dir runs/rehearsal-r1

# R2: R1 plus LLM timeout/schema and patch/workspace authority faults.
.venv/bin/python -m rex.cli rehearse --level R2 \
  --output-dir runs/rehearsal-r2
```

R1 covers preparation and finalization interruption, timeout, OOM, NaN,
corrupt bundles, evaluator failure, prediction misalignment, database rollback
and stale takeover, and a permanently broken candidate followed by a valid next
candidate. R2 adds provider interruption/schema recovery, protected-file and
path-traversal rejection, and sandbox workspace-escape rejection. Component
tests separately exercise real subprocess process-group termination and
resource classification.

```bash
.venv/bin/python -m pytest -q tests/fault_injection
.venv/bin/python -m pytest -q
```

## Scope after KuaiRand-Pure

The implemented release path is intentionally limited to KuaiRand-Pure. These
items remain optional follow-up work and do not reduce the required Pure score:

- KuaiRand-1k and KuaiRand-27k;
- randomized-exposure and off-policy evaluation analysis;
- neural sequence models and multi-task auxiliary objectives;
- censored watch-time modeling;
- an optional demo video and richer visualizations.

## Limitations and future improvements

- The final autonomous run converged after three scored iterations. It produced
  a verified improvement, but did not exhaust the broader feature and model
  queue.
- The winning model is a five-member deterministic ensemble, but the complete
  run was not repeated under several independent outer-run seeds. A longer
  confirmation study would provide stronger variance estimates.
- Only KuaiRand-Pure was submitted. KuaiRand-1k and KuaiRand-27k remain
  unattempted optional benchmarks.
- The result measures offline GAUC and nDCG@5. It does not establish that the
  same improvement would transfer unchanged to an online recommendation system.

With more time, we would extend the controlled search across inference-safe
metadata, temporal history, multi-feedback objectives, sequence models, and
censored watch-time losses; confirm finalists across additional seeds; and run
the same sealed workflow on the two bonus benchmarks.

## Team member contributions

- **Thangaraju Sibiraj** (also recorded as **Sibi** in earlier commits): built
  the production control plane, Docker isolation, scientific experiment and
  model/feature systems, fault recovery, evidence reporting, documentation,
  and final submission workflow.
- **Amudhan**: created the initial autonomous harness, including convergence and
  budget logic, the first training/orchestration path, schemas, and submission
  validation.

The Git history is the source of truth for these contribution summaries.

The compatibility entrypoints `training/train.py`, `agent/orchestrator.py`, and
`scripts/run_agent.sh` remain, while maintained code lives under `src/rex/`.
