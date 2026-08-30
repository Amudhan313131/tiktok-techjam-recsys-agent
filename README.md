# REX — evidence-audited recommendation research autopilot

REX is a guarded autonomous experiment runner for the KuaiRand-Pure
`long_view` ranking task. It includes the production validation search, a clean
one-command dress-rehearsal envelope, and a separately authorized final
submission path. The small generated-fixture path remains available for fast
tests.

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
  -> run the trusted official evaluator on validation only if the full gate passes
  -> record metrics, resources, hashes, diagnostics, and failures
  -> keep the validation champion or reject the candidate
  -> obtain an evidence-bound diagnosis
  -> continue with the next eligible method card
```

The fixed queue can run without an LLM. The eligible scientific cards cover
pairwise FM, grouped LightGBM LambdaRank and its no-stat control, candidate
history, delta-nDCG weighting, BCE stabilization, repeat exposure,
user-author/duration affinity, recency, and a shadow-selected two-branch blend.
Each comparison is bound to a control so the intended scientific change is
isolated.

The loop stops at the first applicable condition: the method queue is
exhausted, three consecutive transactions fail to improve by the configured
0.002 threshold, 50 hypotheses/evaluations are consumed, or the configured
six-hour upper bound reaches its finalization reserve. That upper bound is a
safety budget, not a requirement to keep running for six hours.

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

## Verified production validation search

The fixed-provider production loop was executed from an isolated clean source
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
`runs/production-20260830-165853-15087c/`. No confirmation sweep, hidden-test
prediction, submission construction, or six-hour dress rehearsal was run.

## Setup

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
alignment, forbidden inference columns, and the absence of a test target.
Production startup repeats the baseline evidence gate before search begins.

## Production prerequisites

Production execution currently requires macOS and a working
`/usr/bin/sandbox-exec`. Other operating systems fail closed because no
equivalent backend has been implemented. Check the data manifest, sandbox, LLM
route, and optional tree model before starting:

```bash
.venv/bin/python -m rex.cli doctor \
  --config configs/run/production.yaml \
  --tree \
  --llm fixed
```

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
.venv/bin/python -m rex.cli run \
  --config configs/run/production.yaml \
  --llm fixed
```

### Locally authenticated Codex CLI

REX invokes `codex exec` in an empty temporary directory with read-only,
non-interactive, ephemeral, JSON-schema-constrained settings. It uses the local
CLI login; it does not attach to this Codex desktop task.

```bash
codex login
.venv/bin/python -m rex.cli doctor \
  --config configs/run/production.yaml --llm codex_cli
.venv/bin/python -m rex.cli doctor \
  --config configs/run/production.yaml --llm codex_cli --live
.venv/bin/python -m rex.cli run \
  --config configs/run/production.yaml --llm codex_cli
```

### Locally authenticated Claude CLI

REX invokes `claude --print` in an empty temporary directory. Tools, slash
commands, Chrome, MCP, settings sources, and session persistence are disabled;
the response must match the requested JSON Schema.

```bash
claude auth login
.venv/bin/python -m rex.cli doctor \
  --config configs/run/production.yaml --llm claude_cli
.venv/bin/python -m rex.cli doctor \
  --config configs/run/production.yaml --llm claude_cli --live
.venv/bin/python -m rex.cli run \
  --config configs/run/production.yaml --llm claude_cli
```

### OpenAI API key

Secrets and the model name come from environment variables only. Responses are
strictly structured, API storage and SDK retries are disabled, and durable call
and token ceilings survive resumption.

```bash
export OPENAI_API_KEY='your-key'
export OPENAI_MODEL='your-available-model-id'

.venv/bin/python -m rex.cli doctor \
  --config configs/run/production.yaml --llm openai_api
.venv/bin/python -m rex.cli doctor \
  --config configs/run/production.yaml --llm openai_api --live
.venv/bin/python -m rex.cli run \
  --config configs/run/production.yaml --llm openai_api \
  --authorize-paid-api
```

`auto` tries the local Codex CLI and then the local Claude CLI. Paid OpenAI API
fallback is disabled unless explicitly authorized:

```bash
.venv/bin/python -m rex.cli run \
  --config configs/run/production.yaml \
  --llm auto \
  --allow-paid-api-fallback
```

Explicit modes do not silently switch providers.

## Resume, inspect, and export evidence

```bash
# Resume the exact transaction recorded for an interrupted run.
.venv/bin/python -m rex.cli run \
  --config configs/run/production.yaml \
  --llm fixed \
  --resume PRODUCTION_RUN_ID

# Inspect durable state, repairs, sessions, baseline, and search promotions.
.venv/bin/python -m rex.cli status \
  --config configs/run/production.yaml \
  --run-id PRODUCTION_RUN_ID

# Export the full evidence index and verify the event chain.
.venv/bin/python -m rex.cli report \
  --database runs/PRODUCTION_RUN_ID/state.sqlite3 \
  --run-id PRODUCTION_RUN_ID \
  --output-dir runs/PRODUCTION_RUN_ID/report
```

The evidence export includes baseline artifacts and pre-experiment LLM failure
artifacts even though those records do not yet have an experiment ID.

## Clean one-command R3 rehearsal

R3 starts its global clock before setup, clones the exact committed source into
an isolated directory, creates a fresh environment from the fully hashed lock,
verifies the Starter Kit and raw data, checks the live LLM route, runs the
validation-only autopilot, and injects exactly one coordinator `SIGKILL` after a
durable worker lease exists. It then resumes the same run and proves the attempt
was recorded once, the prior champion survived, and no test prediction or test
metric was created.

The output directory must be outside the repository and must not already
contain another run:

```bash
python3 scripts/run_clean_rehearsal.py start \
  --data-dir "$PWD/data/KuaiRand-Pure/data" \
  --output-dir /absolute/path/to/rex-r3-output \
  --llm codex_cli
```

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
.venv/bin/python -m rex.cli finalize-submission \
  --run-id PRODUCTION_RUN_ID \
  --config configs/run/production.yaml \
  --data-dir data/KuaiRand-Pure/data \
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
.venv/bin/python -m rex.cli handoff-submission \
  --run-id PRODUCTION_RUN_ID \
  --job-id SUBMISSION_JOB_ID \
  --seal-sha256 SEALED_MANIFEST_SHA256 \
  --target-dir /absolute/path/to/final-handoff \
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

The compatibility entrypoints `training/train.py`, `agent/orchestrator.py`, and
`scripts/run_agent.sh` remain, while maintained code lives under `src/rex/`.
