# Current Phase: Larger-Margin Research and Release Path

Date: 2026-08-31

Status: the validation search, clean Docker R3 envelope, final-submission state
machine, and larger-margin E16-E30 research block are implemented. The earlier
three-card production run remains valid historical evidence, but it is not a
description of the current queue. Direct shadow research now identifies E19
inference-safe item metadata as the strongest new treatment: seed-0 mean primary
delta `+0.001281`, positive on A/B/C. This is discovery evidence from an
in-progress source tree, not a new official-validation champion. A clean
committed Docker run, multi-seed confirmation, the single locked official
validation, and any authorized test prediction remain separate runtime actions.

## Scope boundary

This phase connects the production research loop through validation-best model
freezing, a clean one-command rehearsal, and a separately authorized final
submission job. It includes real data verification, leakage-safe temporal
experiments, fixed and LLM-authored research paths, production sandboxing,
durable recovery, bounded offline crash rehearsals, clean-environment recovery,
and check-only final packaging.

The six-hour value is a maximum that covers clean setup, search, recovery,
reporting, and sealing. The run may stop earlier through convergence or a cap.
The R3 envelope remains validation-only; hidden-test prediction is never an
implicit continuation of autonomous search.

## 1. Independent data and baseline gate

The bootstrap path verifies the frozen raw CSV hashes before materializing any
view. It enforces the official dates and row counts, creates sequential row IDs,
separates train/validation targets into a label vault, and emits a test feature
view with no target path. Forbidden current-row outcome columns cannot enter an
inference view.

Observed dataset contract:

| Split | Rows | Observed dates | Target available to development code |
|---|---:|---|---|
| Train | 1,141,112 | 2022-04-09–2022-04-21 | yes, isolated vault |
| Validation | 124,909 | 2022-04-22–2022-04-28 | yes, trusted evaluator only |
| Hidden test | 170,588 | 2022-04-29–2022-05-08 | no |

Independent validation-only reproduction:

| Control | Primary |
|---|---:|
| Random | 0.4826598382 |
| Item popularity | 0.5807219293 |
| FM, five-seed mean | 0.6015720538 |
| FM, five-seed standard deviation | 0.0003161765 |

The frozen official FM validation reference is 0.6016. The baseline gate stores
all supporting artifact IDs before allowing search. It never reads or scores a
hidden-test target.

Observed production run `production-20260830-165853-15087c` selected the best
baseline seed at `0.6020372016` primary (`0.6679479199` GAUC,
`0.5361264833` nDCG@5). The fixed queue then recorded three consecutive
non-improving transactions and stopped with `epsilon_plateau`:

| Card | Evidence reached | Decision |
|---|---|---|
| E01 pairwise FM | full A/B/C | rejected; mean shadow delta `-0.01753` |
| E02 tree + point-in-time video/author rates | official validation | rejected; valid primary `0.5978742`, delta `-0.0041630` |
| E03 history length | cheap A | rejected; primary delta `+0.0005442`, below the `+0.001` cheap gate |

The exact baseline checkpoint/config/predictions remain frozen in that run's
`best-valid` bundle. Confirmation, test prediction, and submission were not
performed.

## 2. Leakage-safe experimental views

The data layer materializes chronological shadow folds A, B, and C plus a
complete-user 10% cheap view of fold A. Feature recipes use only state available
strictly before the row being scored and are cached with data, recipe, and code
identities.

Implemented recipes and controls include:

- train-fitted static item and coarse user metadata with unknown handling;
- train-fitted `user×tab` and `video×tab` categorical crosses;
- strict `time_ms` ordering with globally unique source-row tie breakers;
- candidate-conditioned user-author/user-video support and recency summaries;
- strictly historical click/like/follow/hate/long-view aggregates from a
  train/validation-only feedback vault;
- point-in-time video statistics and a no-stat tree control;
- history length / simple candidate history;
- repeat exposure, prior outcome, and elapsed time;
- user-author and duration affinity;
- recency-decayed history;
- control recipes that preserve the same pipeline while zeroing the intended
  treatment feature.

Equal-timestamp events are emitted before any state update, and apply views use
state frozen at the training cutoff. The month-level outcome-statistics side
table is forbidden. Tags are not silently synthesized when the safe source
schema cannot support them. A method must be evaluated using the actually
materialized columns, not a claim in its prose description. The detailed data
capabilities, method cards, commands, and direct evidence are in
[`larger_margin_research.md`](larger_margin_research.md).

## 3. Connected scientific supervisor

`configs/run/production.yaml` enables a single validation-only command. The
supervisor owns one durable hypothesis transaction at a time:

```text
eligible method card
  -> durable proposal and effective config
  -> isolated worktree and safety gates
  -> cheap fold A
  -> full shadow A/B/C when promising
  -> diagnostics and evidence-bound diagnosis
  -> next eligible card or lock one finalist
  -> one atomic official-validation evaluation
  -> validation champion update or rejection and seal
```

The fixed provider executes versioned configs without pretending an LLM wrote a
patch. Codex CLI, Claude CLI, and OpenAI API modes can propose and diagnose, but
their patch authority is limited to the card-specific experimental allowlist.
Each effective config is copied into durable run evidence and checked by hash so
resume does not depend on a worktree that may have been cleaned.

Method cards E01–E08, E10, E15, and the larger-margin E16-E30 block are wired to
executable adapters. E16 isolates one versus five members with mean aggregation fixed; E17/E18 test
one context cross at a time; E19/E20 add item then user metadata; E21 tests
field-pair weights; E22/E23 add strictly point-in-time history; E24 repairs the
tree control; E25 derives a shadow-only blend from two supported families;
E26/E27/E29 test pointwise metadata trees; E28 tests categorical recency buckets;
and E30 tests FM latent capacity on the item-metadata branch.
Prerequisite-gated cards do not consume experiments. E09 and neural E11–E13 are
not selectable in this phase. E14 confirmation remains deferred.

The trusted comparison gates enforce:

- one isolated treatment against its intended control;
- complete-user fold boundaries;
- cheap rejection before expensive execution;
- two-of-three temporal support for full-rung promotion;
- confidence and component-regression checks;
- official validation against the current validation champion;
- shadow-only blend weight selection;
- no test evaluation.

Exactly one convergence transaction is counted for each terminal hypothesis,
including a final failed candidate. A positive delta smaller than or equal to
epsilon can become the best recorded validation score while still advancing the
non-improvement streak. Patience 3 cannot stop the current search until at least
six valid families have been evaluated. The loop otherwise stops on queue
exhaustion, the 50-hypothesis/evaluation caps, or the wall-clock reserve. The
official-validation budget contains one durable token: discovery first locks a
finalist, then one atomic transition evaluates it and seals that result rather
than reopening model selection.

## 4. Prediction model paths

The production worker can execute:

- pointwise FM controls;
- context FM ensembles with explicit member-level evidence;
- field-weighted FM with a matched same-field FM control;
- experimental same-user pairwise FM variants;
- deterministic grouped, regularized LightGBM ranking variants;
- a two-branch pairwise/tree blend whose weights are selected using shadow
  evidence only.

Training writes a complete bundle manifest with member hashes, plugin identity,
commit, effective-config hash, environment, data-view identity, and feature
schema. Prediction reloads that bundle in a new worker process. Missing members,
hash drift, NaN/Inf values, or row misalignment are typed failures, never valid
results.

## 5. Production execution boundary

Production runs use the fail-closed Docker boundary on macOS, Windows, and
Linux. Candidate workers are networkless, non-root, credential-free, and
receive only explicit read/write mounts plus resource limits. Native macOS
`sandbox-exec` remains a legacy rollback/testing path, not the cross-platform
production promise. An unavailable Docker boundary cannot silently fall back to
unsandboxed production execution.

The root project must be a clean committed Git checkout. Candidate worktrees are
created at exact commits; the supervisor rejects dirty worktrees, wrong commits,
protected changes, path traversal, out-of-root workspaces, and unsafe Python
capabilities.

## 6. LLM provider choices

All providers implement the same strict structured contract for proposal,
patch, and diagnosis roles.

- `fixed`: deterministic method-card queue; no LLM call.
- `codex_cli`: local Codex authentication, empty temporary working directory,
  read-only and ephemeral `codex exec`, no interactive approval, JSON Schema.
- `claude_cli`: local Claude authentication, empty temporary working directory,
  tools/MCP/Chrome/settings/session persistence disabled, JSON Schema.
- `openai_api`: key and model from environment variables only, Responses API,
  strict output, `store=false`, no tools, bounded calls/tokens, no SDK retries.
- `auto`: local Codex then local Claude; paid OpenAI fallback only when the
  operator passes the explicit authorization flag.

Provider timeouts, invalid schemas, request IDs, token usage, redacted failures,
and fallback decisions are durable. A failed proposal call can occur before an
experiment exists, so reporting resolves its artifacts through the run-scoped
LLM call instead of dropping them.

## 7. Durable ownership and recovery

SQLite stores run ownership, heartbeats, experiments, state transitions,
attempt reservations, repairs, metrics, immutable artifact provenance, LLM
calls, resources, lessons, promotions, convergence decisions, and an event
outbox. Event export is replay-safe and hash chained.

Resume behavior is exact rather than heuristic:

- a live coordinator blocks a second owner;
- a proven-stale coordinator is explicitly taken over;
- the existing proposal, effective config, worktree provenance, and current
  state are reconstructed;
- repeated writes must be identical or fail closed;
- an interrupted finalization can be resumed;
- the prior validation champion remains intact after candidate failure;
- repairs are numbered one and two, with no third repair.

The finalizer creates a validation-best bundle containing the exact model,
validation predictions, config, metrics, commit, and evidence index. Its
manifest explicitly records that no test prediction was created. The release
job later treats this sealed bundle and the completed production database as
read-only inputs.

## 8. Crash rehearsal levels

R1 and R2 are short, deterministic, offline production-control rehearsals. They
operate on a generated committed repository, not KuaiRand data, and therefore
make no scientific model claim.

R1 covers:

- preparation and finalization interruption/resume;
- typed timeout, OOM, NaN, corrupt-bundle, evaluator, and alignment failures;
- database rollback plus stale-owner takeover;
- a persistent failure through repairs one and two, followed by continuation to
  the next candidate.

R2 includes R1 and adds:

- interrupted and invalid-schema LLM calls with restart;
- protected-file and path-traversal patch rejection;
- sandbox workspace-escape rejection.

The component suite supplies the lower-level evidence for real subprocess
termination, descendant cleanup, sandbox policy enforcement, resource limits,
artifact corruption, event replay, and incumbent invariance.

## 9. Commands and acceptance

Prepare data and verify the baseline:

```bash
.venv/bin/python -m rex.cli bootstrap --output-dir runs/data
.venv/bin/python -m rex.cli baseline --view-dir runs/data --seeds 0,1,2,3,4
```

Check production prerequisites, then run or resume:

```bash
.venv/bin/python -m rex.cli doctor \
  --config configs/run/production.yaml --tree --llm fixed
.venv/bin/python -m rex.cli run \
  --config configs/run/production.yaml --llm fixed
.venv/bin/python -m rex.cli run \
  --config configs/run/production.yaml --llm fixed --resume RUN_ID
```

Run the fault rehearsals:

```bash
.venv/bin/python -m rex.cli rehearse --level R1 \
  --output-dir runs/rehearsal-r1
.venv/bin/python -m rex.cli rehearse --level R2 \
  --output-dir runs/rehearsal-r2
```

Phase acceptance requires the complete static and test suites, the opt-in real
data contract test, the Docker production doctor, R1/R2 passing, and a clean
committed snapshot before real search. Direct development evidence is not a
replacement for those gates. E19 is the current shadow-fold finalist candidate,
but the previous sealed model remains the truthful validation winner until a
clean autonomous run, multi-seed confirmation, and the single official
validation establish otherwise.

## 10. Clean one-command Docker R3 envelope

`scripts/run_docker_rehearsal.py start` is the supported cross-platform outer
trust boundary. Its deadline begins before setup, and the launcher:

1. requires a clean committed source and an immutable image labeled with that
   exact commit;
2. records the source tree, image identity, dependency inventory, Starter Kit,
   evaluator, and raw-data identities;
3. bootstraps sanitized views and proves the test view has 170,588 rows and no
   target or feedback capability;
4. runs the Docker, LightGBM, and explicitly selected live-provider doctors;
5. starts one validation-only production run under the external deadline;
6. injects one controlled controller failure after durable work exists, then
   resumes the same run ID and proves its champion and counters survived;
7. audits the database, artifacts, requests, and command transcripts for any
   test prediction or scoring work; and
8. recursively revalidates the winning checkpoint, reports, logs, recovery
   evidence, hashes, source commit, and image digest before sealing the R3
   manifest.

Full folds use a resource-bounded executor. Candidate and control pipelines may
overlap, but each pipeline keeps fit before predict, and evidence is combined in
canonical A/B/C order. Optional baseline and control caches are keyed by all
scientific identities, atomically published outside the run, revalidated on
every hit, and quarantined if corrupt.

The envelope stops through convergence, the 50-hypothesis/evaluation caps, or
the external deadline. The six hours are a ceiling, not a target duration. The
success manifest is removed if contract validation or evidence sealing crosses
the deadline.

The launcher writes local snapshots without an LLM call. A scheduled monitor
may run `scripts/run_docker_rehearsal.py status --output-dir OUTPUT` every 30
minutes and report a compact summary; it must never edit source, rebuild the
image, restart a healthy process, or advance an experiment. The previous
`run_clean_rehearsal.py` path remains a legacy native rollback rehearsal, not
the production isolation boundary.

## 11. Final submission state machine

Test prediction is isolated from R3 and the research database. The
`finalize-submission` command requires `--authorize-test-prediction`, reads only
a `COMPLETE` production run, and fingerprints its database, report, best-valid
manifest, config, commit, model bundle, and every checkpoint member.

Its durable states are:

```text
CREATED -> SOURCE_VERIFIED -> WORKTREE_READY -> PREDICTING -> PREDICTED
        -> CSV_BUILT -> FIRST_CHECK_VALID -> STAGING -> SECOND_CHECK_VALID
        -> SEALED -> READY_FOR_HANDOFF -> HANDOFF_IN_PROGRESS -> HANDED_OFF
```

The prediction request uses the exact winner plugin and bundle, a detached
exact-commit worktree, the canonical test feature hash, `split=test`,
`operation=predict`, and `target_view_path=null`. Production sandboxing and the
normal worker lease/replay logic remain active.

CSV generation preserves canonical row order and duplicate user/video pairs,
requires finite scores and exactly 170,588 rows, and emits only
`row_id,user_id,video_id,score`. Both validation gates invoke only
`submit.py --check --split test`: once before staging and once against the copied
CSV. `--score` and `--make` are rejected capabilities.

The shared reporting finalizer atomically stages the immutable winner, complete
source report, predictions, submission, configs, checker transcripts, and
hashes. Resume reuses identical persisted requests and artifacts rather than
creating another test prediction. The resulting seal records
`test_scored: false`.

Filesystem handoff requires `--authorize-once` and the exact sealed-manifest
SHA-256. A prior handoff can be replay-verified but cannot be redirected to a
different destination. There is deliberately no automatic organizer upload or
local hidden-test scoring.

## 12. Deferred until after KuaiRand-Pure

KuaiRand-1k, KuaiRand-27k, randomized-exposure/OPE analysis, neural sequence
models, multi-task objectives, censored watch-time modeling, and optional demo
visualizations remain out of the critical path. They do not reduce the required
KuaiRand-Pure score and must not delay the sealed Pure submission.
