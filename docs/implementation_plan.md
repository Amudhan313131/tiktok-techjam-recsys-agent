# Detailed Implementation Plan

Date: 2026-08-29

Status: archived broader roadmap; **not the active implementation scope**.

The active fixture-only plan is
[`docs/current_phase_plan.md`](current_phase_plan.md). Scientific model search,
winning-model confirmation, the six-hour dress rehearsal, and final submission
work in this document are explicitly deferred.

Companion research: [`docs/winning_strategy_research.md`](winning_strategy_research.md)

## 1. Outcome and success criteria

The implementation should produce one system that can start from the organizer FM, autonomously propose and apply code changes, execute and evaluate experiments, recover from failures, select the validation-best checkpoint, and emit a valid final KuaiRand-Pure submission plus a complete evidence bundle.

The build is successful only when all four layers work together:

1. **Scientific quality:** the selected model beats the official FM validation reference with evidence that the gain is not a single-seed or single-window accident.
2. **Autonomy:** at least one accepted improvement is proposed, implemented, tested, diagnosed, and promoted by the agent rather than selected manually from a fixed configuration menu.
3. **Reliability:** crashes, timeouts, invalid patches, NaNs, malformed artifacts, and process restarts are handled without corrupting the incumbent or run state.
4. **Auditability:** every final claim links to the hypothesis, code diff, command, hashes, raw evaluator output, diagnostics, reflection, and final artifact that supports it.

Internal score gates:

| Gate | Target | Meaning |
|---|---:|---|
| Benchmark parity | FM valid primary within ±0.002 of 0.6016 | The harness is trustworthy |
| Minimum competitive candidate | Three-seed valid mean at least 0.6056 | Two convergence epsilons above FM; minimum ship target, not a predicted winning score |
| Stretch candidate | Three-seed valid mean at least 0.6116 | Approximately +0.010 over FM validation |
| Stability | At least 2/3 shadow folds positive; seed std ≤0.0015 | Gain is not isolated to one window or seed |
| Metric safety | Neither GAUC nor nDCG@5 down more than 0.002 | Avoid winning one half by sacrificing the other |

The hidden-test result is unknown until organizer scoring. These targets control engineering decisions; they are not claims about the winning threshold.

## 2. Planning assumptions

- Team size is two people.
- KuaiRand-Pure is the only required benchmark. Bonus datasets are deferred until the complete Pure pipeline is finished and rehearsed.
- The organizer kit stays at `kuairand-starter-kit/`. Do not rename or relocate it during the hackathon; record its hashes and protect it from writes.
- Raw data will be downloaded separately and excluded from git.
- PyTorch and one tree-ranking library are permitted. The reference implementation being NumPy/CPU-only is not a participant constraint.
- Test labels are inaccessible to the research/model processes even if the public dataset technically contains them.
- Random-exposure rows and supplementary captions/categories remain disabled for training unless organizers approve them in writing.
- The stable platform and model primitives may be team-authored before the judged autonomous run. The agent must still choose, modify, test, and combine scientific changes itself.

## 3. Non-negotiable architecture decisions

1. **Replace the center, preserve compatibility temporarily.** New logic lives under `src/rex/`. Existing `agent/orchestrator.py` and `training/train.py` become thin warning/compatibility shims only after the new path is working.
2. **Freeze the benchmark boundary.** The starter evaluator, split definitions, row order, and submission checker are hashed and callable only through protected adapters.
3. **Separate model execution from authoritative evaluation.** Model workers output predictions and artifacts. A protected evaluator process computes official metrics.
4. **Use transactional state.** SQLite is the live source of truth. JSONL is an append-only, hash-chained event export, not the primary state store.
5. **Represent search as executable experiment nodes.** Each experiment has a parent commit, hypothesis, patch, rungs, artifacts, metrics, diagnosis, and terminal decision.
6. **Fail closed.** Missing or malformed artifacts, evaluator drift, firewall violations, protected-path edits, NaNs, and submission errors cannot promote a candidate.
7. **Keep scientific and infrastructure work distinct.** Repairing an import is a recovery event; it is not reported as a model improvement.
8. **Use one primary change per experiment.** This makes success and failure attributable and creates useful memory.
9. **Keep three independent trackers.** Search champion chooses promising parents; official best-ever chooses the final checkpoint; non-improvement streak decides convergence.
10. **Always have a valid fallback.** As soon as FM reproduction succeeds, produce and preserve a validated FM submission that finalization can use if every later branch fails.

## 4. Target repository layout

```text
src/rex/
  cli.py
  contracts.py
  control/
    coordinator.py
    state_machine.py
    search_policy.py
    budget.py
    recovery.py
    finalizer.py
  agents/
    proposer.py
    coder.py
    reflector.py
    memory.py
    provider.py
    schemas/
      proposal.json
      patch.json
      reflection.json
      lesson.json
    method_cards/
      ranking_loss.yaml
      temporal_features.yaml
      history.yaml
      multitask.yaml
      watchtime.yaml
      ensemble.yaml
  data/
    bootstrap.py
    manifest.py
    views.py
    temporal.py
    groups.py
    firewall.py
  evaluation/
    service.py
    official_adapter.py
    diagnostics.py
    submission.py
  execution/
    worker.py
    runner.py
    worktrees.py
    patch_guard.py
    telemetry.py
    artifacts.py
  models/
    base.py
    official_fm.py
    rank_fm.py
    tree_ranker.py
    history.py
    din.py
    multitask.py
    watchtime.py
    ensemble.py
    experimental/
  features/
    base.py
    temporal_aggregates.py
    repeat_exposure.py
    history_summaries.py
    experimental/
  losses/
    ranking.py
    experimental/
  store/
    schema.sql
    db.py
    repository.py
    event_log.py
    migrations/
  reporting/
    evidence_bundle.py
    report.py
tests/
  fixtures/
  unit/
  integration/
  security/
  fault_injection/
configs/
  frozen/
    starter_manifest.json
    benchmark.json
  run/
    default.yaml
  experiments/
  security/
    protected_paths.yaml
kuairand-starter-kit/             # unchanged organizer source
runs/<run_id>/                    # generated and gitignored
```

The root `agent/` and `training/` packages remain only until the new CLI passes the complete baseline vertical slice. They should not be maintained as a second implementation.

## 5. Frozen contracts

Freeze these contracts before platform and modeling work split between teammates.

### 5.1 Benchmark contract

`configs/frozen/benchmark.json` contains:

- dataset name and version;
- label `long_view`;
- train/validation/test date ranges and expected row counts;
- metric names and primary formula;
- random, popularity, FM, and oracle reference values;
- iteration, convergence, and wall-clock rules;
- exact final CSV header and split row-order requirements; and
- a denylist of post-exposure outcome columns.

`configs/frozen/starter_manifest.json` records SHA-256 hashes for at least:

- `kuairand-starter-kit/data.py`;
- `evaluate.py`;
- `submit.py`;
- `baseline.py`; and
- `baseline_scores.json`.

Verify hashes at run initialization, before every official evaluation, and during finalization.

### 5.2 Model worker contract

The only supported worker entrypoint is:

```bash
python -m rex.execution.worker --request request.json --result result.json
```

`RunRequest` fields:

- schema version;
- run, experiment, attempt, parent, and commit IDs;
- plugin import path;
- immutable configuration artifact ID;
- seed, rung, split view, and fold;
- capability-scoped input paths;
- output directory;
- absolute deadline and per-attempt CPU/GPU/RAM limits; and
- code, data-view, environment, and config hashes.

`RunResult` fields:

- typed status: `success`, `syntax`, `import`, `contract`, `timeout`, `oom`, `nan`, `crash`, `invalid_artifact`, or `interrupted`;
- exit code/signal and failure summary;
- checkpoint and prediction artifact references;
- stdout/stderr/telemetry artifact references;
- command, commit, config, data, and environment hashes; and
- wall time, CPU time, peak RSS, and GPU use where applicable.

The worker does not return authoritative metrics. This prevents experimental code from grading itself.

### 5.3 Model plugin contract

Every plugin implements:

```python
fit(train_view, config, seed, output_dir) -> ModelArtifact
predict(model_artifact, inference_view, output_dir) -> PredictionArtifact
save(model, path) -> None
load(path) -> Model
```

Every model consumes the same sanitized row view and writes the same artifact schema. No plugin imports the official evaluator or raw label vault.

### 5.4 Prediction artifact contract

Use a compressed NumPy artifact rather than introducing a Parquet dependency solely for predictions:

```text
predictions.npz
  row_id:int64
  user_id:int64
  video_id:int64
  score:float64
```

Its JSON manifest records split, row count, experiment, commit, config, model, data-view and file hashes, feature cutoff, seed, and schema version. The finalizer converts this artifact into the organizer CSV without reordering.

### 5.5 Experiment proposal contract

`ExperimentProposal` requires:

- experiment and parent IDs;
- one operator: `REPAIR`, `LOSS`, `FEATURE`, `SEQUENCE`, `AUX_TASK`, `MODEL_BLOCK`, `HYPERPARAMETER`, `ENSEMBLE`, or `ABANDON`;
- one falsifiable hypothesis and mechanism;
- exact files allowed to change;
- expected GAUC, nDCG@5, and segment effects;
- falsifier and prespecified promotion rule;
- leakage analysis;
- cheap/full rung plan;
- estimated seconds/resources; and
- rollback action.

A proposal that lacks a falsifier, touches a protected path, or changes multiple independent mechanisms is invalid before code generation.

### 5.6 Protected paths

The autonomous patch allowlist is restricted to:

- `src/rex/models/experimental/**`;
- `src/rex/features/experimental/**`;
- `src/rex/losses/experimental/**`;
- `configs/experiments/**`; and
- `tests/experiments/**`.

Everything else is protected, especially the starter kit, contracts, data manifest/views/firewall, evaluator, metric aggregation, state machine, budget, store/event writer, patch guard, telemetry, submission validation, and protected tests.

Reject symlinks, submodules, binary patches, path traversal, protected renames, and generated files outside the allowlist.

## 6. Data and test-label firewall

The bootstrap process is the only component allowed to read raw logs. It produces capability-scoped views:

| View | Visible data | Consumer |
|---|---|---|
| Training features | Inference-safe current-row fields plus earlier history | Model worker |
| Training targets | `long_view` and approved auxiliary targets | Model worker during fit only |
| Validation features | Inference-safe fields; no current outcomes | Model worker |
| Validation labels | Native `long_view` only | Protected evaluator |
| Test features | Inference-safe fields; no current outcomes | Model worker/finalizer |
| Test labels | Never materialized in research run | Nobody in development |
| Random exposure | Sanitized diagnostic view | Protected diagnostics only |

Mandatory rules:

- Current-row `long_view`, `play_time_ms`, click, like, follow, comment, forward, hate, dwell, and profile-enter fields are never inference features.
- Training auxiliary outcomes are passed through a separate target view, never mixed into feature columns.
- Historical features use rows strictly earlier than the candidate timestamp. Define deterministic order for equal timestamps before feature generation.
- Target rates are out-of-fold or computed from strictly earlier dates with smoothing, minimum support, and cutoff metadata.
- Validation histories end at the end of training; test histories may include validation only if organizer rules explicitly allow final train+validation retraining.
- The supplied monthly statistics file is disabled until every used column has proven point-in-time provenance.
- Unknown encoders and duration bins are fitted on the training period only.

Required poison tests:

1. Replace every forbidden current-row outcome with absurd values and prove predictions are byte-identical.
2. Shift future labels and prove all earlier point-in-time features are byte-identical.
3. Shuffle validation labels and prove only protected evaluator output changes.
4. Attempt to open the label vault from a worker and verify access denial.
5. Attempt to modify the evaluator or manifest from an experiment patch and verify rejection.

## 7. State, persistence, and lifecycle

### 7.1 SQLite schema

Enable WAL, foreign keys, and one coordinator writer. Use `BEGIN IMMEDIATE` for state transitions.

| Table | Purpose |
|---|---|
| `runs` | Run status, hashes, caps, deadline, counters, trackers, incumbent, stop reason |
| `process_sessions` | Host/PID, start/end, heartbeat, monotonic duration, console-log hash |
| `experiments` | Parent, operator, hypothesis, proposal, state, iteration, commit/config hashes |
| `transitions` | Ordered state transitions with unique idempotency keys |
| `attempts` | Rung/repair/status/command/timing/error/stdout/stderr for each execution |
| `metrics` | Split/fold/seed/evaluator hash/GAUC/nDCG/primary/CI/diagnostic artifact |
| `artifacts` | Immutable path, kind, hash, size, schema, experiment/attempt provenance |
| `promotions` | Previous/new official best and linked checkpoint/prediction/submission artifacts |
| `llm_calls` | Role/provider/model/request ID/schema validity/tokens/cost/artifact IDs |
| `resource_usage` | Wall/CPU/RSS/GPU/I/O/token samples by attempt/call |
| `lessons` | Global safety or branch-local scientific lessons with evidence IDs |
| `interventions` | Explicit actor/action/reason/timestamp/evidence |
| `event_outbox` | Ordered hash-chain source for append-only JSONL export |

Store canonical metric values at fixed precision. Use scaled integers for epsilon/convergence comparisons rather than binary float comparisons.

### 7.2 Run states

```text
INITIALIZING -> BASELINE_VERIFYING -> SEARCHING
-> FINALIZING -> COMPLETE
```

Terminal run states: `BASELINE_BLOCKED`, `BUDGET_EXHAUSTED`, and `FATAL`.

### 7.3 Experiment states

```text
PROPOSED -> WORKTREE_READY -> PATCHED -> STATIC_VALID
-> FIXTURE_VALID -> CHEAP_RUNNING -> CHEAP_COMPLETE
-> FULL_RESERVED -> FULL_RUNNING -> FULL_COMPLETE
-> DIAGNOSED -> CONFIRMING -> CONFIRMED
-> SUBMISSION_BUILDING -> SUBMISSION_VALID -> PROMOTED
```

Terminal experiment states: `REJECTED`, `ABANDONED`, and `FAILED_FINAL`.

Repair flow:

```text
FAILED_REPAIRABLE -> REPAIRING -> PATCHED
```

Every side effect is preceded by a committed transition/reservation and followed by a committed result. Unique idempotency keys make restart safe.

### 7.4 Promotion transaction

Before updating the official best-ever reference, require:

1. matching evaluator hash;
2. finite predictions and metrics;
3. exact row count and user/video alignment;
4. verified code/config/data/environment hashes;
5. valid checkpoint and prediction artifacts;
6. required fold/seed confirmation;
7. a test submission produced from the same commit/config/checkpoint; and
8. success from frozen `submit.py --check --split test`.

Only then insert the promotion row and update `runs.best_ever_experiment_id` in the same transaction.

## 8. Search, convergence, and budget semantics

### 8.1 Trackers

- **Search champion:** candidate with the best robust internal utility across temporal folds; used for parent selection.
- **Official best-ever:** highest fully valid official-validation primary; used for final checkpoint/submission. Any strictly higher score may update it, even when improvement is ≤epsilon. Exact ties retain the earlier candidate.
- **Non-improvement streak:** controls official convergence only.

For each counted scientific experiment with a finite official validation result:

```text
delta = candidate_primary - best_ever_primary_before_experiment
delta > 0.002  => streak = 0
delta <= 0.002 => streak += 1
streak == 3    => begin finalization
```

Failed or proxy-rejected experiments never fabricate metrics. Keep `attempt_count`, `hypothesis_count`, and `official_iteration_count` separately. Until organizers clarify cap semantics, enforce the strictest local cap: no more than 50 hypotheses and no more than 50 official evaluations.

### 8.2 Search policy

Maintain at most three active scientific branches:

1. current robust search champion;
2. one controlled mechanism variant; and
3. one diverse model family.

Rank parents using:

```text
utility = mean_shadow_primary
          - 0.5 * seed_std
          - instability_penalty
          - small_runtime_penalty
          + small_diversity_bonus
```

Suggested proposal mix: 70% improve a strong branch, 20% test a different mechanism, 10% repair or resolve uncertainty.

Successive-halving rungs:

1. static/import/protected-path checks;
2. synthetic ranking fixture;
3. one complete-user temporal shadow fold and seed;
4. all shadow folds;
5. official validation;
6. three-seed finalist confirmation; and
7. five-seed final blend if time permits.

Cheap row-random sampling is forbidden because it breaks group and temporal structure.

### 8.3 Recovery policy

| Failure | Recovery | Maximum |
|---|---|---:|
| Syntax/import/schema | Localized patch and rerun static/fixture checks | 2 repairs |
| Timeout | Reduce one scope control such as batch/history once | 1 retry |
| OOM | Reduce batch/history once | 1 retry |
| NaN/non-finite | Inspect loss scale/LR and patch locally | 2 repairs |
| API/transient provider | Exponential backoff and resume | Configured time budget |
| Data contract | Repair only if protected contract remains unchanged | 1 repair |
| Evaluator/firewall/protected-path violation | Immediate rejection | 0 |
| Invalid checkpoint/predictions/submission | Reject; do not promote | 0 |
| Metric regression | Scientific rejection, not repair | 0 |

### 8.4 Wall-clock policy

- Create an absolute epoch deadline at run initialization: start plus 21,600 seconds.
- Use monotonic clocks for elapsed durations and epoch integers for restart-safe deadline checks.
- Refuse new scientific proposals with fewer than 20 minutes remaining.
- At 20 minutes remaining, enter finalization using the official best-ever validated candidate.
- Kill active process groups at the deadline and retain the last valid incumbent.
- Track LLM tokens by call, CPU time, peak RSS, optional GPU time/memory, artifact sizes, and total wall time.

## 9. Work packages and dependency graph

Dependency graph:

```text
W0 Contracts/data freeze
  -> W1 Baseline-to-CSV vertical slice
     -> W2 Temporal evidence + leakage controls -----> W4 Model library
     -> W3 Runner/store/promotion core --------------> W5 Code evolution
W2 + W3 + W4 + W5
  -> W6 Scientific experiments
W3 + W5
  -> W7 Recovery + evidence reporting
W6 + W7
  -> W8 Rehearsals
  -> W9 Final run and release
```

### W0 — Contract and data freeze

Owner: Person A; Person B reviews. Estimate: 3–4 person-hours.

Files:

- `configs/frozen/benchmark.json`
- `configs/frozen/starter_manifest.json`
- `configs/security/protected_paths.yaml`
- `src/rex/contracts.py`
- `src/rex/data/manifest.py`
- `docs/task_contract.md`
- dependency lockfile and `.gitignore`

Tasks:

1. Download raw data outside tracked source and verify archive/file hashes where available.
2. Calculate starter hashes and expected data rows/dates/schema.
3. Consolidate the authoritative task contract; remove obsolete click/nDCG@10/Recall@50 and CPU-only claims from future runtime context.
4. Define Pydantic/JSON schema versions for requests, results, proposals, artifacts, metrics, and final bundle.
5. Freeze protected paths and create the first clean committed snapshot used by worktrees.

Acceptance gate G0:

- expected files, dates, row counts, schemas, and hashes match;
- no raw data or secrets are tracked;
- validation/test worker views exclude current outcomes;
- frozen contracts serialize/deserialize in unit tests; and
- implementation stops with a clear `BASELINE_BLOCKED` status on mismatch.

### W1 — Trusted baseline-to-CSV vertical slice

Owners: Person B owns loader/FM; Person A owns evaluator/submission. Estimate: 8–10 person-hours.

Files:

- `src/rex/data/bootstrap.py`
- `src/rex/data/views.py`
- `src/rex/data/firewall.py`
- `src/rex/models/base.py`
- `src/rex/models/official_fm.py`
- `src/rex/execution/worker.py`
- `src/rex/evaluation/official_adapter.py`
- `src/rex/evaluation/submission.py`
- `tests/integration/test_baseline_reproduction.py`
- `tests/fixtures/golden/`

Tasks:

1. Wrap the starter loader while preserving exact split row order.
2. Generate sanitized split views, evaluator-only validation labels, and row-alignment manifests.
3. Port/wrap random, popularity, and FM behind the model-worker contract.
4. Execute the frozen evaluator in a protected subprocess.
5. Save predictions, checkpoint, config, metrics, evaluator output, and hashes.
6. Produce a validation CSV and a test CSV without inspecting test labels/scores.
7. Run the real organizer checker against the generated files.
8. Preserve a validated FM checkpoint/submission as the run fallback.

Acceptance gate G1:

- validation random primary within ±0.001 of 0.4834;
- validation popularity primary within ±0.001 of 0.5807;
- five-seed FM validation mean within ±0.002 of 0.6016 and std ≤0.0015;
- prediction metrics equal the frozen evaluator output;
- test CSV contains exactly 170,588 data rows, finite scores, and the exact header;
- organizer `submit.py --check --split test` succeeds; and
- no candidate process can read validation/test labels outside its capability.

Nothing else proceeds if G1 is red.

### W2 — Temporal evidence and leakage controls

Owner: Person B; Person A reviews security tests. Estimate: 6–8 person-hours. Depends on W1.

Files:

- `src/rex/data/temporal.py`
- `src/rex/data/groups.py`
- `src/rex/features/base.py`
- `src/rex/evaluation/diagnostics.py`
- `tests/unit/test_temporal_features.py`
- `tests/security/test_label_firewall.py`

Tasks:

1. Create three rolling shadow folds inside the training period.
2. Implement complete-user sampling and deterministic same-timestamp ordering.
3. Implement reusable point-in-time aggregate primitives with cutoff metadata.
4. Add user-bootstrap confidence intervals and deterministic segment reports.
5. Cache sanitized split/fold artifacts by data/schema/feature hash.
6. Implement all poison and future-label tests.

Initial shadow folds:

- train 08–14, evaluate 15–16;
- train 08–16, evaluate 17–18; and
- train 08–18, evaluate 19–21.

Adjust only if EDA shows inadequate group/positive coverage; record the final boundaries in the frozen run config.

Acceptance gate G2:

- complete user groups and chronological order are preserved;
- earlier features are invariant to future-label changes;
- forbidden current outcomes cannot affect predictions;
- target statistics are OOF/earlier-time only; and
- diagnostics are deterministic across repeated runs.

No target-derived feature experiment is eligible before G2 passes.

### W3 — Transactional runner, store, and promotion core

Owner: Person A; Person B supplies the unchanged FM consumer. Estimate: 10–14 person-hours. Depends on W1; runs in parallel with W2.

Files:

- `src/rex/store/schema.sql`
- `src/rex/store/db.py`
- `src/rex/store/repository.py`
- `src/rex/store/event_log.py`
- `src/rex/control/state_machine.py`
- `src/rex/control/budget.py`
- `src/rex/execution/runner.py`
- `src/rex/execution/telemetry.py`
- `src/rex/execution/artifacts.py`
- `src/rex/evaluation/submission.py`

Tasks:

1. Create migrations and repository methods with foreign-key/idempotency tests.
2. Implement the run/experiment state machines and legal transition table.
3. Launch workers in process groups; persist stdout/stderr/telemetry before classification.
4. Kill the whole process group on timeout or absolute deadline.
5. Validate result schema, finite metrics/predictions, artifact hashes, and row alignment.
6. Implement monotonic/absolute budget accounting and the three trackers.
7. Make submission validation and promotion one fail-closed transaction.
8. Export the event outbox to append-only hash-chained JSONL.
9. Resume incomplete states idempotently after process interruption.

Integration checkpoint C1:

The unchanged FM plugin must traverse the real lifecycle through `PROMOTED` using only the frozen CLI/JSON contracts. A crash after any state transition must resume exactly once, never double-count, and never corrupt the incumbent.

### W4 — Minimum high-value model library

Owner: Person B; Person A owns contract tests. Estimate: 14–18 person-hours. Depends on W2 and W1.

Files:

- `src/rex/losses/ranking.py`
- `src/rex/models/rank_fm.py`
- `src/rex/models/tree_ranker.py`
- `src/rex/features/temporal_aggregates.py`
- `src/rex/features/repeat_exposure.py`
- `src/rex/features/history_summaries.py`
- `src/rex/models/history.py`
- `src/rex/models/ensemble.py`
- `configs/experiments/*.yaml`

Tasks:

1. Implement fixed-K same-user PairLogit sampling.
2. Implement delta-nDCG@5 pair weights and normalized hybrid BCE.
3. Adapt the FM to ranking losses without changing evaluation.
4. Add one pinned LightGBM LambdaRank branch with truncation near 5.
5. Add point-in-time item/author rates, trends, and backoff.
6. Add prior repeat count, last outcome, and time since user-item exposure.
7. Add user-author/tag/duration affinity and recency-decayed summaries.
8. Add simple candidate-history similarity features.
9. Implement per-user percentile-rank and standardized-score blending.

Blocking tests:

- no cross-user negative pair is possible;
- every positive receives fixed/capped negatives;
- all-positive/all-negative groups have explicit behavior;
- Lambda weights focus on swaps affecting nDCG@5;
- hybrid loss scales are finite and recorded;
- every feature exposes a cutoff/provenance record;
- every model emits the exact plugin artifact contract; and
- rank blending is invariant to per-user positive affine transformations.

Do not implement DIN, MTL, or watch-time heads in this work package. They require experimental evidence to unlock.

### W5 — Minimal real autonomous code evolution

Owner: Person A; Person B writes method cards and scientific validators. Estimate: 10–14 person-hours. Depends on W3 and a stable W4 plugin.

Files:

- `src/rex/agents/provider.py`
- `src/rex/agents/proposer.py`
- `src/rex/agents/coder.py`
- `src/rex/agents/reflector.py`
- `src/rex/agents/memory.py`
- `src/rex/agents/schemas/*.json`
- `src/rex/agents/method_cards/*.yaml`
- `src/rex/execution/worktrees.py`
- `src/rex/execution/patch_guard.py`
- `src/rex/control/coordinator.py`
- `src/rex/control/recovery.py`
- `src/rex/control/search_policy.py`

Tasks:

1. Implement configurable provider/model routing with lazy imports and usage/request capture.
2. Add a deterministic fake provider that needs no API key.
3. Force proposal, patch metadata, reflection, and lesson schemas.
4. Create clean worktrees from explicit parent commits under `codex/rex/<run>/<experiment>`.
5. Accept unified diffs only and enforce the patch allowlist.
6. Persist proposal and patch before execution.
7. Run static/import/protected/fixture tests before allocating training.
8. Implement bounded typed repair and rollback.
9. Store global safety lessons separately from branch-local scientific lessons.
10. Make reflection cite artifact/metric IDs and classify the hypothesis as supported, contradicted, or inconclusive.

Integration checkpoint C2:

- one agent-generated allowed loss/feature patch completes end to end;
- one evaluator-touching patch is rejected before execution;
- one syntax failure is repaired or rolled back with a bounded recovery record;
- the parent, diff, commit, command, predictions, metrics, and diagnosis are mutually linked; and
- the full flow works with the fake provider offline.

### W6 — Scientific experiment run

Owner: Person B is scientific owner; Person A observes operations without editing the run. Estimate: 10–16 hands-on person-hours plus machine time. Depends on W2–W5.

Every experiment preregisters its mechanism and falsifier. One complete-user shadow fold is the cheap rung. A candidate advances from the cheap rung when primary delta is at least 0.001 and neither component falls more than 0.002. Official validation is used only after all-shadow-fold evidence.

Run experiments in the exact order defined in section 10. Stop when official convergence, the iteration cap, or the wall-clock finalization reserve triggers.

Go/no-go G3 for a competitive finalist:

- three-seed official-valid mean ≥0.6056;
- at least two of three shadow folds positive;
- seed std ≤0.0015;
- neither GAUC nor nDCG@5 down more than 0.002;
- all leakage/provenance tests green; and
- linked valid checkpoint/prediction/submission artifacts exist.

If no candidate clears G3, select the highest fully confirmed valid candidate honestly. Do not react by introducing an untested large architecture.

### W7 — Fault injection and evidence reporting

Owner: Person A; Person B verifies scientific semantics. Estimate: 6–8 person-hours. Runs in parallel with W6 after W5.

Files:

- `tests/fault_injection/`
- `src/rex/reporting/evidence_bundle.py`
- `src/rex/reporting/report.py`
- `src/rex/control/finalizer.py`

Inject:

- syntax/import failure;
- child crash;
- hung child and descendant process;
- NaN loss/metric/prediction;
- OOM/resource limit where supported;
- missing/corrupt checkpoint;
- malformed/misaligned CSV;
- forbidden patch;
- invalid LLM schema and transient provider failure;
- process kill between every running transition; and
- database/event-export interruption.

Acceptance gate:

- each failure is typed and produces bounded recovery/terminal behavior;
- retries do not double-count scientific iterations;
- incumbent remains unchanged on every invalid path;
- resume is idempotent;
- SQLite and event export reconcile; and
- generated evidence links every report claim to source artifacts.

### W8 — Rehearsals

Owners: both. Estimate: 8–10 person-hours plus one six-hour dress run. Depends on W6 and W7.

R0 — fixture rehearsal:

- clean environment;
- fake provider;
- three iterations;
- forced restart;
- target under 15 minutes.

R1 — benchmark integration rehearsal:

- real data and FM;
- one passing candidate and one rejected candidate;
- invalid CSV injection;
- restart/resume;
- target under 60 minutes.

R2 — autonomy rehearsal:

- six to eight candidate transactions;
- at least one accepted agent-written patch;
- deliberately injected crash/provider failure;
- complete report/evidence bundle;
- target 60–90 minutes.

R3 — full dress rehearsal:

- fresh clone/clean environment;
- one documented launch command;
- no interactive stdin;
- no human code/config mutation during run;
- full six-hour budget and finalization reserve;
- final artifact and evidence bundle produced automatically.

Six-hour schedule:

| Time | Allocation |
|---|---|
| 0:00–0:36 | Manifest, EDA, random/popularity/FM self-check |
| 0:36–4:12 | Scientific search |
| 4:12–5:06 | Finalist confirmation |
| 5:06–5:42 | Freeze hypotheses; final checkpoint, test predictions, CSV, report |
| 5:42–6:00 | Safety reserve and bundle verification |

Gate G4: do not claim a fully autonomous demonstration unless R3 starts clean, finishes within caps, performs no manual mutations, survives recovery, promotes only validated candidates, and emits the complete evidence bundle.

### W9 — Final run and release

Owners: both. Estimate: 4–6 person-hours plus run time. Depends on G4.

Outputs:

- `runs/<run_id>/final/submission.csv`;
- linked checkpoint/config/commit;
- SQLite snapshot and `events.jsonl`;
- experiment graph and iteration table;
- validation metrics, fold/seed uncertainty, segment diagnostics;
- resource and intervention reports;
- environment/data/starter manifests;
- generated evidence index and report;
- bundle manifest hashing every file; and
- final public README/Devpost description based only on verified artifacts.

Run the organizer test checker when the candidate is generated and again after copying it into the final bundle. Record the final CSV and checkpoint hashes in the promotion/final bundle.

## 10. Scientific experiment curriculum

| ID | Parent | Change | Advance evidence | Stop condition |
|---|---|---|---|---|
| E00 | Organizer | Five-seed FM reproduction and fallback | G1 passes | Stop project if reference fails |
| E01 | E00 | Same-user fixed-K PairLogit FM | GAUC/primary shadow gain | Reject if multi-fold mean is not positive |
| E02 | E00 | LightGBM LambdaRank with inference-safe base/context fields | Diverse nonlinear branch | Reject if it only reproduces popularity ordering |
| E03 | E02 | Point-in-time item/author rates and trend | Broad primary gain | Reject on leakage test or date instability |
| E04 | E01 | Delta-nDCG@5 pair weighting | Top-five gain without material GAUC loss | Revert if GAUC falls >0.002 |
| E05 | E04 | Small normalized BCE stabilizer | Lower seed variance with stable ranking | Drop if it changes calibration only |
| E06 | E03 | Repeat count, last outcome, time since user-item exposure | Overall and repeated-pair gains | Do not keep segment-only gain that hurts overall |
| E07 | Best tree | User-author/tag/duration affinities | Personalization gain and lower popularity correlation | Reject cold-user collapse |
| E08 | Best tree | Recency-decayed summaries | Later-fold and active-user gain | Reject unstable decay across dates |
| E09 | Best branch | Simple candidate/history similarity | Reliable sequence headroom | Do not unlock DIN without gain |
| E10 | Best FM + tree/history | Per-user percentile-rank ensemble | Robust gain from prediction disagreement | Retain simpler single model if delta is noise |
| E11 | E09 only if gated | Small DIN attention over last 20/50 history | Gain beyond E09 on 2/3 folds | Stop neural history if capacity adds no value |
| E12 | E11 only if gated | Add tab/interface-conditioned click auxiliary | Long-view improvement without gradient conflict | Remove on negative transfer |
| E13 | Best neural only | Threshold-aware watch-time auxiliary | Duration-segment and primary improvement | Fall back to binary head if unstable |
| E14 | Best singles/blend | Three-seed confirmation; five for final blend if time | G3 | No new mechanism after this point |

Scientific rules:

- Use one feature family or objective change per experiment.
- Sample negatives from the same user only.
- Use fixed/capped negatives per positive to approximate GAUC weighting.
- Exclude all-positive/all-negative users from pairwise terms while retaining valid pointwise/auxiliary contributions.
- Normalize pairwise, Lambda, and BCE loss scales before applying coefficients.
- Tune ensemble weights only on shadow folds, then make one official-validation confirmation.
- Use within-user percentile ranks or standardized scores for blending.
- Log effective pairs per user, cutoff/provenance, popularity correlation, and incumbent disagreement.
- Do not run embedding-size sweeps, raw static-field expansion, large transformers, all-task MTL, or arbitrary hyperparameter searches.

## 11. Two-person delivery schedule

The estimates total roughly 79–108 person-hours, or four to six intensive elapsed days for two people plus unattended machine time.

| Day | Person A — platform | Person B — data/model | Shared gate |
|---|---|---|---|
| Day 1 | W0 contracts, evaluator/submission boundary | Data acquisition, loader, FM adapter | G0 then G1 |
| Day 2 | W3 store/state/runner | W2 folds/firewall/diagnostics | C1 and G2 |
| Day 3 | W3 promotion/resume/fault foundations | W4 ranking FM/tree/features | Plugin contract integration |
| Day 4 | W5 agent/worktrees/patch guard | W4 history/ensemble/method cards | C2 |
| Day 5 | W7 evidence and fault injection | W6 scientific experiments | G3 candidate |
| Day 6 | W8 full rehearsals/finalizer | Final confirmations/ablation review | G4 and W9 |

Operating rules:

- Person A owns frozen control-plane code; Person B reviews contract changes.
- Person B owns model/feature code; Person A reviews contract compliance; both review leakage-sensitive features.
- Freeze CLI/JSON contracts before parallel work.
- Integrate through contracts, never undocumented direct imports across ownership boundaries.
- Run the FM lifecycle smoke test on main daily.
- No feature branch remains unintegrated for more than four working hours.
- At joins, merge contracts/tests first, producer second, consumer third.
- During autonomous/dress runs, neither person edits code or configuration. Any operational intervention is an explicit event.

## 12. Blocking test matrix

| Area | Required tests |
|---|---|
| Benchmark | Exact dates, rows, schema, hashes, deterministic row order |
| Metrics | Hand-computed GAUC/nDCG, ties, all-positive, all-negative, empty/invalid inputs, official parity |
| Firewall | Capability denial, forbidden-column poison, future-label invariance, evaluator protected-path rejection |
| Features | OOF rates, cutoff metadata, equal-timestamp ordering, complete-group sampling, repeated-pair context |
| Losses | Same-user pairs, fixed negatives, all-pos/all-neg behavior, finite gradients, Lambda cutoff and normalization |
| Plugins | Common request/result/artifact contract, deterministic seeds, save/load/predict parity |
| Store | Migration, foreign keys, legal/illegal transitions, idempotency, crash recovery, event reconciliation |
| Budget | Fake-clock 50/6h caps, epsilon boundary, declining scores, early peak, exact tie, finalization reserve |
| Runner | Process-tree timeout, stdout/stderr retention, NaN/Inf/missing artifact classification, telemetry |
| Patch safety | Allowlist, path traversal, symlink, binary diff, protected rename/evaluator edit |
| LLM | Schema failures, retry/backoff, token reconciliation, fake-provider no-key execution |
| Promotion | Validation before incumbent update; invalid path leaves incumbent unchanged; linked artifacts |
| Submission | Header, count, row IDs, repeated pairs, user/video alignment, finite scores, real organizer checker |
| End to end | Offline fixture, real FM, one agent-written patch, forced restart, clean-environment replay |

Performance targets for the test suite:

- static/unit/fixture suite under two minutes;
- daily FM lifecycle smoke under five minutes after cached data setup;
- five-seed baseline reproduction under fifteen minutes on the target machine if measured starter runtime permits; and
- no test depends on hidden-test scoring.

## 13. Evidence bundle and judging deliverables

Each completed iteration must retain:

- proposal and falsifiable hypothesis;
- exact unified diff and resulting commit;
- parent commit and branch;
- configuration and command;
- environment, data-view, evaluator, and code hashes;
- stdout/stderr and typed errors;
- checkpoint and predictions;
- raw evaluator output and derived diagnostics;
- reflection with evidence IDs;
- recovery attempts;
- LLM tokens/cost and compute telemetry; and
- terminal promotion/rejection decision.

`runs/<run_id>/final/` contains:

```text
submission.csv
checkpoint/
experiment_manifest.json
evidence_index.json
events.jsonl
run.sqlite
resource_report.json
intervention_report.json
environment_manifest.json
data_manifest.json
starter_manifest.json
experiment_graph.json
report.md
bundle_manifest.json
```

The generated report must include:

- random/popularity/FM reproduction;
- validation-best GAUC, nDCG@5, primary, and absolute deltas;
- multi-fold/multi-seed uncertainty;
- readable experiment tree and iteration table;
- at least one successful autonomous code change;
- rejected experiments and what they taught the agent;
- fault injection and recovery evidence;
- total iterations, wall clock, tokens, CPU/GPU/RAM, and interventions;
- final artifact hashes and reproduction command;
- limitations and unattempted branches; and
- team contributions.

Use the wording “tamper-evident evidence of zero recorded intervention” unless an external trusted runner truly enforces absence of human activity.

## 14. Explicitly deferred scope

Defer until KuaiRand-Pure G4 is green:

- KuaiRand-1K and KuaiRand-27K bonus runs;
- full MCTS or a broad multi-agent topology;
- dashboard/live UI;
- Kubernetes or generalized distributed execution;
- generic model DSL;
- full 12-task MMoE/PLE/PCGrad;
- large transformers/SIM/long-history systems;
- DCN-V2/DeepFM capacity sweeps;
- semantic captions/categories or pretrained content features;
- random-exposure training or propensity correction;
- moving/renaming the organizer kit;
- arbitrary agent edits to platform infrastructure;
- broad hyperparameter sweeps;
- raw monthly statistics without provenance;
- FM dimension sweeps and generic class weighting;
- test-label scoring; and
- train+validation final retraining without written approval.

## 15. First implementation sequence

Execute these tasks in order when implementation begins:

1. Obtain the data and verify exact files, dates, row counts, and hashes.
2. Commit a clean starting snapshot; record starter file hashes.
3. Create the dependency lock and new `src/rex` package skeleton.
4. Write frozen Pydantic/JSON contracts and serialization tests.
5. Implement bootstrap, sanitized views, label vault, and row-alignment manifest.
6. Wrap the official evaluator and organizer submission checker in protected subprocesses.
7. Implement the common worker CLI and artifact manager.
8. Port the organizer FM and reproduce random/popularity/FM references.
9. Preserve the validated FM fallback submission.
10. Implement shadow folds, group sampling, diagnostics, and poison tests.
11. Implement SQLite migrations, repositories, state machine, and event outbox.
12. Implement process-group runner, telemetry, deadline, and typed failures.
13. Implement atomic submission validation and promotion.
14. Prove resume/idempotency with the FM lifecycle.
15. Implement ranking loss, rank-FM, tree ranker, temporal/repeat/history features, and ensemble.
16. Implement provider abstraction, fake provider, agent schemas, and method cards.
17. Implement git worktrees, unified-diff guard, proposer/coder/reflector, bounded recovery, and memory.
18. Pass C2 with one accepted agent patch, one protected rejection, and one repaired syntax failure.
19. Run E00–E10; unlock E11–E13 only through their evidence gates.
20. Complete fault injection, R0–R3 rehearsals, final judged run, and evidence-based README/Devpost package.

## 16. Final definition of done

- [ ] G0: benchmark contract, data, hashes, and sanitized roles verified.
- [ ] G1: random/popularity/five-seed FM references reproduced and FM fallback CSV validated.
- [ ] G2: temporal folds, point-in-time features, and all firewall poison tests pass.
- [ ] C1: real FM completes the transactional lifecycle and resumes idempotently.
- [ ] Model library passes loss, feature, plugin, and ensemble contract tests.
- [ ] C2: the agent proposes and writes at least one accepted improvement end to end.
- [ ] Protected edits and malformed artifacts fail closed without incumbent mutation.
- [ ] Crash, timeout, NaN, API failure, invalid submission, and restart recover within bounded policy.
- [ ] Every scientific iteration has hypothesis, parent, diff, commit, config, metrics, artifacts, diagnosis, and recovery evidence.
- [ ] Official best-ever, search champion, and convergence streak behave correctly at all stop conditions.
- [ ] Competitive finalist is confirmed across folds and seeds, or the strongest confirmed fallback is selected honestly.
- [ ] Final checkpoint, predictions, CSV, commit, config, metrics, and experiment row are mutually linked.
- [ ] Final CSV passes the frozen organizer checker after final packaging.
- [ ] Resource and intervention totals reconcile with event-level records.
- [ ] R3 completes from a clean environment with one command and without unlogged mutation.
- [ ] Final evidence bundle hashes and replay command verify.
- [ ] Public README and Devpost description contain no stale task facts or claims stronger than the artifacts prove.
