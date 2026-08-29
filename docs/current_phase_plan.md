# Current Phase Implementation Plan

Date: 2026-08-30

Status: implemented and verified

## Scope

This phase builds and verifies the autonomous control plane without performing
scientific model search. All end-to-end execution uses generated fixture data.

Explicit exclusions:

- no ranking-loss, history-feature, LightGBM, or ensemble comparison;
- no three-fold or three-seed winning-model confirmation;
- no six-hour dress rehearsal;
- no final test prediction or competition submission.

## 1. Dual structured LLM providers

Implement one `StructuredProvider` contract for proposal, patch, and diagnosis
roles.

Codex CLI route:

- call locally authenticated `codex exec`;
- use ephemeral, read-only, non-interactive execution;
- require a role-specific JSON Schema and a final response file;
- kill the process group on timeout;
- record bounded stdout/stderr, request ID, tokens, model, and elapsed time.

OpenAI API route:

- read `OPENAI_API_KEY` and the model from environment variables only;
- use the Responses API with strict structured output, `store=false`, and no tools;
- disable SDK retries so the control plane owns retry accounting;
- enforce call and token ceilings and rehydrate durable successful usage on resume;
- redact credentials and authorization headers from persisted errors.

Routing:

- support `codex_cli`, `openai_api`, `fixed`, and `auto` modes;
- retry a provider no more than two times after its first attempt;
- keep paid API fallback disabled unless explicitly authorized;
- persist fallback/degradation evidence without starting a new hypothesis.

Acceptance: provider schema, timeout, error classification, redaction, retry,
budget, and fallback tests all pass without live credentials.

## 2. Durable ownership and exactly-once persistence

Extend SQLite with:

- durable experiment worktree/branch/commit/config provenance;
- process sessions with PID, host, heartbeat, staleness, and exit reason;
- pre-execution attempt reservations;
- immutable artifact ownership links;
- exactly-once LLM, attempt, metric, artifact, resource, and event writes;
- atomic, replay-safe JSONL event export.

Resume behavior:

- a live owner blocks a second coordinator;
- a stale owner is explicitly closed by takeover;
- the new coordinator dispatches from the persisted experiment state;
- repeated result ingestion must either be identical or fail closed.

Acceptance: stale takeover, duplicate replay, interrupted event export, and
repair-number tests preserve all prior state and incumbent fields.

## 3. Isolated model workspaces and complete bundles

For every candidate:

- create a disposable Git repository containing only source, fixture tests, and
  frozen firewall contracts;
- create a candidate worktree from an exact parent commit;
- accept diffs only in the experimental allowlist;
- run path, AST/capability, compile, and fixture gates before execution;
- require fixture LLM patches to change only the approved finite numeric
  `DEFAULT_BIAS` assignment;
- commit the accepted diff and execute the worker from that clean worktree;
- reject dirty worktrees and commit mismatches.

Training emits a complete model bundle manifest with member hashes, plugin,
commit, config, data, and feature-schema provenance. Prediction loads the bundle
without mutating the experiment config.

Acceptance: corrupt or incomplete bundles, missing predictions, NaN values, and
wrong workspaces cannot be interpreted as successful attempts.

## 4. Connected fixture-only autopilot

Expose one command that performs:

```text
proposal -> patch -> safety gates -> cheap fixture run -> conditional full
fixture run -> metric record -> diagnosis -> reject/close -> next proposal
```

The loop:

- generates its own tiny feature and target arrays;
- never opens competition data or the official evaluator;
- counts one proposal as one hypothesis transaction;
- records worktree, LLM, attempt, metric, resource, and evidence artifacts;
- can resume from durable experiment states after coordinator interruption;
- never enters confirmation, submission, or promotion states;
- reports all production-science and final-submission flags as false.

Acceptance: a normal run completes three fixture transactions from one command,
including one cheap rejection and two evidence-bound diagnoses, with a valid
event chain. The connected fault rehearsal adds a fourth transaction that must
reach `FAILED_FINAL` after its initial attempt plus two repairs.

## 5. Optional LightGBM model path

Pin LightGBM and its scikit-learn runtime in the `tree` optional dependency.
Harden the LambdaRank plugin with deterministic seeds/threads, stable group
ordering, categorical declarations, date offsets, finite checks, and complete
bundle metadata.

The current phase runs only a tiny synthetic doctor. It does not compare
LightGBM with FM or make a model-quality claim.

Acceptance: the doctor trains, predicts, saves, reloads, and validates a finite
synthetic bundle deterministically.

## 6. Production-style fault matrix

Intentionally test:

- worker crash, timeout, descendant cleanup, simulated OOM, and interruption;
- NaN predictions/loss and missing, malformed, or corrupt result artifacts;
- LLM timeout, API interruption, invalid schema, retry/fallback, and secret redaction;
- dirty, wrong-commit, out-of-root, or protected worktree changes;
- stale coordinator takeover and duplicate process ownership;
- interrupted database-backed event export and exactly-once replay;
- repair numbers zero, one, and two with no third repair;
- incumbent invariance after every failed candidate;
- malformed or incorrectly aligned organizer CSV validation.

Acceptance: every failure is typed, no broken artifact promotes, repair count is
bounded at two, prior best fields remain unchanged, and recovery evidence is
durable.

## 7. Short integration rehearsal and handoff

Run a sub-15-minute generated-fixture rehearsal that:

- starts with one command;
- injects one retryable provider interruption;
- injects a one-shot worker NaN and verifies the first repair succeeds;
- injects a persistent worker NaN and verifies attempts zero, one, and two are
  recorded before `FAILED_FINAL`, with no third repair;
- proves a protected patch is rejected;
- completes the four-transaction fault run without stranding the coordinator;
- verifies the hash-chained event export;
- asserts no experiment is promoted.

Publish setup instructions for fixed, Codex CLI, OpenAI API, and explicit paid
fallback modes. Run the complete test suite and static checks. Do not commit
until the user requests a commit.
