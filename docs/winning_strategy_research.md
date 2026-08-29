# Winning Strategy Research: Autonomous ML Research Agent for KuaiRand-Pure

Date: 2026-08-29

This document combines five sources of truth:

1. the challenge text supplied by the organizers (last updated 2026-08-27);
2. the organizer-provided `kuairand-starter-kit/` now present in this repository;
3. the team's existing Google Doc and `docs/spec.md`;
4. a full audit of the current implementation; and
5. primary research and official documentation on autonomous ML agents, learning-to-rank, sequential recommendation, multi-task learning, and watch-time modeling.

The central recommendation is to build a **metric-aligned, evidence-audited, code-evolving research agent**, not merely an LLM that selects hyperparameters from a fixed menu.

## 1. Executive verdict

The current work has a good product story and a useful first layer of reliability plumbing. Its strongest ideas are:

- specializing the agent for recommender systems;
- separating the convergence tracker from the best-checkpoint tracker;
- forcing evidence-backed reflection;
- isolating training in a timed subprocess;
- logging autonomy and resource consumption; and
- validating submissions before final promotion.

However, it is not yet close to satisfying the challenge end to end. The current agent does not reproduce the baseline, inspect the data, generate or modify model code, perform tree search, train a real model, calculate the official metrics, produce a submission, retry repairs, or record code diffs. Several implemented gates also do not enforce what their documentation claims.

The existing competitive thesis—"other teams will ignore the non-model 65%"—is directionally useful but too complacent. Technical quality is still the gate: Feasibility is scored only after beating the baseline, and the strongest current ML agents already use structured search, ablations, memory, and ensembling. The safest winning posture is:

> Beat the FM baseline by a defensible margin using ranking-aligned objectives, leakage-safe behavioral history, and complementary ensembles; then make the entire discovery process unusually autonomous, reproducible, and easy for judges to audit.

## 2. Authoritative task contract

The starter kit and its immutable `evaluate.py` should override stale or contradictory prose elsewhere.

| Item | Authoritative value |
|---|---|
| Required dataset | KuaiRand-Pure |
| Task | Rank each user's logged impressions; not full-catalog retrieval |
| Label | `long_view` |
| Metrics | GAUC and nDCG@5 |
| Primary | `(GAUC + nDCG@5) / 2` |
| Train dates | 2022-04-08 through 2022-04-21; 1,141,112 rows |
| Validation dates | 2022-04-22 through 2022-04-28; 124,909 rows |
| Test dates | 2022-04-29 through 2022-05-08; 170,588 rows |
| Official FM validation | GAUC 0.6674 / nDCG@5 0.5357 / primary 0.6016 |
| Official FM hidden test | GAUC 0.6610 / nDCG@5 0.5282 / primary 0.5946 |
| Random test reference | primary 0.4753 |
| Popularity test reference | primary 0.5715 |
| Test oracle ceiling | primary 0.8645 |
| Run limits | 50 iterations, 6 hours, convergence epsilon 0.002 / patience 3 |
| Submission | `row_id,user_id,video_id,score`, exactly aligned to split rows |

Two contradictions must be removed from team-facing material:

- The older section of the Google Doc says click / nDCG@10 / Recall@50. That is obsolete.
- The supplied challenge text contains one stale constraint row with those same old values, while the later dataset section and starter kit pin `long_view`, GAUC, and nDCG@5. The starter kit is executable and therefore decisive.

The challenge permits PyTorch, RecBole, TorchRec, LightGBM, pretrained weights, papers, and public solutions. "The baseline is NumPy/CPU" does **not** mean our solution must be NumPy-only or CPU-only. GPU use is allowed; it must merely be reported.

## 3. What the starter kit tells us strategically

The organizers have already tested two tempting but low-value directions:

- Adding the CWM-style static fields to the pointwise FM did not improve primary.
- Sweeping FM embedding dimensions 8/16/32 did not improve primary.

That is unusually valuable prior knowledge. The experiment portfolio should not begin with generic feature-count or capacity tuning. The starter kit explicitly identifies the likely headroom:

1. ranking-aligned pairwise or listwise objectives;
2. user behavior history;
3. auxiliary feedback / multi-task learning;
4. watch-time and duration-bias modeling;
5. richer architectures only after the above;
6. temporal drift; and
7. random-exposure data as an unbiased diagnostic.

There are also metric-specific invariants the agent should know:

- Any feature or score term constant within a user cannot change either metric.
- Any strictly monotonic transformation applied independently within a user leaves both metrics unchanged.
- All-positive and all-negative users carry no GAUC ranking signal.
- All-negative users always contribute zero nDCG, so the nDCG ceiling is below one.
- Because GAUC weights each user by positive count, uniformly sampling a small number of negatives per positive is a natural approximation to its pairwise weighting.
- Reusing the one official validation window for 50 adaptive decisions can cause validation overfitting even without test leakage.

These invariants are more valuable than generic class-imbalance advice. For example, class weighting may change calibration without changing order; a ranking loss directly targets the property that is scored.

### 3.1 Dataset-specific deductions from the official schema

The exact label definition changes the modeling strategy. In the official KuaiRand documentation,

```text
tau(video) = min(duration_ms, 18,000)
long_view = 1[play_time_ms >= tau(video)]
```

For videos at most 18 seconds long, a positive means complete play; for longer videos, it means at least 18 seconds watched. This yields several concrete decisions:

- Add continuous duration, `tau`, short/long indicator, duration bucket, and user-by-duration preference. Do not rely only on a duration ID or embedding.
- Treat the primary problem as estimating `P(watch_time >= tau | user, item, context)`, while still optimizing within-user order.
- Current-row `play_time_ms`, other feedback labels, profile dwell, and comment dwell are post-exposure outcomes. They may be auxiliary targets during training but must never be candidate-row input features at inference.
- Completion is right-censored by short video length. A distributional or censored watch-time auxiliary can model more information than a binary head without redefining the official label.

The meaning of `is_click` also depends on the interface. In the two-column interface it is a literal click; in the single-column interface it is a valid-play threshold. A one-head click auxiliary mixes two behavioral mechanisms and can create negative transfer. Condition the head on `tab`/interface or use separate heads when both interfaces are present.

The starter test set contains repeated `(user_id, video_id)` pairs—about 3.06% of rows, with some pairs appearing up to 12 times. A static user/item model assigns identical scores to those rows. Timestamp, tab, previous-exposure count, last outcome, and time-since-last-exposure can break these ties. This is more challenge-specific and promising than another FM embedding-size sweep.

The official item-statistics files aggregate behavior by date and scenario over a month. They cannot be joined naively: any statistic whose window reaches beyond the prediction timestamp leaks future behavior. Rebuild statistics from permitted earlier rows or prove each supplied column's cutoff before use.

## 4. Existing repository audit

### 4.1 What is good and should be retained

| Component | Verdict | Why |
|---|---|---|
| `docs/spec.md` task facts | Retain with corrections | Correct label, metrics, references, and dual-tracker intent |
| `agent/trick_menu.yaml` | Retain as knowledge, not as the search engine | Good mechanism/evidence vocabulary |
| Structured reflection schema | Retain and expand | Makes diagnosis judge-readable and machine-checkable |
| Subprocess timeout wrapper | Retain and harden | Correct process-isolation boundary |
| Atomic state write | Retain | Prevents truncated state after crashes |
| Best-ever checkpoint separate from convergence | Retain | Matches the final checkpoint rule |
| Submission validator concept | Retain and make blocking | Essential final-artifact gate |
| Resource counters | Retain and make complete | Required deliverable |

### 4.2 What is incomplete or incorrect

| Requirement | Current evidence | Status / required change |
|---|---|---|
| Reproduce official baseline | No integration; dataset absent | Missing. Import and wrap the starter FM first. |
| EDA | `eda_findings = {}` | Missing. Generate immutable dataset profile and diagnostic artifacts. |
| Agent writes code | Move is stored as free text and passed as config | Missing. Add isolated patch generation, application, tests, and rollback. |
| Tree search | One linear loop | Missing. Persist parent/child experiment graph and search policy. |
| Real training | Stub returns `None` metrics | Missing. Replace `training/train.py` contract implementation. |
| Data loader | Imports nonexistent `training/data/loader.py` | Missing. Adapt starter `data.py` behind a split-safe interface. |
| Official evaluation | No call to starter `evaluate.py` | Missing. Treat its hash as immutable. |
| Code-diff logging | No diff captured | Missing and explicitly required by the challenge. |
| Cheap-test gate | Every successful process is "promising" | Incorrect. Use a calibrated threshold and uncertainty. |
| Submission gate | Best state updated before validation; missing CSV skips validation | Incorrect. Validation must succeed before promotion. |
| Official validator discovery | Code expects `kuairand_starter_kit`, actual folder uses `kuairand-starter-kit` | Broken path. |
| Convergence | Uses range of last three scores | Incorrect. Three declining scores never converge. Track consecutive epsilon-non-improvements against the incumbent. |
| Wall-clock | UTC timestamp parsed as local time | Broken in UTC+8; a fresh run appears about eight hours old. |
| Restart config | YAML threshold is ignored in favor of a constant | Inconsistent. Use one configuration source. |
| Recovery | No bounded code-repair retry or rollback | Missing. |
| Failure typing | NaN is inferred only from failing stderr | Incomplete. Validate finite loss, metrics, predictions, and artifacts. |
| Resume | Prior experiment summaries are not reloaded | Incomplete. Long-horizon context is lost on restart. |
| Dry run | Requires API key and eager Anthropic import | Broken as a true offline smoke test. |
| Reproducibility | Fixed seed only; no environment/data/code hashes | Incomplete. |
| Test integrity | Starter exposes test labels and current design has no firewall | High risk. Prevent the research agent from reading test labels. |
| Tests | No test suite | Missing. Add unit, integration, recovery, and fault-injection tests. |

### 4.3 Assessment of the Google Doc ideas

Good ideas to keep:

- full-loop framing rather than model-only AutoML;
- recsys-specific diagnostics;
- cheap-before-expensive promotion;
- typed failures, timeouts, checkpoints, and seed logging;
- explicit submission validation;
- results anchored to random, popularity, FM, and oracle; and
- a readable run narrative for judges.

Ideas to revise:

- A static trick menu alone is not a meaningful tree search. It should be a retrieved prior used by code-editing operators.
- `5% data / 1 epoch` is unsafe for grouped and sequential ranking. Sampling must preserve complete user groups, time order, label mix, and cold-start segments.
- DeepFM/MMoE should not be the assumed winning model. The starter kit says the first test should be the loss, not capacity.
- Class weighting and generic negative sampling are lower-priority than groupwise ranking objectives.
- "Checkpoint every iteration" should mean model artifact + code commit + config + data fingerprint, not only a model file.
- A heartbeat gap is not proof of a human intervention. It is only evidence of downtime. Manual actions need a separate auditable channel.
- A self-authored mutable JSON file does not prove zero intervention. Run the final demonstration in a clean container and preserve append-only process logs, git commits, hashes, and one launch command.
- The document currently duplicates a long outdated version and a corrected version. Consolidate it before using it as agent context, or the LLM will receive contradictory task definitions.

### 4.4 Keep, rewrite, or remove: a decisive file-level verdict

Do not optimize for preserving sunk work. The current runtime is small enough that replacing its center is safer than progressively adding behavior to contracts that are already wrong.

| Current artifact | Decision | Replacement action |
|---|---|---|
| `kuairand-starter-kit/` | Keep immutable | Vendor it under `vendor/organizer/`, record hashes, and prohibit agent writes. |
| `agent/run_training_subprocess.py` | Keep the boundary, rewrite internals | Preserve process isolation and timeout; add process-group termination, typed result schema, stdout/stderr artifacts, finite-value checks, memory tracking, and retry policy. |
| `agent/submission_validator.py` | Rewrite | Resolve the real hyphenated path, verify exact row alignment against the split, fail closed, and validate before best-candidate promotion. |
| `agent/state.py` | Replace | Use transactional SQLite for experiments and append-only JSONL for events. Keep only the concept of atomic durable state and a separate incumbent. |
| `agent/orchestrator.py` | Replace completely | The existing linear config loop cannot become a code-evolving search engine without changing its fundamental data model and lifecycle. |
| `agent/llm_client.py` | Refactor behind an interface | Make provider/model routing configurable, lazily import clients, capture request IDs and usage, validate structured responses, and support offline deterministic tests. |
| `agent/trick_menu.yaml` | Replace with versioned method cards | Retain useful prose, but give every method prerequisites, forbidden features, expected segment effects, cost, falsifier, compatible parents, and citations. |
| `agent/schemas/reflect_schema.json` | Replace | Separate proposal, diagnosis, repair, and reusable-lesson schemas. Reflection must cite experiment artifacts and uncertainty, not only scalar metrics. |
| `training/train.py` | Replace completely | Implement a stable CLI around real model plugins, prediction artifacts, official evaluation, checkpointing, and resource telemetry. |
| `training/models/` and `training/data/` references | Create for real | They are documented but absent. Implement models and point-in-time data contracts before autonomous mutation. |
| `configs/budget.yaml` | Rewrite | Correct compute assumptions; include per-rung time/memory limits, retry budgets, seed budgets, model routing, and the organizer's caps in one source of truth. |
| `scripts/run_agent.sh` | Rewrite | Use an explicit environment entrypoint, no unconditional API-key requirement for dry-run, run manifest creation, signal handling, and resumable execution. |
| `README.md` | Rewrite after the system works | It currently claims CPU-only constraints, tested plumbing, and “structural proof” that the code does not establish. README claims must be generated from verified artifacts. |
| `docs/spec.md` | Archive as an early draft | It contains corrected task facts but outdated strategic assumptions. Replace it with a short authoritative contract linking to this blueprint. |
| `notebooks/` | Optional, outside the critical path | Use scripts to generate reproducible EDA artifacts; notebooks may consume them for presentation but must not define production logic. |
| `logs/` | Generated, never mutable source truth | Store each run under a unique manifest ID; preserve final evidence bundles as immutable release artifacts. |
| `requirements.txt` | Replace with a pinned environment | Add PyTorch and one tree-ranker stack only when used; lock exact versions and record hardware/runtime metadata. |

Three statements in the current public-facing material should be removed immediately: that the task is CPU-/NumPy-only, that the existing plumbing is tested end to end, and that heartbeats structurally prove zero intervention. All three are stronger than the evidence.

## 5. Lessons from current autonomous-ML systems

The strongest pattern across current systems is not "more reflection." It is **search over executable artifacts under an objective evaluator**.

- [AIDE](https://arxiv.org/abs/2502.13138) represents candidate solutions as a tree of code and uses draft, debug, and improve operators.
- [AI Scientist-v2](https://arxiv.org/abs/2504.08066) uses progressive agentic tree search managed by an experiment manager rather than relying on human-authored templates.
- [AI Research Agents for Machine Learning](https://arxiv.org/abs/2507.02554) shows that search policy and operator design must be chosen jointly; its best pairing improved MLE-bench Lite medal rate from 39.6% to 47.7%.
- [MLE-STAR](https://research.google/pubs/mle-star-machine-learning-engineering-agent-via-search-and-targeted-refinement/) combines web-grounded initialization, code-block-level refinement, ablation-guided targeting, and ensembling; it reports medals on 64% of MLE-bench Lite competitions.
- [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) combines a fast model for broad proposals, a strong model for depth, automated evaluators, and an evolutionary program database.
- [KAPSO](https://arxiv.org/abs/2601.21526) adds git-native isolated experiments, structured external knowledge, and episodic memory distilled from traces to address lost state and repeated failures.
- [MARS](https://arxiv.org/abs/2602.02660) adds budget-aware MCTS, modular Design-Decompose-Implement construction, and comparative memory. Its authors report that 63% of utilized lessons came from cross-branch transfer. This directly argues against the Google Doc's proposal to cut all cross-branch learning; a small shared lesson store is likely worth the effort even if complex branch-to-branch messaging is not.
- [MLEvolve](https://arxiv.org/abs/2606.06473) uses progressive graph search, retrospective memory, and separate strategic-planning and code-generation modes. The directly transferable idea is not a large multi-agent topology; it is allowing promising branches to reference useful discoveries from other branches while gradually shifting from exploration toward exploitation.
- A 2026 [controlled study of memory in MLE agents](https://aclanthology.org/2026.findings-acl.525/) finds that memory improves reliability in sequential agents but can reduce diversity in tree search. REX should therefore retrieve failure-prevention lessons globally while keeping scientific hypotheses and branch-specific design context local unless cross-branch evidence is explicitly relevant.
- [ScientistOne](https://arxiv.org/abs/2605.26340) treats score verification, specification compliance, reference validity, and method-code alignment as auditable evidence chains. This is an especially good match for a hackathon where judges inspect run logs and where a polished narrative can otherwise drift away from the code that actually ran.
- [MLE-bench](https://openai.com/index/mle-bench/) is a warning against overconfidence: the original best AIDE-based configuration only reached bronze level on 16.9% of 75 competitions. A generic loop is not enough.

The resulting design principle for this hackathon is:

> Use a small, reliable coordinator with explicit operators and durable state. Let the LLM propose hypotheses and localized patches; let deterministic evaluators, tests, ablations, and search policy decide what survives.

The architecture should therefore include a lightweight **chain of evidence**:

```text
paper/source claim
  -> structured hypothesis
  -> exact code diff + config
  -> executable command + environment/data hashes
  -> raw evaluator output
  -> derived metric table
  -> reflection claim
  -> final report claim
```

Every arrow should be machine-resolvable. This gives the judges a stronger autonomy and scientific-integrity story than prose-only reflection.

## 6. Proposed winning agent: REX (Recommender Experiment eXplorer)

The name is optional; the architecture is the important part.

### 6.1 Immutable control plane

Before autonomous experimentation begins:

1. Hash and lock the official `evaluate.py`, split definitions, and submission schema.
2. Build a data-access layer with roles:
   - `train`: labels readable;
   - `validation`: labels accessible only to evaluator process;
   - `test`: labels inaccessible to the research agent and training code;
   - `random_exposure`: diagnostics only unless organizers explicitly approve training use.
3. Run self-check rungs: random, popularity, and official FM.
4. Fail closed if row counts, hashes, reference metrics, or split dates disagree.

This is both a scientific-integrity guard and a strong presentation point because the public KuaiRand release makes accidental test-label access technically possible.

### 6.2 Experiment object

Every candidate should be a structured record, not free text:

```json
{
  "experiment_id": "exp_017",
  "parent_id": "exp_011",
  "operator": "loss_change",
  "hypothesis": "Within-user pairwise loss should improve GAUC without sacrificing nDCG@5",
  "mechanism": "Optimizes positive-negative ordering inside each user group",
  "files_to_change": ["training/losses.py", "configs/experiments/exp_017.yaml"],
  "expected_metrics": ["GAUC"],
  "risk": "May underweight top-5 positions",
  "cheap_test": {"dataset": "shadow_fold_2", "seeds": [42]},
  "promotion_rule": "primary_delta >= 0.001 and no metric delta < -0.002",
  "fallback": "rollback_git_commit"
}
```

Persist:

- parent and child relationships;
- exact git commit and diff;
- config and command;
- data/evaluator/environment hashes;
- stdout/stderr and failure/recovery events;
- cheap/full metrics by fold and seed;
- model and prediction artifacts;
- token, CPU/GPU, and wall-clock usage; and
- the evidence-backed diagnosis that created the next experiment.

SQLite is safer for the live store than one increasingly large JSON file. Export JSONL and Markdown tables for judging.

### 6.3 Search operators

Use explicit operators so the LLM does not rewrite the world every iteration:

1. `REPAIR`: minimal patch for syntax, import, runtime, NaN, timeout, or artifact failure.
2. `LOSS`: pointwise BCE -> PairLogit/BPR -> LambdaRank/LambdaLoss/listwise variants.
3. `FEATURE`: add one leakage-safe feature family and its ablation.
4. `SEQUENCE`: add candidate-conditioned history representation.
5. `AUX_TASK`: add or remove one auxiliary task or change task sharing/weights.
6. `MODEL_BLOCK`: replace one block while preserving data/evaluator contracts.
7. `HYPERPARAMETER`: tune only after a mechanism has demonstrated signal.
8. `ENSEMBLE`: optimize blend of diverse surviving candidates.
9. `ABANDON`: terminate a branch with a documented reason.

Each patch runs static checks, unit tests, a tiny fixture, and data-firewall checks before consuming an experiment budget.

### 6.4 Search policy

A practical two-person hackathon policy is a cost-aware beam, not full MCTS:

- Keep the best three viable branches: incumbent, mechanism variant, and diversity branch.
- Rank candidates by robust expected improvement, not a single score:

```text
utility = mean_primary_across_folds
          - 0.5 * seed_std
          - instability_penalty
          - small_cost_penalty
          + novelty_bonus
```

- Allocate roughly 70% of proposals to improving a strong branch, 20% to a different mechanism, and 10% to repair/uncertainty resolution.
- Use successive halving: fixture -> one shadow fold/seed -> all shadow folds -> official validation -> multi-seed confirmation.
- Use the strong LLM for proposal, diagnosis, and difficult repair; use a cheaper model for log compression, schema normalization, and obvious error classification.
- Distill every completed experiment into short reusable lessons so restarts do not lose context.

The official convergence counter must be implemented exactly and kept separate from search utility. Define and document whether cheap probes count toward the organizer's 50-iteration cap; use the conservative interpretation unless the organizers clarify otherwise.

### 6.5 Replacement repository architecture

A concrete target structure is:

```text
src/rex/
  cli.py                         # init-run, run, resume, report, validate-submission
  contracts.py                   # frozen typed schemas and status enums
  control/
    coordinator.py               # state machine only; no model logic
    search_policy.py             # beam selection and budget allocation
    budget.py                    # monotonic wall clock, attempts, tokens, compute
    recovery.py                  # bounded typed repair policy
  agent/
    proposer.py                  # hypothesis + experiment plan
    coder.py                     # localized patch creation
    diagnostician.py             # evidence-bound interpretation
    memory.py                    # global safety lessons + branch-local scientific memory
    method_cards/                # versioned recsys knowledge with citations
  data/
    manifest.py                  # hashes, roles, dates, schema
    access.py                    # train/evaluator/test capability boundaries
    temporal_features.py         # point-in-time feature computation
    groups.py                    # complete-user sampling and pair generation
  evaluation/
    official_adapter.py          # only wrapper allowed to call frozen evaluate.py
    shadow_folds.py
    diagnostics.py
    submission.py
  models/
    fm.py
    rank_fm.py
    tree_ranker.py
    history_din.py
    multitask.py
    watchtime.py
    ensemble.py
  execution/
    sandbox.py
    runner.py
    artifacts.py
    git_workspace.py
  store/
    schema.sql
    experiments.py
    events.py
tests/
  unit/
  integration/
  fault_injection/
vendor/organizer/                # read-only starter kit with recorded hashes
runs/<run_id>/                   # manifest, DB, events, branches, artifacts, final bundle
```

The coordinator should be deliberately boring. It advances experiments through an explicit state machine:

```text
PROPOSED -> PATCHED -> STATIC_VALID -> FIXTURE_VALID
         -> CHEAP_COMPLETE -> FULL_COMPLETE -> CONFIRMED
         -> SUBMISSION_VALID -> PROMOTED
```

Any transition may instead end in `REJECTED`, `FAILED_REPAIRABLE`, or `FAILED_FINAL`. State and event records must commit before the next action. A candidate is not the incumbent until its metrics are finite, evaluator hash matches, predictions align exactly, required artifacts exist, and submission validation succeeds. This ordering fixes the present “promote first, warn later” bug.

### 6.6 Contracts the LLM may not change

Give the coding agent write access only to an experiment worktree. At minimum, deny modification of:

- data manifest, role definitions, and split dates;
- official evaluator and its expected hash;
- test-label firewall;
- metric aggregation and promotion rules;
- run-budget accounting;
- artifact validator; and
- the event-log writer.

The LLM may change model, loss, feature, and experiment configuration code. Changes outside that allowlist require a `REPAIR_INFRASTRUCTURE` proposal, deterministic tests, and a separately labeled non-scientific attempt. This prevents a model from “improving” by accidentally altering evaluation.

### 6.7 One complete experiment transaction

For every scientific iteration:

1. Retrieve the incumbent, branch-specific history, globally relevant safety lessons, remaining budget, and allowed method cards.
2. Produce a structured hypothesis with one primary change, expected metric/segment effect, falsifier, estimated cost, and leakage analysis.
3. Create a clean git worktree from the chosen parent and save the proposed patch before executing it.
4. Run formatting, imports, static checks, unit tests, evaluator-hash check, and a tiny synthetic ranking fixture.
5. Execute the cheap rung on complete user groups from a temporal shadow fold.
6. Promote only if the prespecified rule passes; otherwise record a scientific rejection rather than asking the LLM to rationalize it.
7. Execute the full rung in a process group with timeout and telemetry.
8. Generate official metrics, confidence intervals, segment diagnostics, prediction correlation, and parent disagreement deterministically.
9. Ask the diagnostician to classify evidence as supported, contradicted, or inconclusive and to cite artifact IDs.
10. Commit a concise reusable lesson, update the search graph, and atomically select the next parent.
11. If it is a potential incumbent, run multi-fold/seed confirmation and submission validation before promotion.

The LLM never supplies the score, decides whether its own output file is valid, or edits the evidence after seeing the result.

## 7. Highest-probability model strategy

### 7.1 Priority 1: ranking-aligned loss on the organizer FM

This is the lowest-risk, highest-information first change.

The baseline optimizes pointwise log loss even though both official metrics depend only on within-user order. [BPR](https://arxiv.org/abs/1205.2618) established the benefit of optimizing positive-negative preference pairs for implicit-feedback ranking. [LambdaLoss](https://research.google/pubs/the-lambdaloss-framework-for-ranking-metric-optimization/) provides metric-driven losses connected to NDCG. The organizer explicitly names this as the top unexplored direction.

Run these controlled experiments before changing architecture:

1. FM + pairwise logistic loss with complete within-user groups.
2. FM + sampled pairwise loss, sampling equal negatives per positive.
3. FM + pair weights based on delta-nDCG@5.
4. Hybrid loss: pairwise ranking + a small pointwise BCE stabilizer.

Acceptance should require both GAUC and nDCG@5 reporting because a pure pairwise objective may improve average ordering while hurting the top five.

There is a useful derivation behind the negative sampler. Let user `u` have `p_u` positives and `n_u` negatives. Its AUC is the average correctness over `p_u * n_u` positive-negative pairs, while GAUC weights that AUC by `p_u`. Therefore GAUC is proportional to a sum of within-user pair correctness in which every pair receives weight `1 / n_u`. Sampling a fixed number `K` of negatives for every positive produces about `p_u * K` pairs per user and approximates the metric's positive-count weighting. Consequences:

- negatives must be sampled from the same user, never globally;
- use a fixed or capped number per positive rather than letting high-impression users dominate;
- exclude all-positive/all-negative groups from pairwise loss while retaining any valid pointwise or auxiliary contribution; and
- log effective pairs per user so the implementation can be audited.

For the nDCG half, use LambdaLoss or pair weights based on the change in nDCG@5 caused by swapping a positive and negative. Normalize the pairwise and Lambda-loss scales before combining them; otherwise a nominal coefficient does not describe their real gradient contribution. A small BCE term can stabilize representation learning, but it should not dominate the ranking gradients.

### 7.2 Priority 2: leakage-safe historical affinity features + tree ranker

Build an efficient, strong non-neural branch with CatBoostRanker or LightGBM LambdaRank. [CatBoost's ranking objectives](https://catboost.ai/docs/en/concepts/loss-functions-ranking) support within-group pair generation and NDCG-oriented objectives; [LightGBM's LambdaRank parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html) expose an NDCG-focused truncation level that should be tuned near the target cutoff.

Feature families, each added as a separate ablation:

- smoothed item and author long-view rates and exposure counts;
- exponentially decayed item/author rates;
- user x author affinity;
- user x tag/category affinity;
- prior user-item impressions, prior positive count, last outcome, and time since prior exposure;
- candidate popularity trend rather than lifetime popularity only;
- candidate duration, duration bucket, and user x duration preference;
- hour, weekday/date, tab, and context interactions;
- candidate similarity to the user's positive-history item/author/tag distribution;
- cold-user/item flags and unseen-value backoff hierarchy.

All target-derived training features must be computed out-of-fold or strictly from earlier timestamps. Validation features may use training history only. For final test training, retrain on train+validation only if the competition protocol permits it, and never use test labels.

The starter's static-feature ablation does not invalidate this branch: it tested raw categorical fields inside an FM, not leakage-safe temporal target statistics and candidate-conditioned affinities inside a ranker.

### 7.3 Priority 3: candidate-conditioned behavior history

[DIN](https://arxiv.org/abs/1706.06978) is a good first neural history model because it computes a candidate-specific interest representation instead of compressing all history into one fixed vector. KuaiRand's authors explicitly identify long sequential modeling as a supported research direction in the [KuaiRand paper](https://arxiv.org/abs/2208.08696), although KuaiRand-Pure contains incomplete histories compared with the larger variants.

Start simpler than a transformer:

- last 20/50 positive and exposed item IDs;
- last 20/50 author and tag/category IDs;
- time gaps and outcomes;
- attention from candidate item/author/tag to history;
- recency decay; and
- fallback to aggregate affinity for short histories.

Compare against cheap non-neural sequence summaries first. A full SIM/SDIM-style long-history system is unnecessary unless DIN-like attention clearly helps.

### 7.4 Priority 4: multi-task learning as auxiliary representation learning

KuaiRand contains 12 feedback signals, and its paper explicitly calls out multi-task learning. [MMoE](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/) learns task-specific gates over shared experts. [PLE](https://doi.org/10.1145/3383313.3412236) was designed to reduce negative transfer and the "seesaw" phenomenon in personalized recommendation. [PCGrad](https://arxiv.org/abs/2001.06782) is a model-agnostic fallback for conflicting task gradients.

Do not begin with all 12 tasks. Use correlation, prevalence, missingness, and gradient diagnostics to choose auxiliaries:

1. `is_click` and watch-time/completion-derived targets;
2. add `is_like` only if it supplies enough positives;
3. add sparse follow/comment/forward signals one at a time;
4. remove tasks whose gradient cosine with `long_view` is persistently negative.

Use long_view as the only promotion metric. Auxiliary metrics are diagnostic, never part of the official objective.

For `is_click`, first report prevalence and agreement with `long_view` separately by interface/tab. Prefer tab-conditioned or interface-specific click heads if the schema confirms mixed UI semantics. This turns a likely source of negative transfer into a testable hypothesis rather than assuming every feedback label shares one task definition.

### 7.5 Priority 5: threshold-aware watch-time distribution

The exact target invites a stronger branch than generic watch-time regression: learn the probability that watch time exceeds the candidate-specific threshold `tau = min(duration, 18 seconds)`.

A practical staged design is:

1. Primary head: binary long-view loss for `P(W >= tau | x)`.
2. Auxiliary head: predict a conditional watch-time distribution or several quantiles from training-row play time.
3. Censoring treatment: a completed short video reveals only that latent interest may extend beyond its duration, rather than proving the user's desired watch time equals the duration.
4. Ranking layer: combine the threshold probability with the same-user pairwise and delta-nDCG@5 losses.

[CWM](https://arxiv.org/abs/2406.07932) models counterfactual watch time under right censoring by video duration. [D2Q](https://arxiv.org/abs/2206.06003) and [Conditional Quantile Estimation for Watch Time](https://arxiv.org/abs/2407.12223) offer distributional alternatives; more recent work such as [DADF](https://arxiv.org/abs/2605.17863) and [Relative Advantage Debiasing](https://arxiv.org/abs/2508.11086) reinforces that raw watch time is duration-biased. These are mechanism priors, not directly comparable challenge results.

Do not port any paper's old environment wholesale. Reimplement the smallest useful loss in current PyTorch and ablate it against a plain auxiliary regression head. Current-row watch time is a training label only. Keep the official native `long_view` label and evaluator unchanged; a paper-specific rebuilt label is not a valid substitute.

### 7.6 Priority 6: feature-cross model only after mechanism gains

[DCN-V2](https://arxiv.org/abs/2008.13535) is a better default than blindly enlarging FM embeddings because it targets efficient learned crosses. DeepFM/xDeepFM remain valid diversity branches, but the organizer's capacity ablation makes them lower priority than loss, history, and auxiliary information.

### 7.7 Final ensembling

Ensembling is likely the safest last source of hidden-test gain because the strongest branches fail differently:

- pairwise FM: collaborative ID interactions;
- boosted ranker: historical aggregates and nonlinear context;
- DIN/PLE: sequence and auxiliary-signal representation;
- CWM head: duration-aware interest.

Blend per-user standardized scores or percentile ranks, because only within-user order matters. Optimize blend weights on internal temporal folds, then confirm once on official validation. Include single models and blends in the experiment graph so the agent, not a human, chooses the final artifact.

### 7.8 Conditional high-upside branch: semantic video content

The official KuaiRand repository now lists supplementary category and caption files for the 7.5K-video candidate pool. If the organizers confirm these files count as part of the allowed KuaiRand assets, they enable:

- hierarchical category IDs and confidence scores;
- candidate-to-positive-history category similarity;
- caption embeddings from an allowed pretrained text encoder;
- user semantic-interest centroids; and
- cold-item backoff when item-ID statistics are weak.

This is **not** an automatic early priority. First ask the organizers in writing whether the supplementary files are permitted under the "no external training data" rule. If approved, test semantic similarity as a feature/ensemble branch. If not approved, do not download or use it for training.

### 7.9 What public KuaiRand implementations do and do not prove

Public repositories are useful implementation references, but their headline results are generally not comparable with this challenge:

- [KuairandRec](https://github.com/Under-the-dome/KuairandRec) implements Shared-Bottom, MMoE, and PLE and includes bias/bucket analyses. Its published tables primarily use AUC and different task/split choices, so they are code priors—not evidence of beating the organizer's GAUC/nDCG@5 FM.
- [FuXi-Linear](https://github.com/USTC-StarTeam/fuxi-linear) is a 2026 time-aware long-sequence architecture with KuaiRand-27K results. It solves sequential full-item recommendation with different metrics, so it is a late research inspiration rather than a drop-in candidate for KuaiRand-Pure's logged-impression ranking.
- CWM reports nDCG variants on a rebuilt label and an older implementation stack. Its loss mechanism is relevant; its numbers are not a challenge baseline.

The agent's knowledge base should store each external method with an explicit `task_match` field: label, split, candidate protocol, metric, and data variant. This prevents invalid cross-paper score comparisons.

### 7.10 Falsifiable first-wave experiment portfolio

These are hypotheses, not promises. Each experiment changes one mechanism relative to a named parent so the agent can learn from failure.

| ID | Parent | Controlled change | Expected evidence | Falsifier / action |
|---|---|---|---|---|
| E00 | Organizer | Reproduce FM across five seeds | Mean valid primary near 0.6016; valid CSV | Stop all research if reference cannot be reproduced |
| E01 | E00 | Same-user PairLogit, fixed negatives per positive | GAUC increases; nDCG stable | Reject if multi-fold primary does not improve |
| E02 | E01 | Add delta-nDCG@5 pair weights | nDCG@5 improves most at ranks 1-5 | Revert if GAUC loss exceeds prespecified tolerance |
| E03 | E02 | Add small BCE stabilizer | Lower seed variance without lost ranking | Drop BCE if only calibration changes |
| E04 | E00 | LightGBM LambdaRank on non-target static/context features | Fast nonlinear diversity branch | Reject if it merely reproduces popularity ordering |
| E05 | E04 | Point-in-time item/author rates | Broad primary gain | Block if leakage tests or date-forward ablation fail |
| E06 | E05 | Prior repeated-exposure features | Gains on repeated-pair segment and overall | Retain only if overall effect survives folds |
| E07 | E05 | User-author/tag/duration affinities | Personalized gains, lower popularity correlation | Reject if cold users regress catastrophically |
| E08 | E05 | Recency-decayed summaries | Gains on later folds and active users | Reject if decay is unstable across dates |
| E09 | E02 | Simple candidate-history dot products | Establish sequence headroom cheaply | Do not build DIN if no reliable gain |
| E10 | E09 | DIN-like candidate attention | Gain beyond E09, especially long-history users | Prefer E09 if added capacity is not justified |
| E11 | E10 | Tab-aware click auxiliary | Shared representation improves long_view | Remove on negative gradient/metric transfer |
| E12 | E10 | Like auxiliary added separately | Additional signal in non-sparse segments | Remove if task is too sparse or conflicting |
| E13 | E10 | Threshold-aware watch-time distribution head | Gains by duration bucket and primary | Fall back to binary head if instability/cost dominates |
| E14 | Winners | Per-user rank blend of rank-FM and tree/history branch | Robust gain from disagreement | Reject if one model dominates all folds |
| E15 | E14 | Add watch-time/MTL diversity branch | Small final complementary gain | Keep simpler blend if delta is below noise |

Never run E10 because it sounds impressive; E09 must first demonstrate that incomplete KuaiRand-Pure history contains useful incremental signal. Never run E11-E13 simultaneously; otherwise failure cannot be attributed.

## 8. Validation design that resists adaptive overfitting

The official validation window should not be the only decision surface for 50 adaptive experiments.

Repeatedly choosing experiments from one holdout is an adaptive-data-analysis problem, not merely a seed problem. The [Reusable Holdout](https://pubmed.ncbi.nlm.nih.gov/26250683/) and [Ladder leaderboard](https://proceedings.mlr.press/v37/blum15.html) formalize why repeated feedback can overfit a holdout even when nobody explicitly trains on its labels. We cannot change the organizer's evaluator, but we can ration official-validation looks and make internal temporal evidence the main search signal.

Create rolling shadow folds inside the training period, for example:

- fold A: train days 08-14, evaluate 15-16;
- fold B: train days 08-16, evaluate 17-18;
- fold C: train days 08-18, evaluate 19-21.

Exact fold boundaries should be chosen after EDA so each fold has adequate user groups and positives. Preserve complete user groups in every evaluation batch.

Promotion policy:

1. Cheap fixture proves correctness only.
2. One shadow fold filters obvious losses.
3. All shadow folds estimate generalization and variance.
4. Official validation is used only for promoted candidates.
5. A new incumbent requires a meaningful delta and no catastrophic metric regression.
6. Finalists run 3 seeds; the final blend/checkpoint runs 5 seeds if budget permits.

This makes the hidden-test result less dependent on one lucky validation seed and gives the reflection system real evidence about temporal drift.

### 8.1 Random-exposure data policy

KuaiRand's random-exposure log is valuable because it weakens the production policy's exposure bias. Use it as a diagnostic or stress-test split to ask whether an improvement reflects preference rather than the logging policy. [Unbiased Learning-to-Rank](https://arxiv.org/abs/1608.04468) and [Dual Learning Algorithm for Unbiased Learning to Rank](https://arxiv.org/abs/1804.05938) provide the relevant propensity-correction background.

Do not train on the random log by default. Its dates overlap the official validation/test era, the fixed challenge split names only the standard log, and the hidden test appears to follow the standard-policy distribution. Training on random exposures could violate the protocol or improve an unbiased diagnostic while hurting the actual score. Only use random rows for training if organizers explicitly approve it; only adopt propensity weighting if it improves temporal folds and the official validation, not because it is theoretically attractive.

### 8.2 Leakage and target-availability matrix

| Signal family | Training-row use | Validation/test-row use | Required safeguard |
|---|---|---|---|
| `user_id`, `video_id`, author, tab, time context | Feature | Feature | Unknown-value handling and frozen encoders |
| Basic item duration/upload metadata | Feature | Feature | Verify availability at impression time |
| Current `long_view` | Primary target | Evaluator only | Inaccessible to model process outside training |
| Current play time/click/like/follow/etc. | Auxiliary targets only | Never feature input | Column denylist enforced by data view |
| Prior interactions for the same user | Point-in-time feature | History ending before row timestamp | Stable tie ordering and no same-timestamp future rows |
| Target rates by item/author/tag | OOF or earlier-time statistics | Built only from allowed earlier split | Smoothing, minimum support, cutoff metadata |
| Supplied monthly statistic file | Disallowed until proven safe | Disallowed until proven safe | Column-level provenance and window-end audit |
| Random-exposure rows | Diagnostic split | Diagnostic split | No training use without written permission |
| Supplementary categories/captions | Conditional | Conditional | Written organizer approval and asset hash |
| Test labels from public KuaiRand source | Never | Evaluator service only, ideally unavailable locally | Separate path/process capability and access audit |

Add automated poison tests: replace forbidden columns with absurd values and prove predictions do not change; shift future labels and prove historical features for earlier rows remain byte-identical; shuffle validation/test labels and prove only the evaluator can observe the difference.

## 9. Diagnostics the agent should generate deterministically

The LLM should interpret artifacts, not invent diagnoses from three scalar metrics.

For every full candidate, generate:

- GAUC, nDCG@5, and primary with deltas against parent and incumbent;
- bootstrap confidence intervals by user;
- per-user-group metric deltas by history length, positive rate, and coldness;
- per-item metrics by popularity, age/recency, duration, and exposure count;
- metric deltas by tab, hour, and date;
- prediction correlation with item popularity;
- pairwise disagreement against the incumbent;
- auxiliary-task gradient cosine matrix for MTL models;
- cheap/full rank correlation for the proxy test;
- inference/training time, memory, token usage, and artifact sizes; and
- code-diff statistics and test results.

Expand the reflection schema to include:

- hypothesis outcome: supported / contradicted / inconclusive;
- causal evidence and uncertainty;
- segment wins and regressions;
- suspected leakage or proxy mismatch;
- next operator and exact parent branch;
- whether the result teaches a reusable lesson; and
- a falsifiable promotion rule for the next experiment.

## 10. Robustness and autonomy design

### 10.1 Failure handling

Classify and route at least:

- syntax/import failure -> localized repair, no training;
- data-contract failure -> abandon or repair data code;
- evaluator/test-firewall mutation -> reject immediately;
- timeout -> reduce scope once, then abandon;
- OOM -> lower batch/history length once, then abandon;
- NaN/non-finite output -> inspect loss/learning rate, bounded repair;
- invalid checkpoint/predictions -> reject candidate;
- metric regression -> retain lesson, do not repair as a code error;
- agent/API failure -> exponential backoff and resume from durable state.

Do not count infrastructure repairs as scientific improvements. Log them separately.

### 10.2 Provenance

For each attempt:

- create an isolated git branch/worktree or commit;
- never mutate the incumbent in place;
- store the patch before execution;
- record the parent commit, environment lockfile, command, seeds, and hashes;
- write logs append-only and optionally hash-chain event records;
- promote by moving an immutable reference after all gates pass; and
- garbage-collect failed model artifacts only after preserving logs and diffs.

### 10.3 Final zero-intervention demonstration

Run from a clean container with:

- one documented launch command;
- pinned dependencies;
- no interactive stdin;
- automatic resume;
- externally captured console logs;
- all experiment commits and artifacts preserved; and
- a final automatically generated intervention/resource summary.

Claim "tamper-evident evidence of zero recorded intervention," not cryptographic proof of no human activity unless an external trusted runner actually enforces that claim.

## 11. Recommended experiment order

Because three consecutive epsilon-level non-improvements can stop the run, try the highest expected-value mechanisms first.

| Priority | Experiment family | Why now | Promotion evidence |
|---|---|---|---|
| 0 | Random/pop/FM reproduction | Validates harness | Matches published references |
| 1 | Pairwise FM | Organizer's top unexplored idea; minimal code delta | Robust GAUC/primary gain |
| 2 | Delta-nDCG/hybrid ranking FM | Balances both metrics | nDCG gain without GAUC loss |
| 3 | Leakage-safe aggregate ranker | High ROI on tabular context | Multi-fold primary gain |
| 4 | Repeat-exposure and affinity features | Candidate-conditioned personalization | Gains beyond popularity correlation |
| 5 | Simple recency/history summaries | Cheap test of sequence headroom | Improves active users without cold collapse |
| 6 | DIN-like attention branch | Richer sequence mechanism | Gain over simple sequence ablation |
| 7 | Small MMoE/PLE auxiliary branch | Exploits multi-feedback | Long_view gain with controlled conflicts |
| 8 | CWM-inspired auxiliary head | Novel duration-bias mechanism | Gain concentrated in duration segments |
| 9 | DCN-V2/diversity model | Only after feature/loss signal exists | Complementary errors |
| 10 | Per-user rank ensemble | Converts diversity to final gain | Robust multi-fold/seed improvement |

### 11.1 Decision matrix for the first serious run

Scores are relative judgments after reading the starter, schema, and literature; `5` means high.

| Branch | Expected gain | Evidence strength | Novelty | Engineering cost | Leakage/protocol risk | Decision |
|---|---:|---:|---:|---:|---:|---|
| Pairwise FM with same-user sampling | 4 | 5 | 2 | 2 | 1 | Build first |
| Delta-nDCG@5 / hybrid FM | 4 | 4 | 3 | 3 | 1 | Build immediately after pairwise |
| Temporal aggregate tree ranker | 5 | 4 | 3 | 3 | 4 | High priority with strict feature tests |
| Repeat-exposure context | 4 | 4 | 4 | 2 | 2 | Promote to early feature branch |
| Simple history summaries | 4 | 4 | 3 | 3 | 3 | Early mechanism test |
| DIN-like candidate attention | 4 | 4 | 3 | 4 | 3 | Build only if simple history helps |
| Tab-aware click auxiliary | 3 | 3 | 4 | 3 | 2 | Targeted MTL branch |
| Threshold/distributional watch-time head | 4 | 4 | 5 | 4 | 2 | Best innovation branch after core gains |
| DCN-V2/DeepFM capacity branch | 2 | 3 | 2 | 3 | 1 | Defer |
| Supplementary caption/category features | 4 | 3 | 4 | 4 | 5 | Block pending written approval |
| Random-log propensity training | 2 | 3 | 4 | 4 | 5 | Diagnostic only unless approved |

### 11.2 Questions to resolve with organizers in writing

1. Are the official KuaiRand supplementary category and caption files permitted, or does the task allow only files shipped in the starter-kit download?
2. Is the random-exposure log allowed for training, or only for analysis/validation?
3. May the final model retrain on train plus validation after search, or must the submitted model use only the fixed training dates?
4. What exactly counts toward the 50 iterations: every subprocess attempt, only full training runs, cheap probes, and infrastructure repair retries?
5. How is the hidden test scored if labels exist in a public KuaiRand release, and what mechanism should teams use to demonstrate that the agent never accessed them?
6. Which LLM providers/models and external paper or public-code retrieval are allowed during the autonomous six-hour run?
7. Does the bonus reward a live agent-generated final submission, and must that run begin from the starter baseline or may it begin from a team-authored component library?
8. Is GPU/network access guaranteed in judging, and are downloaded pretrained weights required to be cached before the timed run?

Until answered, choose the conservative interpretation and record each assumption in the run manifest.

### 11.3 A realistic 50-iteration / six-hour allocation

The exact cap semantics need confirmation, so this plan assumes every scientific full run counts and repair attempts are separately logged but still budgeted conservatively. Because the official convergence rule may terminate after three weak iterations, ordering matters more than reserving nominal late slots.

| Iterations | Maximum allocation | Goal | Stop/advance rule |
|---|---:|---|---|
| 1-3 | 3 | Harness and FM reference reproduction | No search until metrics and submission align |
| 4-10 | 7 | Pairwise, Lambda-weighted, hybrid FM | Keep one ranking-loss incumbent |
| 11-21 | 11 | Tree ranker and point-in-time aggregate/repeat features | Add one feature family per ablation |
| 22-29 | 8 | Recency and simple history summaries | Establish whether sequence signal exists |
| 30-36 | 7 | DIN-like history only if justified | Limit architecture/hyperparameter branching |
| 37-42 | 6 | Tab-aware MTL and threshold watch-time | One auxiliary mechanism at a time |
| 43-47 | 5 | Cross-family ensembles | Optimize only on shadow folds first |
| 48-50 | 3 | Confirmation, packaging, final validation | No new high-risk architecture |

Wall-clock allocation should reserve approximately 10% for initialization/EDA, 60% for scientific training, 15% for confirmations, 10% for final artifact generation, and 5% as a hard safety margin. The coordinator should forecast remaining time from observed runtimes, not fixed guesses. If deep runs exceed forecast, shrink the branch portfolio rather than shortening every run until results become meaningless.

The six-hour demonstration is not the first time the team should discover which methods work. Before the judged run, manually build and test stable primitives. The autonomous run should still choose, modify, combine, and reject them itself; autonomy does not require starting from an empty editor.

Avoid spending early iterations on:

- FM embedding size sweeps;
- raw static feature expansion without interactions;
- calibration-only changes;
- pure user-side features;
- all-12-task MTL from the start;
- huge transformers;
- training on random-exposure logs without explicit permission; or
- dashboard polish before a baseline-beating candidate exists.

## 12. Rubric strategy

### Technical Execution — 35%

- Beat 0.6016 validation by more than noise, preferably with a multi-seed margin.
- Demonstrate exact starter-kit reproduction.
- Show fault-injection tests for crash, timeout, NaN, invalid output, API interruption, and resume.
- Submit an officially validated artifact generated by the best checkpoint.

### Innovation & Problem Insight — 20%

- Make ranking-loss alignment the first scientific hypothesis.
- Show deterministic segment diagnostics and confidence intervals.
- Include one real negative-transfer or duration-bias diagnosis with before/after evidence.
- Connect every major experiment to a paper and an observed dataset symptom.

### Impact & Relevance / Autonomy — 20%

- Demonstrate code-writing and self-repair, not config tuning only.
- Provide a complete experiment DAG with automatic parent selection.
- Report zero or near-zero interventions from the event log.
- Show resume after an injected failure.

### Feasibility — 15%

- Use successive halving and targeted refinement.
- Cache data features and embeddings across experiments.
- Route cheap and strong LLM calls by task.
- Report all wall-clock, token, CPU/GPU, and iteration counts.
- Do not sacrifice model quality merely to appear cheap; this category is gated by beating baseline.

### Presentation — 10%

Tell one coherent story:

1. The starter FM optimized the wrong kind of objective.
2. The agent discovered a ranking-aligned gain.
3. It diagnosed the remaining errors by user/history/duration segment.
4. It autonomously added sequence or auxiliary modeling.
5. It rejected failed branches and blended complementary winners.
6. It survived injected failures and produced a valid final submission without intervention.

The most persuasive visual is an experiment tree annotated with primary score, hypothesis, code diff, and diagnosis—not a generic architecture diagram.

### 12.1 Evidence matrix for judge-facing claims

| Claim | Required artifact | Automatic verifier |
|---|---|---|
| “We reproduced the baseline” | Five seed records, raw predictions, evaluator output | Reference tolerance test |
| “The agent wrote the improvement” | Proposal, parent hash, patch, resulting commit | Patch provenance and allowlist check |
| “This mechanism caused the gain” | Single-change child and ablation/control | Experiment graph comparison |
| “The score is valid” | Prediction file and frozen evaluator output | Evaluator hash plus finite/alignment checks |
| “The final CSV is valid” | Submission linked to incumbent experiment | Official `submit.py --check` result |
| “The agent recovered autonomously” | Injected failure, typed event, repair patch, resumed run | Fault-injection integration test |
| “No human intervened during the demonstration” | External process log plus append-only events | Run-manifest continuity and intervention channel |
| “The result is reproducible” | Code/data/environment hashes, command, seed, hardware | Clean-container replay |
| “Reflection was evidence-based” | Claims carrying artifact/metric IDs | Schema and referential-integrity check |
| “Resource reporting is complete” | LLM usage, monotonic wall time, CPU/GPU/RAM telemetry | Totals reconciled against per-event records |

A judge should be able to click from the final score to the exact experiment, code diff, prediction hash, evaluator output, and hypothesis without trusting narrative prose.

## 13. Concrete build order for two people

### Phase A: make the benchmark undeniable

1. Download data and verify hashes/row counts.
2. Reproduce random, popularity, and five-seed FM validation references.
3. Add unit tests for `evaluate.py` edge cases and keep its source immutable.
4. Build the test-label firewall and official submission path.

### Phase B: replace the stub with a real experiment platform

1. Define experiment/proposal/result schemas.
2. Use SQLite + append-only event JSONL.
3. Implement git-isolated patch, test, execute, evaluate, rollback, and promote operators.
4. Fix wall-clock, convergence, resume, validation, and dry-run behavior.
5. Add fault-injection integration tests.

### Phase C: establish a manual upper-bound portfolio

Before trusting autonomous search, implement and validate templates for:

- pairwise FM;
- leakage-safe ranker;
- simple history features; and
- per-user ensemble.

These are safe primitives the agent can modify. The challenge requires the agent to write improvements, but a reliable component library prevents every iteration from rebuilding infrastructure.

### Phase D: autonomous scientific loop

1. Add targeted code operators and structured hypotheses.
2. Add ablation-driven component selection.
3. Add cost-aware beam search and episodic lessons.
4. Add deterministic diagnostic reports.
5. Run shadow-fold trials, then an uninterrupted official demonstration.

### Phase E: final evidence package

1. Five-seed finalist comparison and final ensemble.
2. Official submission validation.
3. Auto-generated run table and experiment-tree figure.
4. Resource/intervention report.
5. README reproduction command, limitations, team contributions, and Devpost narrative.

### 13.1 Two-person division that avoids integration deadlock

| Person A: experiment platform | Person B: modeling/data | Shared checkpoints |
|---|---|---|
| Frozen evaluator adapter and firewall | Baseline reproduction and data manifest | Agree on prediction/result contracts first |
| SQLite/event store and state machine | Pairwise/Lambda loss and group sampler | E00-E03 run through the real coordinator |
| Git worktrees, runner, recovery | Point-in-time aggregates and tree ranker | Leakage tests reviewed by both |
| LLM schemas, model routing, memory | History/MTL/watch-time model plugins | Every plugin passes the same fixture |
| Reporting and evidence-chain generator | Segment diagnostics and ensemble | Final uninterrupted rehearsal together |

Neither person should build against undocumented Python function calls. Freeze CLI/JSON schemas early so the platform can execute any model plugin and every model produces the same prediction/result artifacts.

### 13.2 Minimum stable component library before autonomy

The agent needs a safe substrate, not a blank repository. Before the autonomous demonstration, provide tested implementations of:

- official FM adapter;
- pair generator and PairLogit/Lambda-weight hooks;
- generic point-in-time aggregate builder;
- tree-ranker adapter;
- simple history encoder;
- model-plugin protocol;
- per-user rank blender; and
- frozen evaluation/diagnostic/submission services.

The agent can then write new feature transforms, losses, model blocks, task heads, and combinations. This is comparable to giving a research scientist a functioning lab; the autonomous contribution is the experimental reasoning and code evolution, not recreating CSV parsing on every run.

## 14. Definition of done

The project is not complete until there is evidence for every item:

- [ ] Random/popularity/FM references reproduced from the starter kit.
- [ ] A real validation score above the official FM baseline.
- [ ] At least one improvement proposed, coded, tested, and evaluated by the agent.
- [ ] Exact convergence and best-checkpoint behavior tested.
- [ ] Test labels inaccessible during research.
- [ ] Every iteration contains hypothesis, patch/diff, metrics, and recovery events.
- [ ] Crash, timeout, NaN, invalid submission, and restart are automatically handled.
- [ ] Final CSV passes the official validator.
- [ ] Final checkpoint and submission are linked to the same experiment record.
- [ ] Token, wall-clock, iteration, CPU/GPU, and human-intervention totals are reported.
- [ ] Clean-environment one-command run is demonstrated.
- [ ] Public README and written project description cover every organizer deliverable.

## 15. Risk register and go/no-go decisions

| Risk | Likelihood | Impact | Leading indicator | Mitigation / decision |
|---|---:|---:|---|---|
| Baseline cannot be reproduced | Medium until data is present | Fatal | Reference delta exceeds tolerance | Freeze research and repair data/evaluator first |
| Validation overfitting | High | High | Official-valid improves while shadow folds diverge | Ration official looks; require multi-fold support |
| Target leakage in aggregate features | High | Fatal | Unrealistic jump or future-date sensitivity | Point-in-time builder, poison tests, provenance |
| Agent edits evaluator or firewall | Medium | Fatal | Diff touches protected paths | Deny writes and reject commit automatically |
| Convergence stops before diverse search | Medium | High | Two weak iterations after incumbent | Put highest-value hypotheses first; clarify rule |
| Cheap rung misranks candidates | Medium | Medium | Low cheap/full rank correlation | Recalibrate proxy or remove it |
| MTL negative transfer | High | Medium | Auxiliary improves, long_view declines | One task at a time, gradient diagnostics, PLE/PCGrad |
| Deep model consumes six-hour budget | Medium | High | Runtime forecast breaches reserve | Cap history/model size; retain tree/FM branch |
| Random/supplementary data violates rules | Medium | Fatal | No written organizer response | Keep branch disabled |
| Test labels accidentally visible | High with public raw data | Fatal reputationally | Test file contains label columns | Physical/process separation and access audit |
| “Zero intervention” claim is challenged | Medium | High | Only self-authored heartbeat evidence exists | External runner/log, conservative wording |
| Agent generates invalid but high-scoring artifact | Medium | Fatal | Missing rows, NaN, alignment mismatch | Validate before transactionally promoting |
| Public-method result is incomparable | High | Medium | Different label/split/metric/candidate set | Store and enforce `task_match` metadata |

The project has four non-negotiable go/no-go gates:

1. No autonomous search before exact FM reproduction.
2. No target-derived feature before leakage tests pass.
3. No candidate promotion before evaluator and submission validation pass.
4. No public “autonomous winning agent” claim before one uninterrupted clean-environment run produces the final evidence bundle.

## 16. Research sources

### Challenge and dataset

- [KuaiRand official repository and data documentation](https://github.com/chongminggao/KuaiRand)
- [KuaiRand paper](https://arxiv.org/abs/2208.08696)
- [Reusable Holdout](https://pubmed.ncbi.nlm.nih.gov/26250683/)
- [The Ladder](https://proceedings.mlr.press/v37/blum15.html)
- [Unbiased Learning-to-Rank](https://arxiv.org/abs/1608.04468)
- [Dual Learning Algorithm for Unbiased Learning to Rank](https://arxiv.org/abs/1804.05938)

### Autonomous ML and code search

- [MLE-bench](https://openai.com/index/mle-bench/)
- [AIDE](https://arxiv.org/abs/2502.13138)
- [AI Scientist-v2](https://arxiv.org/abs/2504.08066)
- [AI Research Agents for Machine Learning](https://arxiv.org/abs/2507.02554)
- [MLE-STAR](https://research.google/pubs/mle-star-machine-learning-engineering-agent-via-search-and-targeted-refinement/)
- [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
- [KAPSO](https://arxiv.org/abs/2601.21526)
- [MARS](https://arxiv.org/abs/2602.02660)
- [MLEvolve](https://arxiv.org/abs/2606.06473)
- [Demystify the Role of Memory in Machine Learning Engineering Agents](https://aclanthology.org/2026.findings-acl.525/)
- [ScientistOne / Chain-of-Evidence](https://arxiv.org/abs/2605.26340)

### Recommender and ranking methods

- [BPR](https://arxiv.org/abs/1205.2618)
- [LambdaLoss](https://research.google/pubs/the-lambdaloss-framework-for-ranking-metric-optimization/)
- [CatBoost ranking objectives](https://catboost.ai/docs/en/concepts/loss-functions-ranking)
- [LightGBM ranking parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [DIN](https://arxiv.org/abs/1706.06978)
- [SIM](https://arxiv.org/abs/2006.05639)
- [MMoE](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/)
- [PLE](https://doi.org/10.1145/3383313.3412236)
- [PCGrad](https://arxiv.org/abs/2001.06782)
- [CWM](https://arxiv.org/abs/2406.07932)
- [D2Q: Duration-Deconfounded Quantile-Based Framework](https://arxiv.org/abs/2206.06003)
- [Conditional Quantile Estimation for Watch Time](https://arxiv.org/abs/2407.12223)
- [DADF: Duration-Aware Distribution Fusion](https://arxiv.org/abs/2605.17863)
- [Relative Advantage Debiasing](https://arxiv.org/abs/2508.11086)
- [DCN-V2](https://arxiv.org/abs/2008.13535)
- [FuXi-Linear](https://arxiv.org/abs/2602.23671)
