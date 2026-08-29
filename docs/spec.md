# Autonomous ML Research Agent for Recommender Systems
## Final Product Spec — Team of 2 (v2, matches official spec update)

> **Archived design note.** This document predates the repository-wide audit and
> contains superseded compute, autonomy, and convergence assumptions. The runtime
> authority is [`task_contract.md`](task_contract.md); the current build sequence
> is [`implementation_plan.md`](implementation_plan.md). Do not use this file as
> an execution contract.

---

## 1. What we're building, in one sentence

A **recsys-specialized autonomous agent** that reproduces the KuaiRand-Pure baseline, then iteratively improves it through a tree-search loop guided by a domain-specific "trick menu," a structured diagnosis system, cheap-before-expensive testing, and bulletproof error recovery — while structurally proving (not just claiming) that it ran with zero human intervention.

**The core bet:** generalist agents (AIDE, R&D-Agent, ML-Master) are built to be okay at any Kaggle problem. We win by being the recsys specialist, and by taking the "boring plumbing" (logging, cost control, diagnosis quality) seriously when other hackathon teams will skimp on it under time pressure.

---

## 2. The task, precisely (corrected)

- **Target label:** `long_view` (did they watch a meaningful chunk of the video) — **not** click. `long_view` is logged on *every* impression, clicked or not, so classic sample-selection bias doesn't apply here — one less problem to defend against.
- **Metrics:** `GAUC` (per-user ranking quality, positive-count weighted) and `nDCG@5` (not @10). **Primary score = mean(GAUC, nDCG@5)`. Recall is not scored — it's ~0.999 for every model including random, so it was dropped as meaningless on this dataset.
- **Reference numbers (hidden test, from the Starter Kit):**
  - Official baseline (Factorization Machine, k=16): **primary 0.5946**
  - Random scoring: 0.4753
  - Popularity-only (always recommend the most-watched videos): **0.5715**
  - Theoretical ceiling: **0.8645**, not 1.0 — ~27% of users have zero positive labels, so no model can ever get them right. Judge every score against this ceiling, not against 1.0.
- **Compute:** the reference pipeline is **numpy-only, CPU-only**, baseline runs in ~40s on one core. No GPU is needed anywhere in this project.
- **Hard caps:** 50 iterations per run, convergence rule ε=0.002 over N=3 consecutive iterations, 6h wall-clock backstop.
- **Submission format:** CSV with `row_id, user_id, video_id, score` — `row_id` is required because `(user_id, video_id)` isn't unique in the eval split. The Starter Kit ships `submit.py --check` to validate format before submission.
- **Feasibility scoring is gated:** it's only scored for submissions that already beat baseline, and it's bucketed into low/medium/high, not ranked continuously. Being cheap doesn't rescue a submission that doesn't work — but it also means we don't need to obsess over shaving the last few tokens, just stay comfortably out of the "high" bucket.

---

## 3. Checkpoint & convergence semantics — locked decisions

These are precise, load-bearing rules, decided now (planning stage) so the eventual build has zero ambiguity. Each is chosen specifically to maximize scoring correctness, not for coding convenience.

**Two independent trackers, not one:**
- **Plateau tracker** — watches only the last 3 iterations' validation primary scores, purely to decide *when to stop* (ε=0.002 rule).
- **Best-ever tracker** — updated every single iteration independent of the plateau tracker, holding the single highest validation primary score seen so far and which checkpoint produced it.

**What gets submitted:** whatever the best-ever tracker holds at the moment the run stops — for *any* stop reason (plateau, 50-iteration cap, or 6h wall-clock). One rule, three possible triggers, no special cases. This directly protects against the realistic failure mode where the score peaks early, drifts down, and the run "converges" on a worse checkpoint than the agent actually found.

**Checkpointing:** save the model checkpoint on **every** iteration, not periodically. The original "periodic checkpointing to save GPU time" reasoning no longer applies — this is CPU/numpy, sub-minute runs, tiny files. Checkpointing every iteration removes all risk of losing the best-ever checkpoint between saves, for effectively zero cost. Pure downside protection, no real tradeoff.

**Tie-break rule:** if two iterations produce an identical validation primary score, prefer the **earlier** one — fewer iterations to reach the same score is consistent with landing in a better Feasibility bucket (less wall-clock, fewer tokens), so ties should resolve toward the metric we're also being graded on.

**Logging implication:** log GAUC and nDCG@5 separately per iteration, not just the combined primary score — this both matches how the real final scoring formula works (delta of each metric, then averaged) and gives Stage 6 a second axis to diagnose against (e.g. "GAUC improved but nDCG@5 dropped" is its own kind of seesaw signal, one level more granular than the click/aux-task seesaw already in the trick menu).

---

## 4. End-to-end architecture

```
┌─────────────────────────────────────────────────────────┐
│  STAGE 0: Orientation (once)                             │
│  Agent reads problem statement + recsys cheat sheet       │
└───────────────────────┬───────────────────────────────────┘
┌───────────────────────▼───────────────────────────────────┐
│  STAGE 1: EDA (recsys-specific, not generic)              │
│  long_view rate skew, signal sparsity, cold-user/item %    │
└───────────────────────┬───────────────────────────────────┘
┌───────────────────────▼───────────────────────────────────┐
│  STAGE 2: Reproduce official baseline (sanity check)       │
│  Confirm we land near primary 0.5946 on validation         │
└───────────────────────┬───────────────────────────────────┘
┌───────────────────────▼───────────────────────────────────┐
│  STAGE 3: Pick next move from the Trick Menu               │
│  (LLM reasons over: EDA findings + past run log +          │
│   diagnosis rules + remaining iteration budget)             │
└───────────────────────┬───────────────────────────────────┘
┌───────────────────────▼───────────────────────────────────┐
│  STAGE 4: Cheap test (5% data, 1 epoch) — kill bad ideas    │
│  early. Validated once against full-run correlation.       │
└───────────────────────┬───────────────────────────────────┘
┌───────────────────────▼───────────────────────────────────┐
│  STAGE 5: Full run + evaluate (GAUC / nDCG@5)               │
│  → always via subprocess + timeout + checkpointing          │
└───────────────────────┬───────────────────────────────────┘
┌───────────────────────▼───────────────────────────────────┐
│  STAGE 6: Structured Reflect (forced JSON diagnosis,        │
│  anchored against popularity baseline + ceiling)            │
└───────────────────────┬───────────────────────────────────┘
                         │  loop back to Stage 3
                         ▼
        until convergence (ε=0.002/N=3) or 50-iteration /
                    6h wall-clock cap hit
```

Since a single iteration is seconds, not minutes (CPU-only, no GPU queueing), we can afford to run the *full* 50-iteration budget in one sitting rather than rationing compute the way a GPU-bound team would — our binding constraint is genuinely the convergence rule and our own idea quality, not resources. This is itself worth stating explicitly in the write-up.

2-3 branches run in a lightweight tree search: mostly exploit the current best, always keep one branch exploring something new.

---

## 4. Core features (build these — this IS the product)

### F1. The Recsys Trick Menu
A written config (not invented by the LLM from scratch) of moves it's allowed to pick from:
- **Feature-side:** feature crossing, sequence features (last-N watched videos), embedding size search
- **Model-side:** DeepFM/DCN-style crossing layers, MMOE-style multi-task learning across the 12 KuaiRand signals (click, like, follow, play_time, etc.) with `long_view` as the scored task and the rest as auxiliary
- **Training-side:** class weighting for `long_view` imbalance, hyperparameter tuning
- **Advanced/bonus:** the organizers explicitly point to CWM (censored-regression loss for watch time) [ref 4] as an optional sophisticated move — not required, but a legitimate, organizer-endorsed source of "originality drawing on published methods" for the Innovation score if time allows

Paired with **diagnosis rules**: "if you observe X pattern, consider Y move." This is what makes Stage 3's picking evidence-based instead of random.

### F2. Structured Stage 6 Diagnosis (JSON schema, forced)
Instead of free-text reflection, force:
```json
{
  "recsys_phenomenon_identified": "seesaw_problem | popularity_bias | cold_start_failure | feature_sparsity | other",
  "evidence": "aux task accuracy +4%, primary score -0.008 vs previous best",
  "notes": "free text, required if 'other'",
  "next_action": "..."
}
```
This is the single highest-leverage feature for Innovation & Problem Insight (20% of grade) — kills lazy "loss didn't converge, try something else" filler.

**New: anchored evidence, not just delta-vs-previous.** Since we now have real reference numbers (popularity baseline 0.5715, ceiling 0.8645), the diagnosis evidence should reference them where relevant — e.g. "primary 0.581, only +0.010 over the popularity-only baseline of 0.5715 → suspect popularity_bias, model may be riding item popularity rather than personalizing." This is a free, specific way to make every diagnosis entry sharper, because we have real anchors the generic version of this idea didn't have.

### F3. Bulletproof Execution Loop
- Every training run launched via `subprocess.run()` with a hard timeout (short — single-core numpy runs finish in under a minute, so a 5-10 min timeout is generous, not 20)
- Checkpoint on **every** iteration, not periodically (see section 3 — cheap here, removes all risk of losing the best-ever checkpoint)
- Failure typed on return: `timeout | crash | nan_loss | success` — each routes the LLM toward a different next move (swapped `oom` for `nan_loss` since there's no GPU memory to exhaust here, but numpy training can absolutely diverge to NaN)
- On crash: LLM sees the real error, gets to retry with a fix (bounded retries)

### F4. Zero-Intervention Proof System
- `agent_state.json`: persisted, updated every cycle — `human_override_count`, iteration history (out of the 50-cap), current best score, budget remaining
- Heartbeat timestamps on every start/stop — a gap larger than normal iteration time is structural evidence of a restart, even if the counter wasn't manually bumped
- This makes autonomy **provable**, not just claimed

### F5. Cheap-Before-Expensive Testing
- `--dev_mode` flag: 5% data sample, 1 epoch, for any new idea before committing a full run
- **Validated once**, early: run 2-3 ideas both cheap and full, confirm the *ranking* agrees, so the shortcut is defensible in the write-up
- Since compute is cheap here (CPU, seconds per run), this trick's role shifts slightly from "essential for survival" to "keeps our wall-clock number in the low/medium bucket for Feasibility" and "lets us explore more of the 50-iteration budget on genuinely promising ideas" — still worth building, just for a slightly different reason than on a GPU-bound benchmark

### F6. Dead Man's Switch
Hard **token + wall-clock** cap (not GPU — there is none), independent of the convergence rule. Force-stops a runaway agent before it burns the 6h backstop or an unreasonable token count. ~20 min to build, protects Feasibility bucket and sanity overnight.

### F7. Seed/Determinism Logging
One field added to the existing run log. Costs nothing since logging infra already exists. Lets either teammate reproduce any specific iteration on demand.

### F8. Submission Validator as a Gate, Not an Afterthought — NEW
The Starter Kit ships `submit.py --check`, which rejects bad headers, row-count mismatches, `row_id` gaps, misalignment, and non-numeric scores. **Wire this into the loop itself**, not just run once at the very end: every time a branch is promoted to "candidate final submission," auto-run the validator against it before accepting the candidate. A formatting rejection at actual submission time is a totally avoidable, embarrassing way to lose Technical Execution points after doing all the hard work — this closes that risk for near-zero build cost.

### F9. Anchored Reporting (Ceiling + Popularity Baseline) — NEW
Because the spec gives us `0.4753` (random), `0.5715` (popularity), `0.5946` (baseline), and `0.8645` (ceiling) as fixed reference points, our results table and write-up should **plot every score against this full range**, not just report a raw number. This does two things other teams likely won't:
1. Shows judgment — "our score of 0.61 captures X% of the attainable range above baseline" reads as more sophisticated than a bare number
2. Makes the popularity-baseline check (0.5715) a built-in sanity gate — if a model's score is barely above 0.5715, that's a red flag the model is just riding popularity, not personalizing, and Stage 6 should catch this automatically via F2's anchored evidence

---

## 6. Deliverable-facing polish (cheap, do near the end)

- **"What we'd do with more time"** section in the write-up — pull a *specific* real example from the agent's own logs (e.g. "iteration 18 flagged unexplored sequence-feature interactions with the cold-start segment"), not a generic statement.
- **Agent vs. Human comparison** — if time allows, one teammate manually attempts to beat baseline for ~30-45 min, timed. Gives you a concrete, punchy report line: "human: 45 min for +0.01 primary. Agent: unattended, converged in N iterations for +0.03." No video is required for this track (a ~3 min video is optional/recommended; a detailed written report is explicitly encouraged as the alternative) — so put the polish into the **report**, not into producing a video. Second priority — only if core system is stable early.
- **Render the run log as a readable table in the report**, not raw JSON — iteration #, hypothesis, phenomenon identified, score, delta vs. previous best. This is cheap (you already have the data logged) and is what a judge will actually read, versus a wall of JSON they have to parse themselves.

---

## 7. Explicitly cut (not worth it for 2 people)

- Cross-branch knowledge sharing between tree-search branches — real complexity, marginal payoff
- Counterfactual/OPE evaluation — advanced, high build cost, only attempt if everything else is done with time to spare
- Live replay viewer / dashboard — a 10-line matplotlib plot in the final hour is enough; logs already contain everything
- Anomaly/surprise detector on top of the fixed diagnosis enum — a whole extra subsystem, too much for 2 people alongside the core

---

## 8. How this wins, mapped to the rubric

| Category | Weight | What wins it |
|---|---|---|
| Technical Execution | 35% | Real positive delta over baseline 0.5946 (F1) + demonstrated robustness across the full 50-iteration run (F3) + a validated, never-rejected submission (F8) |
| Innovation & Problem Insight | 20% | Structured, evidence-backed diagnosis (F2) naming real recsys phenomena, anchored against popularity baseline and ceiling (F9) |
| Impact & Relevance (Autonomy) | 20% | Structurally provable zero-intervention run (F4) |
| Feasibility & Practicality | 15% | Validated cheap-test methodology (F5) + hard cost cap (F6), landing comfortably in the low/medium bucket, reported with real wall-clock and token numbers — but only scored at all once baseline is beaten |
| Presentation | 10% (finals) | Agent-vs-human story + specific, log-derived "next steps" + a readable rendered run-log table in the report |

**The strategic insight underneath all of it:** most hackathon teams will pour nearly all their time into Technical Execution (the model) and treat logging, cost tracking, and diagnosis quality as an afterthought. Those "boring" pieces are 65% of the grade combined, and F1–F9 are all comparatively cheap to build well. A team with a *merely decent* model but airtight, provable plumbing beats a team with a slightly better model and sloppy everything else.

---

## 9. Build order for 2 people

1. Baseline reproduction using the actual Starter Kit (`python3 baseline.py --model fm`) — confirm we land near primary 0.5946/0.6016 (test/val) before touching anything else
2. Bulletproof loop: subprocess + short timeout + per-iteration checkpointing + failure typing (F3) — needed before you can trust anything to run unattended
3. Logging schema: `agent_state.json` + heartbeats + seeds + the plateau/best-ever dual-tracker from section 3 (F4, F7) — build once, used everywhere after
4. Trick menu + diagnosis rules written as config, targeting `long_view` not click (F1)
5. Structured Stage 6 JSON schema, anchored evidence against popularity/ceiling (F2, F9)
6. Dev-mode cheap test + one-time validation check (F5)
7. Dead man's switch on tokens + wall-clock (F6)
8. Wire `submit.py --check` into the promotion step (F8)
9. Let it run unattended through the 50-iteration/convergence cap, iterate on what breaks
10. Write-up: results table anchored against the 0.4753–0.8645 range, rendered run-log table, cost numbers, "what we'd do with more time" pulled from real logs, optional agent-vs-human comparison
