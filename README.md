# Autonomous ML Research Agent — Recommender Systems (KuaiRand-Pure)

An LLM-driven agent that autonomously reproduces the official KuaiRand-Pure
baseline and then iterates to beat it, using a recsys-specific "trick menu,"
structured diagnosis, cheap-before-expensive testing, and a bulletproof
execution loop — while structurally proving zero human intervention.

**Full product spec: [`docs/spec.md`](docs/spec.md)** — read this first. It
has the corrected task facts (target label, metrics, reference scores), the
locked convergence/checkpoint decisions, and the build order.

## The task, in short

- Target label: `long_view` (NOT click)
- Metrics: `GAUC` + `nDCG@5`, averaged into `primary`
- Reference scores (hidden test): random=0.4753, popularity-only=0.5715,
  **official baseline=0.5946** (beat this), ceiling=0.8645 (not 1.0)
- CPU-only, numpy-only — no GPU needed anywhere
- Hard caps: 50 iterations, ε=0.002/N=3 convergence rule, 6h wall-clock

## Project layout

```
recsys-agent/
├── docs/
│   └── spec.md               # the full product spec — read this first
├── agent/
│   ├── orchestrator.py        # main loop: Stage 3 -> 4 -> 5 -> 6 -> repeat
│   ├── llm_client.py          # wraps Claude API calls (reasoning + structured output)
│   ├── state.py                # agent_state.json: dual tracker (plateau vs best-ever), heartbeats
│   ├── trick_menu.yaml         # recsys moves + diagnosis rules (the domain knowledge)
│   ├── schemas/
│   │   └── reflect_schema.json     # forced JSON schema for Stage 6 diagnosis
│   ├── budget_guard.py         # dead man's switch + the official 50-iter/6h caps
│   ├── run_training_subprocess.py  # bulletproof subprocess + timeout wrapper
│   └── submission_validator.py     # F8: gates every new-best candidate before it's trusted
├── training/
│   ├── train.py                # subprocess-callable training script (--dev_mode, checkpointing)
│   ├── models/                 # TODO: real model, starting from the organizer's FM baseline
│   └── data/
│       └── loader.py           # KuaiRand-Pure loading + recsys-specific EDA
├── scripts/
│   └── run_agent.sh            # entrypoint to kick off an unattended run
├── configs/
│   └── budget.yaml             # the exact official caps (50 iter, ε=0.002/N=3, 6h)
└── logs/                        # agent_state.json + per-iteration run logs land here
```

## Setup

1. Download `kuairand-starter-kit.zip` from the challenge page and extract
   the CSVs into `data/kuairand_pure/` (see `training/data/loader.py` for
   the expected filenames).
2. (Optional but recommended) Place the organizer's `submit.py` at
   `kuairand_starter_kit/submit.py` so `agent/submission_validator.py` uses
   the real official check instead of the local fallback.
3.
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

## Running

```bash
bash scripts/run_agent.sh            # real run
bash scripts/run_agent.sh --dry-run  # exercise the harness with fake data/scores, no API calls
```

This runs unattended until convergence, the 50-iteration cap, or the 6h
wall-clock cap — whichever comes first. Check `logs/agent_state.json` for
live progress.

## Monitoring an unattended run

- `logs/agent_state.json` — `best_ever_primary_score` (what will be
  submitted), `iteration_count`, `converged`, `human_override_count`, heartbeat
- `logs/iterations/iter_NNN.json` — one file per iteration: hypothesis, metrics
  (primary/GAUC/nDCG@5), diagnosis, submission validation result

## Build status

- [x] Harness plumbing tested (state, convergence, budget guard, submission validator)
- [ ] Baseline reproduction (organizer's `baseline.py --model fm`)
- [ ] Real model in `training/models/`
- [ ] Real GAUC/nDCG@5 scoring wired in (`training/train.py` currently returns `None`)
- [ ] Stage 3 move-parsing (currently a TODO in `orchestrator.py`)
- [ ] First unattended full run through convergence
