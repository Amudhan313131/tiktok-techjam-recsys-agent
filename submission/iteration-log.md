# Autonomous iteration log

This is the concise reviewer-facing form of the sealed raw log
`source-report/iteration_logs.json`. The raw JSON remains the authoritative
record and has SHA-256
`76fff049a58a464b14078050ef858c7e045bf7ff9bb4da6c70031e80b822466f`.

## Run-level recovery event

During iteration 1, the Docker rehearsal intentionally terminated the active
controller after it had durably leased an E15 cheap-rung worker. REX detected
the stale owner, resumed the same run under the original deadline, recovered
the leased work, preserved the incumbent and experiment counters, and finished
the run. The final manifest records `recovered-and-complete`; no human edit,
retry, or model-selection intervention was made.

Manual interventions for the run: **0**.

## Iteration 1 — E15 mean context-FM ensemble

**Hypothesis.** Changing only five-member context-aware FM aggregation from
median to mean will improve primary by at least 0.001 over the matched control.
Arithmetic-mean aggregation should retain and average probability variation
across all five stochastic FM initializations.

**Applied change / code diff.** Versioned configuration transaction only; no
source-code patch. Set `aggregation: mean` while preserving
`ensemble_members=5`, `epochs=7`, the plugin, and every other model setting.
Effective-config SHA-256:
`2075f8dbdb83e62cd602cf89c893c3ff2fae17ba85f590b30036318147ce6c45`.

| Rung | Fold | GAUC | nDCG@5 | Primary |
|---|---|---:|---:|---:|
| Cheap shadow | A-cheap | 0.636515 | 0.556094 | 0.596305 |
| Full shadow | A | 0.656683 | 0.552171 | 0.604427 |
| Full shadow | B | 0.661124 | 0.533268 | 0.597196 |
| Full shadow | C | 0.658840 | 0.492884 | 0.575862 |
| Official validation | — | **0.670204** | **0.537110** | **0.603657** |

**Decision.** Promoted. E15 became the validation champion, improving primary
by `+0.002057` over the official 0.6016 baseline. The controlled controller
failure was recovered automatically; the experiment itself recorded no model
failure.

## Iteration 2 — E01 same-user PairLogit

**Hypothesis.** Replacing pointwise BCE with same-user fixed-K PairLogit while
holding the FM architecture, features, ensemble, and training settings fixed
will improve primary by at least 0.001. Pairwise optimization should align the
loss more closely with within-user ranking.

**Applied change / code diff.** Versioned configuration transaction only; no
source-code patch. Replace pointwise BCE with same-user fixed-K PairLogit and
change no other scientific variable. Effective-config SHA-256:
`bfbdf06eac247a0d14b136668614bdd3a27c284b2159c17e27afd319cfb09b44`.

| Rung | Fold | GAUC | nDCG@5 | Primary |
|---|---|---:|---:|---:|
| Cheap shadow | A-cheap | 0.635488 | 0.554137 | 0.594813 |
| Full shadow | A | 0.631843 | 0.541968 | 0.586905 |
| Full shadow | B | 0.627622 | 0.524104 | 0.575863 |
| Full shadow | C | 0.623732 | 0.484074 | 0.553903 |

**Decision.** Rejected by the full temporal evidence gate. No official
validation token was consumed for E01. No error or repair event occurred.

## Iteration 3 — E02 point-in-time tree statistics

**Hypothesis.** Enabling only point-in-time item and author target statistics
in grouped LightGBM LambdaRank will improve primary by at least 0.001 over its
matched no-stat control. Strictly historical rates should expose nonlinear
propensity signals without changing groups or tree settings.

**Applied change / code diff.** Versioned configuration transaction only; no
source-code patch. Toggle the point-in-time item/author target-statistics branch
from disabled to enabled and hold every other scientific variable fixed.
Effective-config SHA-256:
`d1dfad359b7d3e4a5b6c05d63029a3ba9d719096391454b119b94ccfd31c3995`.

| Rung | Fold | GAUC | nDCG@5 | Primary |
|---|---|---:|---:|---:|
| Cheap shadow | A-cheap | 0.624167 | 0.550726 | 0.587446 |
| Full shadow | A | 0.650903 | 0.551195 | 0.601049 |
| Full shadow | B | 0.651300 | 0.530861 | 0.591081 |
| Full shadow | C | 0.650917 | 0.490993 | 0.570955 |
| Official validation | — | 0.662334 | 0.533414 | 0.597874 |

**Decision.** Rejected because official validation did not safely improve the
incumbent. E15 remained immutable. No error or repair event occurred.

## Stop decision

After the three scored iterations, the configured cumulative
`epsilon=0.002`, `N=3` rule declared an epsilon plateau. The validation-best E15
checkpoint—not the last attempted model—was selected for final prediction.

