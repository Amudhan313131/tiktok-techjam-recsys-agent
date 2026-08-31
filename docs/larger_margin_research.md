# Larger-margin KuaiRand-Pure research

This document describes the current larger-margin research path. Its purpose is
to discover a model that transfers across time, not to manufacture a single
high validation number. The supplied validation reference is `0.6016`; the
current engineering target is `0.613+`, with `0.615` as a stretch target. Those
numbers are goals, not achieved results.

## Safety boundary

The research path observes these rules:

- Discovery uses only chronological shadow folds built from the training
  period. The cheap rung is a deterministic, stratified sample of complete
  users; it is not a random sample of rows.
- Candidate and matched control see aligned rows and are scored by the same
  organizer-compatible evaluator.
- The official validation split is locked during discovery. A finalist can
  consume the run's single official-validation token only after the discovery
  gates pass. The transition from `FINALIST_LOCKED` to
  `OFFICIAL_EVALUATED` is durable and cannot be replayed as tuning feedback.
- Hidden-test labels are never loaded or scored. Test prediction remains a
  separate, explicitly authorized release job after a completed run.
- Every direct or autonomous result preserves its config, predictions, model
  bundle, metrics, hashes, source identity, and `test_scored: false` evidence.
  Failed runs are kept under their original run ID and are never overwritten.

## Data capabilities

Bootstrap now creates three distinct capability classes.

### Inference-safe feature views

Train, validation, and test feature views contain the identifiers and context
needed at prediction time, plus two stable temporal keys:

- `time_ms`, normalized as a non-negative 64-bit integer;
- `source_row_key`, a globally unique source-row identity used to make ordering
  deterministic when timestamps are equal.

The basic user and video side tables are joined by ID. Categorical values are
fit on training data with explicit unknown behavior, and the transform and
source hashes are included in the data manifest. The allow-listed item block
contains video/upload/music type, tag, aspect, upload age, and duration
consistency. The allow-listed user block contains coarse activity, creator and
streamer state, relationship-count ranges, and registration-age ranges. Numeric
user quantities are stored as bounded `log1p` values for models that support
numeric inputs.

The month-level `video_features_statistic_pure.csv` table is explicitly
forbidden. Its outcome-derived aggregate fields cannot be smuggled into a safe
view under engineered aliases.

### Primary target vault

`long_view` for train and validation is stored outside the feature views. No
test target path is created. The ordinary model interface receives a target
capability only for an authorized training or trusted evaluation operation.

### Auxiliary feedback vault

The bootstrap also writes a capability-separated train/validation vault for
`is_click`, `is_like`, `is_follow`, `is_hate`, and `long_view`. There is no test
feedback artifact. Historical recipes may use these values only for events
strictly earlier than the row being featurized. Rows with equal `time_ms` are
processed as one atomic group, so no row can observe another outcome from the
same timestamp. Apply views use a state frozen at the training cutoff and never
read apply-split outcomes.

## Larger-margin method cards

The autonomous queue retains earlier controls, but its new discovery block is
E16-E30:

| Card | One isolated question | Matched comparison |
|---|---|---|
| E16 | Does increasing ensemble membership help? | five-member mean versus one-member mean |
| E17 | Does a train-fitted `user×tab` cross help? | same context FM without that cross |
| E18 | Does a train-fitted `video×tab` cross help? | same context FM without that cross |
| E19 | Does the inference-safe item metadata block help? | same five-member context FM without item metadata |
| E20 | Does coarse user metadata add value after item metadata? | E19 item-metadata model |
| E21 | Do learned field-pair weights beat uniform FM interactions? | same-field FM control |
| E22 | Do candidate-conditioned point-in-time recency features help? | same regularized tree without that recipe |
| E23 | Do strictly historical auxiliary-feedback summaries help? | same regularized tree without that recipe |
| E24 | Does the regularized `rank_xendcg` tree repair the old tree branch? | old no-stat tree control |
| E25 | Do two supported, diverse families blend beneficially? | stronger component, with weight selected on shadow OOF predictions only |
| E26 | Does an item-aware pointwise classifier beat its core-field control? | same LightGBM classifier without item metadata |
| E27 | Does the complete static metadata block help the pointwise tree? | core-field pointwise tree |
| E28 | Do categorical recency buckets help the item-metadata FM? | E19 item-metadata FM |
| E29 | Does user-only metadata help the pointwise tree? | core-field pointwise tree |
| E30 | Does increasing FM latent dimension from 16 to 32 help? | E19 item-metadata FM |

E25 is derived from complete aligned A/B/C prediction evidence; its weight is
not selected on official validation. It remains prerequisite-gated until two
diverse families are scientifically supported.

Search is family-aware. The old three-transaction epsilon plateau cannot stop
the run until at least six valid model/feature families have been evaluated.
The remaining limits are still 50 hypotheses/evaluations, the external
six-hour ceiling, and a one-hour finalization reserve.

## Direct shadow research

The direct runner is a reproducible development tool for E16-E24 and E26-E30. It never
opens official validation and never predicts test rows. First bootstrap the
current views, then run a new output ID:

```bash
.venv/bin/python -m rex.cli bootstrap \
  --data-dir data/KuaiRand-Pure/data \
  --output-dir runs/data

.venv/bin/python scripts/run_direct_shadow_research.py \
  --config configs/run/production.yaml \
  --output-dir /absolute/path/to/direct-research \
  --run-id direct-cheap-001 \
  --rung cheap \
  E16 E17 E18 E19 E20 E21 E22 E23 E24 E26 E27 E28 E29 E30
```

Promising cards should then receive a fresh full-rung run rather than reusing
or overwriting the cheap output:

```bash
.venv/bin/python scripts/run_direct_shadow_research.py \
  --config configs/run/production.yaml \
  --output-dir /absolute/path/to/direct-research \
  --run-id direct-full-e19-001 \
  --rung full \
  E19
```

For release-quality evidence, run from a clean committed source. The report
records `SOURCE_COMMIT-dirty` when implementation files are uncommitted, which
is useful during development but cannot be promoted directly to an immutable
Docker winner.

## Current direct evidence

The table below is development evidence from seed 0. It did not use official
validation or hidden test. The current reports were produced while the source
tree was being implemented, so they guide the next run but do not establish a
final champion.

| Card and rung | Mean primary delta | Fold support | Interpretation |
|---|---:|---:|---|
| Earlier E16 cheap screen | `+0.000596` | 1/1 cheap fold | Mean beat median on the cheap sample. The production E16 card now separately tests five members against one while holding mean aggregation fixed. |
| E17 cheap | `+0.000258` | 1/1 cheap fold | Weak signal, below a larger-margin promotion threshold. |
| E18 cheap | `-0.004054` | 0/1 | Rejected. The video-by-tab cross was harmful. |
| E19 full | `+0.001281` | 3/3 | Strongest current discovery signal; item metadata transferred across every temporal fold. |
| E20 full | `+0.000073` | 2/3 | Effectively flat; fold C regressed `-0.000409`. Do not stack the user block by default. |
| E21 cheap | `-0.000005` | 0/1 | Flat to slightly negative; reject the current FwFM setting. |
| E22 cheap | `-0.004947` | 0/1 | Rejected. Current candidate-recency/tree combination is harmful. |
| E23 cheap | `-0.001457` | 0/1 | Rejected. Current historical-feedback/tree combination is harmful. |
| E24 cheap | `+0.011132` versus old tree | 1/1 | Tree repair is real relative to its weak control, but absolute primary `0.593655` remains below the context-FM family. |
| E26 cheap | `-0.000536` | 0/1 | Rejected. Item metadata alone hurt the pointwise tree. |
| E27 full | `+0.000666` | 3/3 | The complete metadata tree improved both components on every fold versus its item-tree parent; the clean card now compares it with the core tree. |
| E28 full | `+0.000100` | 2/3 | Rejected. Fold C regressed `-0.001118`, and one component regressed on fold A. |
| E29 cheap | `-0.000974` | 0/1 | Rejected. User metadata alone did not explain E27. |
| E30 full | `-0.000666` | 0/3 | Rejected. The k=32 FM looked positive cheaply but lost on every full fold. |
| E19/E27 shadow blend | `+0.000510` over E19 | 2/3 | A regularized 70/30 blend improved folds A and C and slightly regressed B; the immutable run must reselect its own weight. |

For E19, the per-fold primary deltas were `+0.000302`, `+0.001801`, and
`+0.001740` on A, B, and C. That consistency makes E19 the current finalist
candidate for multi-seed confirmation. It does **not** mean the project has
reached `0.613`, and a shadow delta cannot simply be added to an unrelated
official-validation score.

The reports are preserved under `runs/direct-research/` in the development
workspace. Each report links the exact candidate/control bundles and prediction
hashes. A fresh clone should rely on newly generated evidence, not assume local
run directories were committed.

## Autonomous Docker workflow

After tests pass, commit the implementation intentionally. Build an immutable
image whose source label matches that exact clean commit, run the production
doctor with the chosen live researcher, and start a new versioned output
directory. The standard launcher is documented in
[`docker-production.md`](docker-production.md); a representative run is:

```bash
python3 scripts/run_docker_rehearsal.py start \
  --source-root "$PWD" \
  --data-dir "$PWD/data/KuaiRand-Pure/data" \
  --output-dir /absolute/path/to/rex-docker-r3-vNEXT \
  --run-id rex-docker-r3-vNEXT \
  --image rex:COMMIT_TAG \
  --llm codex_cli \
  --codex-home "$HOME/.codex"
```

The controller performs validation-only research. It explores eligible
families, rejects weak ideas at the cheap rung, uses all three shadow folds for
promising ideas, and locks one finalist before the sole official-validation
evaluation. Candidate/control work may run in parallel, but evidence is joined
in canonical fold order.

Monitoring is read-only and does not require an LLM call. A scheduled check can
run every 30 minutes without changing the experiment:

```bash
python3 scripts/run_docker_rehearsal.py status \
  --output-dir /absolute/path/to/rex-docker-r3-vNEXT
```

While the run is `INITIALIZING`, `RUNNING`, or `RECOVERING`, do not edit source,
rebuild its image, restart a healthy controller, predict test rows, or generate
a submission. If it fails, preserve the complete directory, diagnose from its
database/manifests/logs, repair in a new commit, and launch a new run ID and
image. If it completes without a genuinely supported improvement, preserve the
evidence and begin the next research cycle with one isolated, evidence-backed
change.

Only a completed, sealed winner can enter the separately authorized
`finalize-submission` flow. That flow verifies exactly 170,588 aligned,
finite test predictions and runs the organizer's format checker; it never runs
local test scoring.

## Promotion criteria

The `0.613+` goal is intentionally difficult. A credible result should be a
three-seed mean, improve at least two of three temporal folds, avoid a material
regression in either GAUC or nDCG@5, have low seed variation, and pass paired
user-bootstrap uncertainty checks. Official validation is a final atomic check,
not another model-selection surface. Until those conditions are met, the
truthful statement is that E19 is a promising shadow-fold treatment and the
previous sealed validation champion remains unchanged.
