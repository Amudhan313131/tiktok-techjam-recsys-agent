# Final results and resource report

## Selected immutable run

| Field | Value |
|---|---|
| Dataset | KuaiRand-Pure |
| Run ID | `r3-docker-20260831-codex-v15` |
| Winning experiment | `r3-docker-20260831-codex-v15-e15` |
| Run source commit | `6b68f0ca4506bbd2bac52a8c9285d0fb1366946e` |
| Winner experiment commit | `451c1e5b7d091607bac59b4a77f3b6c16fe18e69` |
| Docker image digest | `sha256:be5e5f218be7b26c3c6cd3313ed382293a5b32ac9a8c8b0e45e6c7adcf978062` |
| Stop reason | `epsilon_plateau` |
| Hidden test scored locally | No |
| Bonus datasets attempted | None |

## Validation metrics

Primary is the arithmetic mean of GAUC and nDCG@5.

| Result | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Organizer official baseline | 0.667400 | 0.535700 | 0.601600 |
| Run-verified strongest baseline seed | 0.667948 | 0.536126 | 0.602037 |
| V15 validation-best E15 | 0.670204 | 0.537110 | 0.603657 |
| Delta over official baseline | **+0.002804** | **+0.001410** | **+0.002057** |
| Delta over run-verified baseline | +0.002256 | +0.000983 | +0.001620 |

All rows above are validation results. The published hidden-test baseline must
not be compared directly with these validation numbers.

## Winning configuration

```yaml
plugin: rex.models.experimental.context_fm:ExperimentalContextEnsembleFMPlugin
k: 16
lr: 0.001
l2: 0.000001
epochs: 7
batch_size: 8192
ensemble_members: 5
aggregation: mean
```

The five ensemble members use seeds 0, 1, 2, 3, and 4. Their scores are
combined by arithmetic mean.

## Resource usage to convergence

| Resource | Recorded usage |
|---|---:|
| Agent wall-clock | 1,484.200829 s (24 min 44.2 s) |
| Complete Docker envelope | 1,517.367295 s (25 min 17.4 s) |
| Iterations | 3 / 50 |
| LLM calls | 11 |
| LLM input tokens | 289,862 |
| LLM output tokens | 20,949 |
| Total LLM tokens | **310,811** |
| GPU-hours | **0.0** |
| Manual interventions | **0** |

The run deliberately injected one controller failure after a durable worker
lease. Recovery resumed the same run and finished without a manual
intervention.

## Final submission validation

| Check | Result |
|---|---|
| Expected test rows | 170,588 |
| CSV columns | `row_id,user_id,video_id,score` |
| First organizer format/alignment check | Passed |
| Second check against the copied CSV | Passed |
| NaN/Inf accepted | No |
| Hidden test scoring command run locally | No |
| Submission CSV SHA-256 | `9765476f68f3f4eaf87c6f33bc4d55c844dac9f946c72fe2fa8ef6acc85e4b8c` |

