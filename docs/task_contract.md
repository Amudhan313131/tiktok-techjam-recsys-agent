# Authoritative KuaiRand-Pure Task Contract

This file is the short runtime-facing contract. The organizer starter kit is authoritative if prose elsewhere conflicts.

- Dataset: KuaiRand-Pure.
- Task: rank each user's logged impressions; this is not full-catalog retrieval.
- Label: native `long_view`.
- Train split boundary: 2022-04-08 through 2022-04-21; the official archive's
  first observed standard-log row is 2022-04-09. Total: 1,141,112 rows.
- Validation: 2022-04-22 through 2022-04-28, 124,909 rows.
- Hidden test: 2022-04-29 through 2022-05-08, 170,588 rows.
- Metrics: GAUC and nDCG@5; primary is their arithmetic mean.
- Official FM validation: GAUC 0.6674, nDCG@5 0.5357, primary 0.6016.
- Published hidden-test FM: GAUC 0.6610, nDCG@5 0.5282, primary 0.5946.
- Caps: 50 hypotheses/evaluations, epsilon 0.002, patience 3, six-hour wall clock.
- Submission: `row_id,user_id,video_id,score`, exactly aligned to split order.
- Current-row outcomes are targets, never inference features.
- Development processes may not read or score hidden-test labels.
- The organizer starter files, split manifest, evaluator, budget, firewall, and submission gate are protected from autonomous edits.

The obsolete click/nDCG@10/Recall@50 text is not part of this task. The participant solution may use PyTorch, LightGBM, GPU compute, papers, public code, and allowed pretrained weights.
