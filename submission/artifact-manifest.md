# Final artifact manifest

## Submission identity

| Field | Value |
|---|---|
| Source run | `r3-docker-20260831-codex-v15` |
| Winner | `r3-docker-20260831-codex-v15-e15` |
| Submission job | `submission-8bd5ba7e055f36828e1a` |
| Winner commit | `451c1e5b7d091607bac59b4a77f3b6c16fe18e69` |
| Configuration SHA-256 | `2075f8dbdb83e62cd602cf89c893c3ff2fae17ba85f590b30036318147ce6c45` |
| Best-valid manifest SHA-256 | `b3524aece2a85c534e8f004e0107212ce9955eb535d456161cd46bd3b34fdf5f` |
| Source report SHA-256 | `5bfad7ef731b8857d85fc75de21ca0fc71fd06fe1384a8d570bbcea0358baf51` |
| Submission seal SHA-256 | `fa02b4829d317cfe95952be6795a73fa9678bf8a238c9fad3c1e1587bffd6b62` |
| Upload ZIP | `tiktok-techjam-submission-r3-docker-20260831-codex-v15.zip` |
| Upload ZIP size | 90,413,158 bytes |
| Upload ZIP SHA-256 | `8e678c1c34df75116b3fd0ad84f9cd388ff0f0e36a42f8956f8cfec7ed9c218f` |
| Handoff status | `HANDED_OFF` / verified |
| Test scored locally | `false` |

## Upload files

All paths below are relative to the sealed handoff root.

| Path | Size (bytes) | SHA-256 |
|---|---:|---|
| `submission.csv` | 6,482,257 | `9765476f68f3f4eaf87c6f33bc4d55c844dac9f946c72fe2fa8ef6acc85e4b8c` |
| `test_predictions.npz` | 1,559,106 | `91bb9e1c70eddc840e2b5ddedf1b4678b67f848f71e605a3879c36e4a08e058e` |
| `best-valid/config.yaml` | 249 | `2075f8dbdb83e62cd602cf89c893c3ff2fae17ba85f590b30036318147ce6c45` |
| `best-valid/model/model_bundle.json` | 2,038 | `9a28f7e52299301506842834fae12aefa94c8c342e81fbb6b4ddd2f8324b2075` |
| `best-valid/model/model-000.npz` | 2,527,339 | `cf7a30ade1cb7453af8533784f4a832ef0ec6a0dff13d0b675bfe49c254b73c5` |
| `source-report/iteration_logs.json` | 580,661 | `76fff049a58a464b14078050ef858c7e045bf7ff9bb4da6c70031e80b822466f` |
| `source-report/manual_interventions.json` | 425 | `f46563249296b2c9d3db76aa25d8e68763ae10d05f19d52cb63ad6867b2c5409` |
| `source-report/resources.json` | 813 | `c6303c982e22d9a95793c87d6c2c204172d2557016ba5b62be0e0dded8c37b45` |
| `source-report/results.json` | 1,230 | `d8511955a9b713c9ab6b4aedd0df7981b95221931316d332b1da572ceb794718` |
| `checks/first.json` | 735 | `a356699bf5ba06a94c99e5a0623bfb7e585c45966aa2035bdb3ef38696a6e9f9` |
| `checks/second.json` | 743 | `7a6d61ccf469054414e2ad01bae143f0830bb84eaa0fc24b38094282c933d3c6` |
| `final_results_summary.json` | 5,939 | `c9286eb97fa46090111f42ddb0874c95356f5ab83bc7baadf8d19adeebb7b861` |

`model-000.npz` is the primary checkpoint, but the model is a five-member
ensemble. Upload the entire `best-valid/model/` directory, not only the primary
member. `submission_seal.json` inventories and hashes every member and evidence
file.

## Validation evidence

The organizer checker ran twice and returned success for exactly 170,588 rows.
Both check transcripts record `test_scored: false`. The copied handoff tree was
recursively validated against the exact seal after finalization.
