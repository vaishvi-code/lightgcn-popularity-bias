# LightGCN-single user and item analysis

## Question and comparison

Does using only the last propagated layer change recommendation quality for different user histories, and how does it change the relationship between item popularity, embedding norms and exposure?

Compare **LightGCN-single** with **standard uniform layer averaging** at K=1ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“4, using the same existing validation-selected checkpoints and seeds 2020ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“2022. This is 24 checkpoint evaluations, with no retraining or new checkpoint selection. LightGCN-single is a final-layer-only architecture choice, not a synonym for K=1.

The frozen roster, checkpoint hashes, configurations, data hashes and saved validation references are in [the analysis plan](../provenance/single-users-items-plan.json). This extension is post-hoc, after earlier results were viewed.

## Evaluation

- **Split:** validation only. Do not describe these results as additional test evaluations. No new test labels are opened.
- **Users:** reuse the existing fitting-history thirds: mostly popular, mixed, mostly tail. Training interactions determine the assignments and popular/tail item labels. These describe observed histories, not true preferences.
- **Group metrics:** overall Recall@20 and NDCG@20, Tail Recall@20, tail share, coverage and exposure Gini. Tail Recall averages only over users with held-out tail positives.
- **Items:** Spearman correlations for popularity versus final embedding norm, learned pre-propagation norm, and recommendation frequency. Include final norm versus exposure and Pearson norm/degree diagnostics for continuity with the original analysis.
- **Controls:** same catalog, fitting graph, top-20 ranking and exclusion of fitting interactions. Uniform and single models are separately trained and paired by depth and seed.
- **Verification:** exact checkpoint and data hashes, reproduction of saved overall validation metrics within absolute tolerance 1e-5, exposure-count L1 tolerance 20 across all recommendation slots, and exact reconciliation of grouped exposures and user denominators within each rerun. Save deviations explicitly rather than changing earlier results.

Three-seed variation is descriptive, not a user-level confidence interval or significance test. Item correlations do not establish causation. Exposure and tail relevance should be interpreted together.

## Status and results

Completed November 23, 2026 as ARC job **346542** on **c2** after shared-storage access recovered. All 24 model evaluations, the independent audit and figure generation passed. The exported results were independently audited again after download.

Read [the completed result report](../results/gowalla/single-users-items-job346542/README.md), including all user-group tables, item correlations, verification records and PNG/SVG figures. No new test evaluation was performed.

Initial attempts stopped during storage/reference preflight or an overly strict duplicate-output check. The completed run includes the teammate's shared checkpoint roots, the canonical frozen popularity-file bytes, and direct reuse of the single-output final tensor for its identical last-layer norm. All reported results come from the completed run.

The source is staged on ARC under `/home/vpatel34/lightgcn-single-users-items-20261122`. The launch script uses the existing environment and original checkpoint locations. Local boundary checks and synthetic artifact corruption checks passed. The audit verifies pooled group metrics, seed summaries, item exports and reference reproduction. The source and plan bytes matching the exported audit hashes are preserved in the result directory's `inference-sources.tar.gz`; editable repository copies may have Git-normalized line endings.

## Outputs

The launch script writes a new `results/single-users-items-job<jobid>/` directory in its ARC source directory. It produces:

- `group-profiles.csv`: aggregate validation user counts and tail-recall denominators.
- `user-groups-per-run.csv` and `user-groups-summary.csv`: all 72 grouped rows and three-seed summaries.
- `item-correlations-per-run.csv` and `item-correlations-summary.csv`: 24 checkpoint diagnostics and summaries.
- One compressed item diagnostic table per checkpoint. These contain item IDs, degrees, norms and aggregate exposure counts, with no user IDs or user interaction histories.
- Per-checkpoint verification records and `audit.json`.
- PNG and SVG figures for user-history outcomes and item correlations.

No per-user assignments or recommendation lists are exported. Previous published results remain separate from this analysis.

The exported results can be checked without a GPU:

```bash
python scripts/audit_single_users_items.py <result-directory>
```

Plots need matplotlib:

```bash
python scripts/plot_single_users_items.py <result-directory>
```
