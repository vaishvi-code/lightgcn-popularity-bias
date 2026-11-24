# Proposal completion: additional seeds and normalization search

**Completed and audited:** [results, conclusions and figures](../results/gowalla/proposal-completion-20261114/README.md), [fourteen-model validation table](../results/gowalla/proposal-completion-20261114/comparison.md), and [all new run files](../results/gowalla/proposal-completion-20261114/runs.md). Teammates can inspect and audit the exported runs without ARC. The results remain validation-only.

This follow-up was planned after the original final-test results were available. It uses fitting/validation data only and does not authorize any new original-test evaluation. Earlier frozen comparisons remain unchanged. Multiple datasets are outside the current scope.

The machine-readable protocol is `provenance/proposal-completion-plan.json`. Training code and source hashes, fitting/validation hashes, existing reference measurements, task roster and selection rule are recorded there before submission.

## Normalization search

At K=3 and training seed 2020, test a in {0, 0.25, 0.5} for each b in {0.5, 0.75, 1.0}. Reuse the existing a=0.5 checkpoint for each b, requiring six new fits. Select a separately for each b by highest best-checkpoint validation Recall@20. Exact ties prefer a closest to 0.5, then smaller a. Tail metrics and test outcomes do not select settings.

After all six runs finish and validate, write the selection record before submitting any selected-setting replications. Keep that same selected a for seeds 2021 and 2022; do not reselect it per seed. If a=b=0.5 is selected, reuse the existing matched standard K=3 runs at seeds 2021/2022. Otherwise submit new runs. This requires at most six more fits.

This is a limited grid, not an exhaustive normalization optimization. The published original default-a comparison is preserved separately. The selected a is conditional on one validation split and seed 2020.

## Independent replication

Run LightGCN-single K=1,2,3,4 for seeds 2021 and 2022 (eight runs) and APPNP beta=0.1/0.2 at K=3 for seeds 2021 and 2022 (four runs). Combine with existing seed-2020 validation measurements only after all required runs finish.

Every new run uses 64-dimensional embeddings, BPR, Adam lr=0.001, regularization=0.0001, batch size 1024, at most 500 epochs, patience 20, and the first strictly highest validation Recall@20 checkpoint. Fitting graph, item groups and validation exclusions match prior runs. The existing evaluator checks reloaded checkpoint Recall/NDCG before exporting all six validation metrics.

## ARC execution and dependencies

Use an isolated BeeGFS workspace `/mnt/beegfs/$USER/lightgcn-proposal-completion-20261112`, source/export workspace `$HOME/lightgcn-proposal-completion-20261112`, existing lightgcn environment and pinned upstream checkout. Old runs are never overwritten. Raw weights stay on ARC.

1. `prepare`: copy fitting/validation files and graph cache, verify hashes and pinned code, and run normalization/single-output numerical checks.
2. Search array: six tasks, maximum two concurrent. Independent replication array: twelve tasks, maximum two concurrent. Both depend on successful preparation.
3. Selection controller: depends on successful completion of the entire search array. It verifies candidate metrics, freezes the selected settings, and submits up to six replication tasks, maximum two concurrent.
4. Summary controller: depends on successful independent and selected-setting replications. It exports validation per-run and three-seed mean/sample-SD tables for nine settings: four single-output depths, two APPNP betas and three selected normalizations.

The pipeline uses at most four GPUs concurrently. Each fit requests six hours on rtx4060ti16g, excluding c0/c23 because of prior storage issues. A failed dependency stops downstream work rather than selecting from partial results. Scheduling depends on cluster availability.

### Submitted execution

The [submission receipt](../provenance/proposal-completion-submission.json) records preparation job **340803**, normalization-search array **340804** (six tasks), independent-replication array **340805** (twelve tasks), and selection controller **340806**. All completed successfully. The controller selected a=0.5 for b=0.5 and a=0.25 for b=0.75/1.0. Array **340847** completed four new selected-setting replications; two standard K=3 references were reused. Summary job **340848** completed. Twenty-one fits stopped by patience; APPNP beta=0.1 seed2021 reached the 500-epoch cap and selected epoch 498.

After successful search completion, the controller writes `state/normalization-selection.json`, optionally `state/selected-submission.json`, and `state/summary-submission.json` in the home workspace. If any training task fails, inspect its log and repair or resubmit that task before replacing the corresponding downstream dependency. Do not manually select from a partial grid.

`scripts/proposal_completion.py` implements the phases; `proposal_completion_arc.sh` runs array tasks; `proposal_completion_control_arc.sh` runs preparation, selection or summary. It does not load the original `data/gowalla/test.txt`. Each completed run gets a follow-up provenance sidecar with plan identity and its task parameters; source checks guard copied training code.

Run artifacts were exported without `.pt` files to the home workspace, with per-task records in `state/`, then downloaded and independently audited. They are versioned under `results/gowalla/proposal-completion-20261114/`, including scheduler logs and configuration/history/metric/exposure files for all 22 new runs. Raw weights remain on ARC. New test inference is outside this protocol.

## Proposal-versus-results review: November 23

This review uses the original [proposal](proposal.pdf), the [current evidence guide](current-results.md), and the [new single-output user/item analysis](../results/gowalla/single-users-items-job346542/README.md). All four questions have been investigated on Gowalla. That does not mean every detail of the original evaluation plan has been fulfilled.

| Original commitment | What is available | Assessment |
| --- | --- | --- |
| Standard and final-layer-only LightGCN at depths 1-4 | All settings have three-seed validation. Standard models have three-seed test; single-output models have seed-2020 test only. | Research question covered; uniform test coverage is incomplete. |
| APPNP-style beta=0.1, 0.2 and 0.5 at depth three | All have three-seed validation. Beta=0.5 has three-seed test; beta=0.1/0.2 have seed-2020 test only. | Research question covered; smaller-beta test replication remains incomplete. |
| Normalization b=0.5, 0.75 and 1, selecting a on validation | Nine seed-2020 candidates and three-seed validation of selected settings. Selected a=0.25 alternatives have no test evaluation. | Search completed. Earlier a=0.5 test variants do not substitute for tests of the selected a=0.25 settings. |
| MF and original LightGCN controls | Three-seed validation/test, with MostPopular as an additional deterministic control. | Completed. |
| Norm/popularity and degree/exposure correlations | MF and standard depth diagnostics, plus matched single/uniform item diagnostics. | Completed and extended. The original strengthening-with-depth hypothesis is not supported consistently. |
| Recall, NDCG, tail share, coverage, Gini and tail recall | All metrics are exported for the applicable validation/test evaluations. | Completed with explicit split and seed labels. |
| Three random seeds per setup, with final numbers on test | Fourteen matched three-seed validation settings. Only the original main families have full three-seed test coverage. | Partially complete at the final-test level. Search candidates and historical exploratory fixed-a variants are not required to become three-seed main comparisons. |
| Validation for selection and early stopping | Fixed fitting/validation split, patience 20, common 500-epoch cap, selected checkpoints and recorded histories. | Implemented. One beta=0.1 run reached the cap with its best checkpoint near the end; convergence is not established for that run. The proposal cited approximately 1,000 epochs as paper context, rather than specifying that every run must train exactly that long. |
| First reproduce the published three-layer Gowalla score | A controlled reproduction holds out validation interactions and fits on fewer interactions than the authors' original training split. | Do not claim exact published-score reproduction. Explain the setup difference rather than treating the published score as a matched benchmark. |
| Additional datasets | Not run, by the team's subsequently agreed scope. | Deliberate scope reduction, not a recommended new task. |

### Already beyond the proposal

The original proposal did not require the observed-history user groups, their matched single-output/averaging analysis, the additional MostPopular control, or the complete export audits. These additions strengthen interpretation and reproducibility without changing the original hypotheses.

### Most useful remaining additions

1. **Complete the missing supplemental test coverage if another experiment is desired.** Eight single-output seed-2021/2022 evaluations, four beta=0.1/0.2 seed-2021/2022 evaluations, and six selected a=0.25 normalization evaluations would complete the fourteen-setting, three-seed comparison on test. These are 18 additional checkpoint evaluations with no new fitting. They require a separate frozen post-hoc protocol and authorization: this validation-follow-up plan does not authorize them. Reuse existing checkpoints, settings, grouping rules and exclusions, and disclose that the test set has already been viewed. New user-group test summaries could be collected during the same inference if fixed in that protocol.
2. **Answer the proposal's accuracy-cost question directly with existing data.** Relative to standard K=3, K=2 loses 0.4202 percentage points of mean test Recall@20 and gains 0.2572 points of Tail Recall@20 and 1.6407 points of tail share. APPNP beta=0.5 loses 1.3797 points of overall recall and gains 0.3529 points of tail recall and 3.3228 points of tail share. These are descriptive changes in the existing three-seed frozen test results, not new model selection or a significance claim.
3. **Optionally check the budget-limited run.** A separately labeled continuation of beta=0.1 seed2021 could examine whether the 500-epoch cap affects that setting's validation conclusions. Keep the original common-budget comparison intact, verify resumable state first, and do not selectively replace its result while presenting it as the unchanged original protocol. This is less important than transparent reporting of the cap.

No additional dataset, new architecture, causal mechanism claim or online deployment experiment is necessary to report the completed Gowalla study. In the final report, distinguish completed research questions, evidence-coverage gaps, and deliberate scope changes.
