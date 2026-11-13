# Proposal completion: additional seeds and normalization search

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

The [submission receipt](../provenance/proposal-completion-submission.json) records preparation job **340803** (passed), normalization-search array **340804** (six tasks), independent-replication array **340805** (twelve tasks), and selection controller **340806** (depends on the search array). The first two search and first two replication tasks started successfully. Training completion and final settings are still pending; this is a submission record, not a results report.

After successful search completion, the controller writes `state/normalization-selection.json`, optionally `state/selected-submission.json`, and `state/summary-submission.json` in the home workspace. If any training task fails, inspect its log and repair or resubmit that task before replacing the corresponding downstream dependency. Do not manually select from a partial grid.

`scripts/proposal_completion.py` implements the phases; `proposal_completion_arc.sh` runs array tasks; `proposal_completion_control_arc.sh` runs preparation, selection or summary. It does not load the original `data/gowalla/test.txt`. Each completed run gets a follow-up provenance sidecar with plan identity and its task parameters; source checks guard copied training code.

Run artifacts are exported without `.pt` files to the home workspace, with per-task records in `state/`. The final tables go to `results/proposal-completion/` there. After completion, review and audit these artifacts before adding them to GitHub. Do not describe pending jobs as completed results.
