# Post-hoc supplemental test completion

**Completed:** [test results, accuracy-cost table, user-group analysis and figures](../results/gowalla/extended-test-job346586/README.md). Evaluation job 346586 and report job 346593 completed successfully. All 24 validation gates, 18 new test evaluations and 6 repeated test checks passed. The combined comparison now covers all 14 main settings across three test seeds.

The team authorized this extension on November 23, 2026 after reviewing the original proposal against the completed study. Original test results have already been viewed. This is a descriptive post-hoc extension, not a new blind test, and does not amend the earlier validation-only protocol.

## Frozen scope

The [machine-readable plan](../provenance/extended-test-plan.json) records checkpoint hashes, configurations, selected epochs, validation references, data hashes, evaluator-source hashes and ranking rules before execution.

There are **18 first-time test evaluations**:

- Eight LightGCN-single checkpoints: depths 1-4, seeds 2021 and 2022.
- Four APPNP checkpoints: beta=0.1 and 0.2, seeds 2021 and 2022.
- Six selected-normalization checkpoints: a=0.25 with b=0.75 or 1, seeds 2020-2022.

Six existing seed-2020 test checkpoints (four single-output and two APPNP) are evaluated again to collect matched user-group metrics. Their overall results must reproduce the earlier reference within the frozen numerical tolerance. Previous published results are preserved.

Combining the 18 newly tested checkpoints with the 24 previously tested matched checkpoints gives **14 settings x 3 seeds = 42 results**. Historical fixed-a normalization variants remain separately labeled exploratory results. MostPopular remains a separate deterministic baseline.

No model is retrained, retuned, selected again or given additional epochs. In particular, the budget-limited beta=0.1 seed2021 checkpoint remains the same best-validation checkpoint at epoch 498. Its 500-epoch cap is disclosed rather than selectively changing the comparison.

## Verification before test access

The evaluator first verifies all 24 extension checkpoints on validation. Only a complete identity-matching verification gate permits its test phase. The gate checks the frozen plan, canonical source hashes, checkpoint hashes, split hashes, all six validation metrics and catalog-wide recommendation counts.

Use the same fitting-only graph, fitting-degree item popularity groups, original pinned Gowalla test file, full-catalog top-20 ranking, and exclusion of fitting plus validation interactions. Numerical tolerances are absolute 1e-5 for reference metric reproduction and exposure-count L1 difference at most 20 across 597,160 recommendation slots. Save every deviation. Do not relax these thresholds after viewing the new outcomes.

Source hashes normalize CRLF to LF; the plan hash uses canonical JSON serialization. Data and checkpoint checksums use exact bytes.

## User groups and exports

Groups reuse fitting-history tail fractions: mostly popular at or below one third, mostly tail at or above two thirds, mixed otherwise. They describe observed histories, not true preferences. Use the same six metrics within groups, averaging tail recall only over users with held-out tail positives. Group counts, weighted user metrics and item exposures must reconcile with the overall rerun.

Reuse the original main-model user-group results for 18 checkpoints. Add the 24 extension checkpoints' grouped outcomes for a complete matched comparison. Keep the historical test summaries and new rerankings' origins explicit. Export only aggregate group statistics and item exposure counts, not individual user assignments, interactions or recommendation lists.

## Reporting

Report three-seed mean and sample SD. Keep validation, original test and post-hoc supplemental test chronology visible. New results cannot establish statistical significance, a causal popularity mechanism, fairness or performance on other datasets.

The report should directly quantify changes in overall recall, tail recall and exposure relative to the frozen standard three-layer control. Include the original fitting-data difference from the published LightGCN setup and the budget-limited run among the limitations. Use the clearer Q4 wording recorded in [the findings guide](current-results.md).

## Execution

`scripts/evaluate_extended_test_arc.sh` runs the gate tests, complete validation verification, and then test inference in an isolated home export directory. Checkpoints remain in their existing ARC locations. Every submission writes a fresh result directory. The reproducible plan builder is `scripts/build_extended_test_plan.py`.

Initial job 346585 stopped during validation verification because the evaluator expected a uniform-combination metadata label for APPNP checkpoints. Before any new test access, that check was corrected to expect `appnp` (older metadata without the field uses the known family), and the source hashes were frozen again. The roster, checkpoints, ranking rules and numerical thresholds did not change.
