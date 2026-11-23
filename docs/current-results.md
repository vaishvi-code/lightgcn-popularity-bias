# Current results and proposal coverage

This guide consolidates the completed Gowalla study for team review. It distinguishes evidence from the original test comparison, later validation replications, and post-hoc analyses. Additional datasets are outside the agreed scope.

## Four original questions

| Proposal question | Completed experiments | Finding | Evidence limits |
| --- | --- | --- | --- |
| Layers: does more propagation favor popular items, and does layer averaging help? | Standard LightGCN K=1–4 and MF across three seeds on validation/test; separately trained single-output K=1–4 across three seeds on validation, seed2020 on supplemental test | More standard depth concentrates exposure. Averaging gives higher tail recall than final-layer-only at every depth in every validation seed. | Higher concentration can accompany better overall recall. Single-output test remains one seed. |
| Weights: do APPNP-style weights change recommendations? | Beta=0.1/0.2/0.5 now have three-seed validation; beta=0.5 has three-seed test, other betas have one-seed test | Beta=0.5 has the largest mean tail recall among tested weights, at an overall accuracy cost relative to uniform K=3. Small beta is not automatically better. | Original beta=0.5 replication used validation selection. Newly added seeds were planned after original test access. |
| Normalization: does stronger down-weighting improve tail recommendations? | Nine seed-2020 candidates: a={0,0.25,0.5} for each b={0.5,0.75,1.0}; selected settings replicated across three seeds on validation | Selected a is 0.5 for b=0.5 and 0.25 for b=0.75/1.0. Tuning improves overall recall over earlier fixed-a variants, but the selected alternatives still trail standard K=3 in mean recall, tail recall and tail share. | Limited grid, selected on one seed/split. Tuned settings have no new test evaluation. Earlier fixed-a test remains one seed. |
| Why: do embedding norms increasingly track degree? | Fifteen frozen MF/K=1–4 checkpoints across three seeds; degree/norm and saved validation-exposure diagnostics | Norms positively correlate with degree; correlation does not consistently increase with depth. This does not support the simple proposed explanation. | Association does not prove the causal scoring mechanism. Norms are only part of a dot-product score. |

## Additional completed analyses

- [User groups](../results/gowalla/user-groups-job340789/README.md): fitting-history groups across all 18 original main checkpoints. More depth offers little mean overall-recall benefit beyond K=2 for mostly-tail users while reducing tail recovery. APPNP beta=0.5 trades overall recall for tail recovery. These users do not have the lowest absolute accuracy; compare changes within groups.
- [Supplemental single-output test](../results/gowalla/supplemental-single-test-job340785/README.md): four separately frozen seed-2020 checkpoints, verified on validation before testing. Preserved as a post-hoc one-seed test comparison.
- [New follow-up](../results/gowalla/proposal-completion-20261114/README.md): 22 new runs; all histories, settings, metrics, exposures and logs are included. An expanded comparison covers 14 settings × 3 seeds on validation. One beta=0.1 run reached the common 500-epoch budget; do not claim every run converged.

## What is complete, and what is not

The [LightGCN-single user/item extension](../results/gowalla/single-users-items-job346542/README.md) is complete for depths 1â€“4 and three seeds, with matched uniform-averaging controls on validation. Averaging has higher tail recall, tail share and coverage in every matched seed/depth/user group. Single-output overall and tail recall decrease with each depth in all groups and seeds. Item normâ€“popularity associations remain nonmonotonic. These new post-hoc **validation** findings are separate from the earlier test analysis.

All four proposal questions have now been investigated on Gowalla. The planned search over a and additional-seed validation replications are complete. Existing original/test results are unchanged.

Three-seed **test** performance is not available for newly replicated single-output or APPNP beta=0.1/0.2 variants. Tuned a=0.25 normalizations have no test evaluation. The earlier fixed-a normalizations have only seed-2020 validation/test. Treat these as evidence limits, not as completed three-seed test experiments. No new test runs were authorized by the validation follow-up protocol.

The work does not establish significance, performance on other datasets, true personal preferences, causal unfairness, or online improvements in visits/revenue. Re-ranking is an optional future extension and has not been run.

## Review without ARC

Start with [the full comparison](../results/gowalla/proposal-completion-20261114/comparison.md), then [the run index](../results/gowalla/proposal-completion-20261114/runs.md). All linked configurations, histories, validation metrics, per-item exposure counts and logs are stored in GitHub. The aggregate audit/rebuild commands use standard Python and do not need an ARC account. Checkpoint inference requires weights, which are not included in the new export.

Original references: [proposal PDF](proposal.pdf), [frozen main test](../results/gowalla/final-test-job337725/README.md), [Q4 diagnostics](../results/gowalla/embedding-diagnostics-job340254/README.md), [new frozen follow-up protocol](../provenance/proposal-completion-plan.json).
