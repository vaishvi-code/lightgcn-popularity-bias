# Current results and proposal coverage

This guide consolidates the completed Gowalla study for team review. It distinguishes evidence from the original test comparison, later validation replications, and post-hoc analyses. Additional datasets are outside the agreed scope.

## Four original questions

| Proposal question | Completed experiments | Finding | Evidence limits |
| --- | --- | --- | --- |
| Layers: does more propagation favor popular items, and does layer averaging help? | MF, standard LightGCN and final-layer-only K=1-4 now have three-seed validation and test coverage. Supplemental tests are post-hoc. | Averaging gives higher global tail recall, tail share and coverage at every depth in every seed. Single-output test overall recall peaks numerically at K=2, while tail recall falls with depth. | One of 36 grouped test tail-recall comparisons favors single output; all group means favor averaging. No significance claim. |
| Weights: do APPNP-style weights change recommendations? | Beta=0.1/0.2/0.5 have three-seed validation and test results, with the later test additions labeled post-hoc. | Beta=0.5 offers higher tail recovery at an overall accuracy cost. Beta=0.1/0.2 trail standard K=3 on overall recall, tail recall and tail share in each test seed. | Beta=0.5 replication was chosen using seed-2020 validation. One beta=0.1 fit reached the common epoch cap. |
| Normalization: does stronger down-weighting improve tail recommendations? | Nine-candidate seed-2020 validation search, then three-seed validation and post-hoc test for selected a=0.25, b=0.75/1.0 settings. | Both selected alternatives have lower overall recall, tail recall and tail share than standard K=3 in every test seed. | Limited grid and selection on one validation split/seed. Earlier fixed-a=0.5 exploratory test results remain one seed and separate. |
| Why: do embedding norms increasingly track degree? | Fifteen frozen MF/K=1–4 checkpoints across three seeds; degree/norm and saved validation-exposure diagnostics | Norms positively correlate with degree; correlation does not consistently increase with depth. This does not support the simple proposed explanation. | Association does not prove the causal scoring mechanism. Norms are only part of a dot-product score. |

## Additional completed analyses

- [Item magnitude intervention](../results/gowalla/magnitude-job347014/README.md): nine fixed MF/standard K=3/final-only K=3 checkpoints, validation only. Equal-length item scoring raised tail recall and tail share but reduced overall recall in all nine seed/model comparisons. Ranking changes were substantial, especially for final-only output. This is inference sensitivity, not a causal explanation of training.

- [Full supplemental test completion](../results/gowalla/extended-test-job346586/README.md): all 14 main settings now have three-seed test summaries, with user-group results, fixed-checkpoint provenance and explicit post-hoc labeling.

- [User groups](../results/gowalla/user-groups-job340789/README.md): fitting-history groups across all 18 original main checkpoints. More depth offers little mean overall-recall benefit beyond K=2 for mostly-tail users while reducing tail recovery. APPNP beta=0.5 trades overall recall for tail recovery. These users do not have the lowest absolute accuracy; compare changes within groups.
- [Supplemental single-output test](../results/gowalla/supplemental-single-test-job340785/README.md): four separately frozen seed-2020 checkpoints, verified on validation before testing. Preserved as a post-hoc one-seed test comparison.
- [New follow-up](../results/gowalla/proposal-completion-20261114/README.md): 22 new runs; all histories, settings, metrics, exposures and logs are included. An expanded comparison covers 14 settings × 3 seeds on validation. One beta=0.1 run reached the common 500-epoch budget; do not claim every run converged.

## What is complete, and what is not

The [LightGCN-single user/item extension](../results/gowalla/single-users-items-job346542/README.md) remains the completed three-seed validation diagnostic. The [completed supplemental test comparison](../results/gowalla/extended-test-job346586/README.md) now adds matched three-seed test and user-group evidence for all 14 main settings. Averaging has higher tail recall in 35 of 36 matched test group/depth/seed comparisons and higher tail share and coverage in all 36. Validation and test outcomes remain separately labeled.

All four proposal questions have been investigated on Gowalla. The normalization search, additional-seed validation replications and missing supplemental test evaluations are complete. Existing original test values are preserved. The report must disclose the fitting-data difference from the original paper, post-hoc test chronology and budget-limited run.

Three-seed test coverage is now complete for all 14 matched main settings, through a separately frozen post-hoc extension. This does not change the earlier validation-only protocol or turn the new results into a blind test. Search candidates not selected for the main comparison and earlier fixed-a=0.5 exploratory normalizations remain separately labeled. The common-budget beta=0.1 seed2021 run is still budget-limited.

The work does not establish significance, performance on other datasets, true personal preferences, causal unfairness, or online improvements in visits/revenue. Re-ranking is an optional future extension and has not been run.

## Q4 framing for the final report

Use the clearer question: **Do popular items have larger embedding magnitudes, and how does this relationship change when we add graph layers?**

Define embedding magnitude before the figure: it is the vector's mathematical length (L2 norm), not the number of embedding dimensions. All compared models use the same embedding dimension. Define popularity as the number of fitting interactions and recommendation frequency as the number of validation recommendation slots containing an item.

Keep the original proposal hypothesis explicit: we expected the popularity-magnitude association to strengthen with depth. The broader question is an explanatory restatement, not a new hypothesis chosen after viewing results. Report the finding that final magnitudes and popularity are positively associated, but the relationship does not consistently strengthen with depth in either aggregation family.

Explain final-output magnitude and learned pre-propagation magnitude separately. The latter uses trained parameters, not untouched random initialization, and each depth is a separately trained model. Put recommendation-frequency correlations in a secondary diagnostic panel, and explain that rank correlation and head/tail exposure shares answer different questions. A dot-product score depends on vector direction as well as magnitude, so these correlations do not establish why popularity concentration occurs.

Introduce Spearman correlation in plain language as agreement between rankings. Lead with the question and finding, then explain each plotted line separately. Keep formulas and detailed coefficients in the methods or appendix rather than using them to introduce the question.

The [new magnitude intervention](../results/gowalla/magnitude-job347014/README.md) adds a direct scoring sensitivity check to Q4. Holding learned directions and user vectors fixed, equal-length item vectors increase tail recovery and exposure but reduce overall recall in each of the nine checkpoints. Standard K=3 retains about a third of its original top-20 items; final-only K=3 retains less than a twentieth. State that magnitude matters for these fixed rankings, while the model was trained with dot products and the intervention changes its scoring rule. Do not describe this as proof that magnitude alone caused the observed popularity patterns.

## Review without ARC

Start with [the completed test comparison](../results/gowalla/extended-test-job346586/README.md) and [accuracy-cost table](../results/gowalla/extended-test-job346586/accuracy-tradeoffs.md), then the [validation comparison](../results/gowalla/proposal-completion-20261114/comparison.md) and [training run index](../results/gowalla/proposal-completion-20261114/runs.md). Aggregate audits and table reconstruction do not require ARC. Checkpoint inference requires weights and the pinned data.

Original references: [proposal PDF](proposal.pdf), [frozen main test](../results/gowalla/final-test-job337725/README.md), [Q4 diagnostics](../results/gowalla/embedding-diagnostics-job340254/README.md), [new frozen follow-up protocol](../provenance/proposal-completion-plan.json).
