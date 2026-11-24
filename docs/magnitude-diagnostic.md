# Item embedding magnitude and ranking sensitivity

## Question

How much do recommendation rankings depend on item embedding magnitude? This is a follow-up to Q4, not a replacement of the original hypothesis that the popularity-magnitude association would strengthen with depth.

The [frozen plan](../provenance/magnitude-plan.json) uses nine existing best-validation checkpoints: MF, standard LightGCN K=3 and LightGCN-single K=3, each with seeds 2020-2022. No training, tuning, additional epoch budget, checkpoint selection or test evaluation is authorized. Earlier test results have already been viewed, so the diagnostic is explicitly post-hoc and validation-only.

## Intervention

An original score is the dot product of the learned final user vector and learned final item vector. It combines user-vector magnitude, item-vector magnitude and directional alignment. For a given user, user-vector magnitude is constant across candidate items.

The intervention divides each final item vector by its mathematical length (L2 norm), keeping its direction and the user vector fixed. Nonzero vectors above epsilon therefore have unit length. Exact-zero vectors remain zero; lengths below the frozen epsilon of 1e-12 use the epsilon denominator. Counts of zero and tiny vectors are reported.

This changes scoring after training. It does not change the number of embedding dimensions, graph degree normalization, learned parameters, graph or candidate catalog. With nonzero vectors, ranking by the intervention score is equivalent to cosine-similarity ranking for each user, since the unchanged user magnitude is a positive constant.

## Scoring controls

All nine original sigmoid-based validation evaluations must pass checkpoint, source, data, metric and exposure verification before intervention inference begins. The legacy reference tolerance is absolute 1e-5 for metrics and exposure-count L1 difference at most 20 across all recommendation slots.

The primary comparison uses **raw dot product versus raw dot product with unit-length item vectors**. Both omit sigmoid, preventing numerical sigmoid saturation or ties from being confused with magnitude effects. Sigmoid is monotonic in exact arithmetic, but finite precision can affect rankings. The analyze phase also repeats sigmoid scoring on the same embeddings and measures its ranking difference from the raw baseline.

There are nine paired primary comparisons, with three inference scoring modes per checkpoint, plus the separate nine-checkpoint verification phase. This is inference only, not new model training.

## Measurements

- Overall Recall@20, NDCG@20, Tail Recall@20, tail share, coverage and exposure Gini on the fixed validation split.
- Mean top-20 overlap and Jaccard similarity, changed top-1/set/ordered-list fractions, and positional agreement against the raw-dot baseline.
- Aggregate outcomes within the existing fitting-history user groups.
- The mean original norm and fitting degree of recommended items, with catalog-wide original item norms saved for verification.

Use the same fitting-only graph, fixed fitting-degree popularity labels, full catalog, top 20 and fitting-interaction exclusions. Export aggregate group statistics and item-level counts only. Do not export user IDs, assignments, histories or individual recommendation lists.

## Interpretation

Changes show sensitivity to this specific inference intervention. They do not establish that embedding magnitude caused popularity bias during training. Directions and magnitudes were learned jointly under the original objective, and unit-length scoring changes that objective's scoring rule. No change would establish only limited sensitivity under this tested intervention, rather than proving that directional alignment is the sole explanation.

Treat relevance and exposure together: more tail exposure does not automatically mean more relevant tail recommendations. Three-seed mean and sample SD are descriptive, not significance tests. Keep this exploratory validation diagnostic separate from the completed frozen test comparison.

## Execution

`scripts/analyze_magnitude_arc.sh` runs hand-built vector tests, the complete legacy verification gate, and then all fixed scoring modes in a fresh export directory. The report should include both the ranking-overlap measurements and accuracy/exposure changes, even if the intervention harms accuracy.

## Completed results (November 24, 2026)

[Results, interpretation and reproducibility](../results/gowalla/magnitude-job347014/README.md). ARC inference job 347014 and reporting job 347032 completed successfully. All nine original validation gates passed before intervention inference. Equal-length scoring increased tail recall and tail share while lowering overall recall in every matched seed for all three models. This is a fixed-checkpoint validation sensitivity analysis, not a trained replacement or a new test evaluation.
