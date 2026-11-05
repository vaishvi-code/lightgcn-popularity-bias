# Experiment protocol and parallel work

## Independent work tracks

Both tracks use the completed standard three-layer run as a shared reference. Use separate branches and ARC checkouts so model changes can be developed independently.

Vaishvi will begin her track first and publish the resulting configurations, histories, metrics, and exposure tables for the teammate's handoff. The teammate can then use the same split and evaluation protocol for the weighting and normalization track.

| Owner | Experiments | Deliverables |
| --- | --- | --- |
| Vaishvi | BPR matrix factorization with zero propagation layers, standard LightGCN with 1–4 layers, and LightGCN-single with 1–4 layers | Depth comparison and item degree/embedding norm/exposure diagnostics |
| Teammate | Three-layer APPNP-style weights with beta 0.1, 0.2, and 0.5, then three-layer normalization with a=0.5 and b=0.75 or 1.0 | Weighting and normalization comparisons against the standard three-layer reference |

Test APPNP weights with standard normalization. Test normalization with standard uniform layer weights. Reuse the standard K=3, seed-2020 result where settings match.

Use seed 2020, a 500-epoch maximum, and early-stopping patience 20 for the current comparisons, selecting by validation Recall@20. Repeat comparisons with training seeds 2021 and 2022. Keep the split seed fixed at 2020. If training duration or stopping rules change, apply the same rule to the compared baselines.

Optional validation-based early stopping is available through `--early-stopping-patience 20`. It stops after 20 consecutive epochs without a strictly higher validation Recall@20. Record the selected rule before running comparisons. When using it, rerun standard baselines under the same rule. The completed historical 200-epoch run used no early stopping. Its continuation and the MF comparison have completed under the 500-epoch maximum and patience 20. The continuation restored the model and optimizer but restarted sampling because the legacy checkpoint lacked random-generator states.

The current training and evaluation scripts support standard LightGCN with configurable depth and BPR-MF as the zero-layer case. LightGCN-single, APPNP-style weights, and normalization variants still need implementation. Use the same variant in training and checkpoint evaluation. For normalization changes, include the normalization exponents in graph-cache paths. The current cache paths distinguish data splits only.

The ARC MF script trains and evaluates a best checkpoint using a 500-epoch maximum and patience 20. Set `MAX_EPOCHS=2` for its short check. Each Slurm job has a separate output directory. See [the MF submission commands](../README.md#matrix-factorization-comparison).

The remaining seed-2020 standard layer-depth runs have been submitted to ARC using `scripts/train_layers_arc.sh`.

| Layers | Slurm job | ARC result directory under `results/gowalla/` |
| --- | --- | --- |
| 1 | 327365 | `lightgcn-k1-seed2020-job327365/` |
| 2 | 327366 | `lightgcn-k2-seed2020-job327366/` |
| 4 | 327367 | `lightgcn-k4-seed2020-job327367/` |

Each run uses the same fitting split, popularity groups, seed, embedding size, optimizer settings, 500-epoch maximum, and patience 20. The job evaluates its best checkpoint automatically after training. Completed metrics still need collection before adding these runs to the comparison table. The completed K=3 continuation provides the current three-layer result, with its documented legacy random-state limitation.

Save each run's configuration, history, selected-checkpoint metrics, and item exposure table under a unique directory in `results/gowalla/`. Include variant settings and seed in the run name. Use [the metrics template](../results/metrics-template.csv) for the combined comparison.

## Shared experimental protocol


The split, popularity grouping, MostPopular evaluation, standard LightGCN training and exposure evaluation, and zero-layer BPR-MF path are implemented. The remaining model variants are planned.

### Data split and ranking

- Preserve the supplied test split for final evaluation. Hold out 10% of each user's original training interactions for validation, keeping at least one fitting interaction per user. Document rounding and sparse-user handling.
- Remove duplicate edges and verify that fitting, validation, and test interactions are disjoint. Build the graph using fitting interactions only.
- Compute item degrees from fitting interactions. Define popular items as the top 20% by degree; use deterministic item-ID tie breaking and document rounding and zero-degree handling. Keep groups fixed across models.
- Use the same split, eligible catalog, and evaluation users for every comparison. Rank the full eligible catalog in batches.
- Exclude fitting positives during validation; exclude fitting and validation positives during final testing.
- Select checkpoints and hyperparameters using validation data. Use test data only after the comparison plan is frozen.
- Report the additional validation holdout when comparing against published results: training data differs from the original paper's setup.

### Model comparisons

| Model or variant | Configuration |
| --- | --- |
| MostPopular | Rank by fitting item degree with per-user exclusions |
| BPR matrix factorization | Zero propagation layers |
| Standard LightGCN | 1–4 layers, uniform combination including layer 0 |
| LightGCN-single | 1–4 layers, final propagated embedding only |
| APPNP-style weighting | 3 layers; beta in {0.1, 0.2, 0.5} |
| Degree normalization | Start with a=0.5; b in {0.5, 0.75, 1.0} |

For APPNP-style weighting, use `alpha_k = beta * (1-beta)^k` for `k < K` and `alpha_K = (1-beta)^K`. Verify that weights sum to one.

For normalization, use `1 / (degree(node)^a * degree(neighbor)^b)` in both aggregation directions. Expand the `a` grid only with a documented validation-selection rule. Changing exponents affects propagation scale as well as degree weighting; discuss this when interpreting outcomes.

### Metrics

- Accuracy: Recall@20 and NDCG@20.
- Exposure: tail share@20, catalog coverage@20, and exposure Gini.
- Tail relevance: tail Recall@20.
- Diagnostics: correlations involving training item degree, embedding norm, and recommendation count.

Compute Recall@20 and NDCG@20 as per-user macro averages over users with held-out positives. Tail share is the fraction of recommendation slots occupied by tail items. Coverage is the fraction of the fixed eligible catalog recommended at least once. Calculate exposure Gini over all eligible items, including those with zero recommendations.

For tail Recall@20, use held-out tail positives as each user's denominator; average only over users with at least one tail positive and report that user count. Validate metrics against hand-computed examples. Greater tail exposure alone does not establish better relevance or fairness.

Use Spearman correlations for the degree/norm and degree/exposure diagnostics. Correlations support analysis but do not establish causality.

Gowalla is the primary dataset. Selected Yelp2018 and Amazon-Book experiments follow only after the main pipeline is reliable. Final reported setups target three training seeds, with actual seed counts disclosed for preliminary runs.

### Reproducibility

Use comparable optimization and validation checkpoint-selection rules. Save the complete configuration, code and upstream commits, split seed/hash, training seed, software versions, hardware, runtime, and best validation epoch with every run. Keep checkpoints and raw logs locally or in shared cluster storage.

## Navigation

- [Setup and commands](../README.md#getting-started)
- [Completed results and artifact index](../results/README.md)
- [Reference baseline configuration](../configs/gowalla-baseline.json)
