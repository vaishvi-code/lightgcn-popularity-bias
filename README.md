# Does LightGCN Favor Popular Items?

CSC 591/791-038: Machine Learning for Graphs — Group 5, Fall 2026.

Team: Sachi Vyas, Sakhi Patel, and Vaishvi Patel.

We study how LightGCN's propagation depth, layer weights, and degree normalization affect recommendation accuracy and exposure for less popular items. This repository starts as a project scaffold; model training and results are not implemented yet.

## Research questions

1. How does increasing the number of propagation layers affect popularity exposure?
2. Does combining layers behave differently from using only the final layer?
3. How do APPNP-style layer weights change accuracy and tail exposure?
4. What trade-offs arise from stronger degree-based neighbor down-weighting?
5. Do item embedding norms and recommendation frequencies correlate with item degree?

This is a controlled reproduction and ablation study. Popularity amplification in graph recommenders is already studied in prior work; we do not claim to discover it for the first time.

## Getting started

1. Obtain Gowalla from the author-linked [LightGCN PyTorch repository](https://github.com/gusye1234/LightGCN-PyTorch). Follow its licensing and data-use terms; do not commit the dataset here.
2. Establish a working Python/PyTorch environment on the team's ARC or NC State HPC cluster. Record the exact environment once verified.
3. Implement the data split, metrics, MostPopular baseline, and LightGCN pipeline in that order.
4. Use [the baseline configuration](configs/gowalla-baseline.json) as the initial experiment specification. It is not yet connected to a training command.
5. Record completed runs in `results/`, following [the results format](results/README.md).

No runnable training command or measured result is supplied yet. Dependencies and scheduler scripts should be added after confirming the cluster environment and job scheduler.

## Repository layout

```text
configs/       Experiment specifications
data/          Local datasets, excluded from Git
notebooks/     Exploratory analysis
results/       Results schema and small reviewable tables
scripts/       Future data preparation, training, evaluation, and plotting commands
src/           Future project implementation
tests/         Future meaningful metric and data-split checks
```

The [project proposal](CSC%20591%28038%29_Group_5_proposal.pdf) is retained at the repository root. Datasets, credentials, checkpoints, and raw run artifacts are excluded from version control.

## Experimental protocol

These choices describe the intended implementation; no experiments have been run in this repository yet.

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

## Results

No measured results are available yet. [The metrics template](results/metrics-template.csv) contains column names only. Commit actual result tables and final plots after evaluation, identifying the dataset, split, seed count, and training status. Keep published scores separate from team measurements.

## References

- He et al. [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126), 2020.
- Wu et al. [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153), 2019.
- Gasteiger et al. [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/abs/1810.05997), 2019.
- Chen et al. [How Graph Convolutions Amplify Popularity Bias for Recommendation?](https://arxiv.org/abs/2305.14886), 2023 preprint.

External code must retain its original attribution and license. No license for redistribution of this team's original work has been selected yet.
