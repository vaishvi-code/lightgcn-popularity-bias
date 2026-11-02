# Does LightGCN Favor Popular Items?

CSC 591/791-038: Machine Learning for Graphs — Group 5, Fall 2026.

Team: Sachi Vyas, Sakhi Patel, and Vaishvi Patel.

We study how LightGCN's propagation depth, layer weights, and degree normalization affect recommendation accuracy and exposure for less popular items. The data pipeline, MostPopular baseline, and a 200-epoch three-layer LightGCN run have completed on ARC. The selected LightGCN checkpoint was evaluated for accuracy and popularity exposure on validation data. Results use one training seed and remain preliminary research findings.

## Research questions

1. How does increasing the number of propagation layers affect popularity exposure?
2. Does combining layers behave differently from using only the final layer?
3. How do APPNP-style layer weights change accuracy and tail exposure?
4. What trade-offs arise from stronger degree-based neighbor down-weighting?
5. Do item embedding norms and recommendation frequencies correlate with item degree?

This is a controlled reproduction and ablation study. Popularity amplification in graph recommenders is already studied in prior work; we do not claim to discover it for the first time.

## Getting started

Use Python 3.10. The observed ARC environment used Python 3.10.22, PyTorch 2.5.1+cu121, and an RTX 4060 Ti with 16 GB GPU memory.

From the repository root, install the CUDA-enabled wheel, followed by the recorded direct dependencies:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Download the author-linked upstream code and processed datasets into the ignored checkout directory. Follow its applicable licensing and data-use terms.

```bash
git clone --depth 1 https://github.com/gusye1234/LightGCN-PyTorch.git LightGCN-PyTorch
python -m pip freeze > environment-freeze.txt
git -C LightGCN-PyTorch rev-parse HEAD > upstream-commit.txt
python scripts/prepare_gowalla.py
python scripts/analyze_popularity.py
python scripts/evaluate_mostpopular.py
```

The upstream commit and complete environment freeze used on ARC still need to be transferred; `requirements.txt` records observed direct dependencies rather than a full lockfile. Keep the commit identifiers for reproductions.

On an allocated GPU node, run a two-epoch check:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u scripts/train_validation.py \
  --dataset gowalla --model lgn --layer 3 --recdim 64 \
  --bpr_batch 1024 --lr 0.001 --decay 0.0001 --epochs 2 \
  --seed 2020 --topks "[20]" --testbatch 256 --tensorboard 0 --multicore 0
```

For ARC batch training, set `PROJECT_ROOT` to this repository's absolute cluster path. Submit from your home filesystem:

```bash
export PROJECT_ROOT="/absolute/path/to/lightgcn-popularity-bias"
cd "$HOME"
sbatch --output="$HOME/lightgcn-k3-%j.out" "$PROJECT_ROOT/scripts/train_arc.sh"
```

The batch script assumes the already-created environment at `/mnt/beegfs/$USER/conda/envs/lightgcn`. Adapt that path for another environment. It starts a fresh 200-epoch run; resume and early stopping are not implemented. The recorded baseline run has completed; submitting this script starts a new run.

The training wrapper supplies fitting interactions and validation interactions to the upstream loader in a separate hash-keyed directory. Its `test.txt` is validation data; the supplied final test set is not evaluated. Graph caches are isolated by split hashes. Each run saves `config.json`, `history.csv`, `best.pt` (selected by validation Recall@20), and `last.pt`.

To evaluate a trusted checkpoint on validation data, set `RUN_DIR` to its directory and run on an allocated GPU node:

```bash
python scripts/evaluate_lightgcn.py --run-dir "$RUN_DIR"
```

This writes `best-validation-metrics.json` and `best-validation-exposure.csv` inside that run. Checkpoints remain on ARC; they are not stored on GitHub. The evaluator verifies the split hashes and uses the saved popular/tail item groups.


The repository scripts were recovered from the team's shared terminal logs and adapted to live under `scripts/`. ARC's existing root-level scripts and checkpoints were not changed by this GitHub sync.

## Repository layout

```text
configs/       Experiment specifications
data/          Local datasets, excluded from Git
notebooks/     Exploratory analysis
results/       Completed baseline metrics and preliminary LightGCN snapshot
scripts/       Data preparation, popularity analysis, baseline evaluation, training, and ARC submission
src/           Future project implementation
tests/         Future meaningful metric and data-split checks
```

The [project proposal](CSC%20591%28038%29_Group_5_proposal.pdf) is retained at the repository root. Datasets, credentials, checkpoints, and raw run artifacts are excluded from version control.

## Experimental protocol

The split, popularity grouping, MostPopular evaluation, and standard three-layer LightGCN training and exposure evaluation are implemented. The remaining ablation variants are planned.

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

Results below use the fitting/validation split with seed 2020 and 29,858 validation users. The final test set remains untouched.

| Model | Validation Recall@20 | Validation NDCG@20 | Tail share@20 | Coverage@20 | Exposure Gini |
| --- | ---: | ---: | ---: | ---: | ---: |
| MostPopular (completed) | 0.02715 | 0.01298 | 0% | 0.0903% | 0.99950 |
| LightGCN K=3 (best epoch 192 of 200, one seed) | 0.18968 | 0.09853 | 10.59% | 35.50% | 0.93371 |

The most popular 8,197 of 40,981 items account for 53.09% of the 737,091 fitting interactions. MostPopular recommends only 37 distinct items and has tail Recall@20 of zero among 18,987 users with validation tail positives.

LightGCN recommends 14,549 distinct items and has tail Recall@20 of 0.04615 among the same 18,987 users with validation tail positives. Popular items occupy 89.41% of its recommendation slots. It improves accuracy and catalog coverage over MostPopular, while exposure remains concentrated on popular items. This comparison alone does not isolate the effect of graph propagation; MF and layer-depth comparisons are still needed.

The run completed 200 epochs, with epoch 192 selected by validation Recall@20. These are single-seed validation results, not final test scores or evidence of convergence. See [result files and provenance](results/README.md).

The result JSON files were recovered from ARC output shared in the chat. The complete learning curve, ARC plots, item-level tables, and full environment export have not yet been transferred. [The metrics template](results/metrics-template.csv) remains header-only. Published scores are not presented as team measurements.

## References

- He et al. [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126), 2020.
- Wu et al. [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153), 2019.
- Gasteiger et al. [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/abs/1810.05997), 2019.
- Chen et al. [How Graph Convolutions Amplify Popularity Bias for Recommendation?](https://arxiv.org/abs/2305.14886), 2023 preprint.

External code must retain its original attribution and license. No license for redistribution of this team's original work has been selected yet.
