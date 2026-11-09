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

The remaining seed-2020 standard layer-depth runs completed on ARC using `scripts/train_layers_arc.sh`. Their verified artifacts are indexed in [the results guide](../results/README.md#completed-layer-depth-comparison).

| Layers | Slurm job | ARC result directory under `results/gowalla/` |
| --- | --- | --- |
| 1 | 327365 | `lightgcn-k1-seed2020-job327365/` |
| 2 | 327366 | `lightgcn-k2-seed2020-job327366/` |
| 4 | 327367 | `lightgcn-k4-seed2020-job327367/` |

Each run uses the same fitting split, popularity groups, seed, embedding size, optimizer settings, 500-epoch maximum, and patience 20. The job evaluates its best checkpoint automatically after training. Their completed configurations, histories, metrics, and exposure counts have been collected and added to the comparison table. The completed K=3 continuation provides the current three-layer result, with its documented legacy random-state limitation.

Save each run's configuration, history, selected-checkpoint metrics, and item exposure table under a unique directory in `results/gowalla/`. Include variant settings and seed in the run name. Use [the metrics template](../results/metrics-template.csv) for the combined comparison.

## First APPNP-style run

The two-epoch GPU verification (ARC job 327901) passed. Its saved settings,
source hashes, data hashes, checkpoint selection and all 597,160 exposure
slots were checked. The full run was submitted on November 6, 2026 as job
328675 and uses the run directory
`lightgcn-appnp-beta0.5-k3-seed2020-job328675`.
The full run completed with early stopping at epoch 211, selecting epoch 191.
Recall@20 was 0.18427, NDCG@20 0.09694, tail share 14.35%, coverage 49.13%,
exposure Gini 0.89639, and tail Recall@20 0.05501. Source and split hashes,
checkpoint selection, stopping history, and exposure accounting were verified.

The [comparison CSV](../results/gowalla/appnp-beta05-comparison.csv) includes
standard K=3, this variant, and standard K=1. Against the continued K=3 run,
beta 0.5 improved tail recall and spread exposure across more items, with a
4.25% relative loss in overall recall. Standard K=1 performed better on all
six reported metrics in these seed-2020 runs. These are preliminary validation
observations. The K=3 reference has the documented legacy sampling restart,
and a fresh K=3 control and repeated seeds remain planned.

The initial weighting experiment uses three layers, beta 0.5, training seed
2020, a maximum of 500 epochs, and early-stopping patience 20. Its weights
over layers 0 through 3 are [0.5, 0.25, 0.125, 0.125]. Standard normalization,
BPR loss, initialization, optimization, and ranking rules follow the pinned
upstream implementation.

`scripts/weighted_lightgcn.py` subclasses the pinned model and overrides
only layer aggregation. Both training and evaluation reconstruct this variant
from `appnp_beta` in checkpoint metadata. The run records layer weights, the
active sampler backend, and hashes of its training, evaluation, and model
scripts. Older checkpoints without `appnp_beta` use standard LightGCN.

`scripts/setup_appnp_arc.sh` prepares the environment in
`/mnt/beegfs/$USER/lightgcn-appnp-20261105`.
`scripts/train_appnp_arc.sh` uses that environment and exports small result
files to the corresponding project under the user's home directory.
Checkpoints stay on BeeGFS.

On ARC, submit the two-epoch check before the full run:

```bash
cd "$HOME"
sbatch --export=ALL,MAX_EPOCHS=2 --output="$HOME/appnp-check-%j.log" \
  "$HOME/lightgcn-appnp-20261105/scripts/train_appnp_arc.sh"
# After the checkpoint evaluation passes:
sbatch --output="$HOME/appnp-full-%j.log" \
  "$HOME/lightgcn-appnp-20261105/scripts/train_appnp_arc.sh"
```

The initial K=3 comparison used the team's continued run, whose legacy
sampling restart at epoch 200 remains a limitation. The fresh standard K=3
control below now serves as the primary APPNP reference. Additional training
seeds remain planned.

## Fresh control and beta 0.2

On November 7, 2026, two seed-2020 jobs were submitted with a 500-epoch
maximum and patience 20. Both start from fresh embeddings on the same ARC
environment used for beta 0.5.

| Configuration | ARC job | Result directory |
| --- | --- | --- |
| Standard K=3, uniform weights | 329225 | `lightgcn-k3-fresh-seed2020-job329225` |
| K=3, APPNP beta 0.2 | 329226 | `lightgcn-appnp-beta0.2-k3-seed2020-job329226` |

The beta 0.2 weights over depths 0–3 are [0.2, 0.16, 0.128, 0.512].
The standard control uses the original upstream LightGCN class.
Beta 0.2 completed at epoch 289 and selected epoch 269. Verified validation
results are Recall@20 0.18985, NDCG@20 0.09883, tail share 9.49%, coverage
31.86%, exposure Gini 0.94171 and tail Recall@20 0.04294.
The fresh standard K=3 control completed at epoch 286 and selected epoch
266. Verified results are Recall@20 0.19399, NDCG@20 0.10147, tail share
11.23%, coverage 37.98%, exposure Gini 0.92714 and tail Recall@20 0.04989.
The [updated comparison](../results/gowalla/appnp-weighting-comparison.csv)
uses this fresh control alongside both completed APPNP variants and the
team's K=1 result. Beta 0.2 performed worse than the fresh control on all six
metrics in these seed-2020 runs. Beta 0.5 improved tail recall and exposure
with lower overall accuracy. Additional seeds remain planned. Each job automatically evaluates its best checkpoint
and exports the small result files to the project in the ARC home directory.
The launcher accepts `APPNP_BETA=uniform` for the fresh standard control.

## Beta 0.1

ARC job 329524 was submitted on November 7, 2026 with beta 0.1, K=3,
seed 2020, a 500-epoch maximum and patience 20. Its weights over depths
0–3 are [0.1, 0.09, 0.081, 0.729]. The result directory is
`lightgcn-appnp-beta0.1-k3-seed2020-job329524`.
The run stopped after epoch 379 and selected epoch 359. Results downloaded
and verified on November 9, 2026: Recall@20 0.18194, NDCG@20 0.09450,
tail share 7.75%, coverage 25.04%, exposure Gini 0.95564 and tail Recall@20
0.03473. Configuration and code hashes match the shared protocol; checkpoint
selection was checked against the full history, and exposure metrics were
recomputed from recommendation counts. Tail recall is the evaluator's measured
output. Beta 0.1 performed worse than the fresh standard K=3 control on all six
metrics in this seed-2020 comparison.

All three planned APPNP weights (beta 0.1, 0.2, 0.5) now have completed
seed-2020 validation results in the updated comparison CSV. Beta 0.5 provides
a tradeoff between overall accuracy and tail relevance/exposure, while beta
0.1 and 0.2 do not improve the fresh K=3 control on any reported metric.
These are preliminary, single-seed findings. Independent normalization
experiments and additional training seeds remain planned; the authors' test
set has not been used.

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

## Normalization experiment: first setting

The first normalization ablation fixes receiver exponent a=0.5 and changes
neighbor exponent b from the standard 0.5 to 0.75. It uses the upstream
LightGCN model with uniform weights [0.25, 0.25, 0.25, 0.25], three layers,
seed 2020, maximum 500 epochs, and validation Recall@20 patience 20. APPNP
weighting is not combined with this change. The fresh K=3 run (job 329225)
is the standard-normalization reference.

For an edge from column neighbor j to row receiver i, the propagation weight
is 1/(degree(i)^a * degree(j)^b), in both user-to-item and item-to-user
directions. Degrees are from fitting interactions only. With a != b the
propagation matrix is asymmetric: the reverse message generally has a
different weight. Isolated nodes send and receive no messages; their ego
embeddings still contribute to the uniform layer average. Larger b also
changes propagation magnitude, so any outcome cannot be attributed solely
to suppressing popularity. No additional rescaling is applied.

`normalized_graph.py` extends the upstream dataset loader and changes only
the graph operator. Default (a,b)=(0.5,0.5) continues to use the original
loader. Nonstandard caches include exact exponent values and a schema
version, inside the fitting/validation-hash directory; cache installation
is atomic. Training checkpoints record the exponents and script hashes.
Evaluation reconstructs the saved graph normalization, and resume rejects
changed normalization settings. Old checkpoints default to (0.5,0.5).

Local numerical checks passed for equivalence to standard normalization,
receiver/neighbor orientation, zero-degree nodes, cache separation/reload,
identical initialized embeddings, three-layer propagation, BPR loss and
regularization, finite gradients, and checkpoint reload. Run them with:

```bash
python tests/check_normalization.py LightGCN-PyTorch/code
```

ARC verification job 332024 completed on November 9, 2026, for two
epochs. Its saved checkpoint passed all six metric evaluations. Configuration,
source/split hashes, checkpoint selection and derived exposure metrics were
independently checked from the exported artifacts. These two-epoch results
verify the pipeline and are excluded from model-performance comparisons. It uses the isolated workspace `lightgcn-normalization-20261109`
and the existing portable Python environment from the APPNP runs. The
launcher verifies the fixed split and upstream commit and automatically
evaluates the selected checkpoint. Small artifacts are exported to the
same project directory in ARC home; checkpoints remain on BeeGFS.

```bash
# Two-epoch verification
sbatch --job-name=norm-check \
  --export=ALL,MAX_EPOCHS=2,NORM_A=0.5,NORM_B=0.75,TRAINING_SEED=2020 \
  --output="$HOME/norm-check-%j.log" \
  "$HOME/lightgcn-normalization-20261109/scripts/train_normalization_arc.sh"
# Full run after verification succeeds
sbatch --job-name=norm-b075 \
  --export=ALL,MAX_EPOCHS=500,NORM_A=0.5,NORM_B=0.75,TRAINING_SEED=2020 \
  --output="$HOME/norm-b075-%j.log" \
  "$HOME/lightgcn-normalization-20261109/scripts/train_normalization_arc.sh"
```

Full job 332026 completed successfully, stopping at epoch 307 and selecting
epoch 287 by validation Recall@20. Verified metrics: Recall@20 0.18212,
NDCG@20 0.09564, tail share 10.72%, coverage 34.78%, exposure Gini 0.93562
and tail Recall@20 0.04205. The [comparison table](../results/gowalla/normalization-comparison.csv)
shows that this setting underperformed the fresh standard K=3 control on
all six metrics in this one-seed experiment. Configuration/source/split
hashes and the full stopping/selection history were verified. Reloaded
checkpoint accuracy agrees with history; tail share, coverage and Gini were
independently recomputed from item counts. Tail recall is the evaluator
measurement. These are preliminary validation results, not a conclusion
about all normalization choices.
Its output directory is `lightgcn-norm-a0.5-b0.75-k3-seed2020-job332026`
and ARC home log is `norm-b075-332026.log`. The b=1.0 setting is submitted
as described below; additional seeds remain planned. The authors' test set remains untouched.


## Normalization experiment: b=1.0

ARC job 332568 was submitted on November 9, 2026, using a=0.5, b=1.0,
three layers, uniform layer weights, seed 2020, maximum 500 epochs and
patience 20. This uses the same verified implementation, fitting/validation
split and evaluation protocol as b=0.75. Numerical tests already cover
b=1.0 and are rerun by the launcher before training. No APPNP weighting
is applied. Results are pending and will remain preliminary until repeated
across training seeds.

```bash
sbatch --job-name=norm-b10 \
  --export=ALL,MAX_EPOCHS=500,NORM_A=0.5,NORM_B=1.0,TRAINING_SEED=2020 \
  --output="$HOME/norm-b10-%j.log" \
  "$HOME/lightgcn-normalization-20261109/scripts/train_normalization_arc.sh"
```

ARC home log: `norm-b10-332568.log`. The result directory is
`lightgcn-norm-a0.5-b1.0-k3-seed2020-job332568`. The launcher evaluates the
best validation checkpoint and exports the small result files to
`$HOME/lightgcn-normalization-20261109/results/gowalla/` after completion.
Compare it with the fresh standard K=3 control and completed b=0.75 run.
The authors' test set remains untouched.
