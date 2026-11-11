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
and ARC home log is `norm-b075-332026.log`. The b=1.0 setting is complete
as described below; additional seeds remain planned. The authors' test set remains untouched.


## Normalization experiment: b=1.0

ARC job 332568 was submitted on November 9, 2026, using a=0.5, b=1.0,
three layers, uniform layer weights, seed 2020, maximum 500 epochs and
patience 20. This uses the same verified implementation, fitting/validation
split and evaluation protocol as b=0.75. Numerical tests already cover
b=1.0 and are rerun by the launcher before training. No APPNP weighting
is applied. The run completed at epoch 232 and selected epoch 212. Results
remain preliminary until repeated across training seeds.

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


Verified b=1.0 validation results: Recall@20 0.17323, NDCG@20 0.09094,
tail share 10.80%, catalog coverage 35.33%, exposure Gini 0.93605 and
tail Recall@20 0.04214. Configuration, source and split hashes match the
protocol. The complete history reproduces strict-best checkpoint selection
and patience-20 stopping. Reloaded checkpoint accuracy agrees with history;
item counts reproduce tail share, coverage and Gini. Tail recall is the
evaluator measurement.

Both b=0.75 and b=1.0 underperform the standard (a,b)=(0.5,0.5) control
on all six reported metrics in these seed-2020 runs. Moving from b=0.75
to b=1.0 further lowers overall accuracy, with small increases in tail
share, coverage and tail recall, and slightly higher Gini. This supports
no benefit for the tested normalization changes under the fixed training
protocol; it does not rule out other normalization or optimization choices.
Changing exponents also changes propagation magnitude, as noted above.

The planned seed-2020 APPNP and normalization grids are now complete.
Next priority is repeating the key comparisons with seeds 2021 and 2022,
keeping the fitting/validation split fixed, before drawing robust conclusions.
No final test results have been measured.


## Seed-2021 replication batch

On November 9, 2026, ARC array 333336 was submitted with at most two jobs
running simultaneously. This batch repeats MF, every standard LightGCN
depth from 1 to 4, and APPNP beta 0.5. The full depth series checks the main
propagation finding; beta 0.5 checks the observed accuracy/tail-relevance
tradeoff. Selection of this variant was based on seed-2020 validation, so
it is not an unbiased comparison of every APPNP/normalization variant over
multiple seeds. Other variants remain single-seed findings until repeated.

Only the training seed changes to 2021. The fitting/validation split remains
fixed (split seed 2020); popularity labels, catalog, exclusion rules,
initialization distribution, embedding size, loss, sampler, optimizer,
regularization and validation checkpoint selection remain unchanged.
Every model starts fresh, with maximum 500 epochs and patience 20. The K=3
reference for seed 2020 is the fresh run, not the continued legacy run.
MostPopular is deterministic on this fixed split and requires no new run.

| Array task | Model | Training seed | Final metrics |
| --- | --- | --- | --- |
| 333336_0 | MF, zero layers | 2021 | verified; selected epoch 262 |
| 333336_1 | LightGCN, one layer | 2021 | complete; selected epoch 138 |
| 333336_2 | LightGCN, two layers | 2021 | verified; selected epoch 327 |
| 333336_3 | LightGCN, three layers | 2021 | verified; selected epoch 374 |
| 333336_4 | LightGCN, four layers | 2021 | verified; selected epoch 317 |
| 333336_5 | APPNP beta 0.5, three layers | 2021 | verified; selected epoch 305 |

Submission command:

```bash
sbatch --export=ALL,TRAINING_SEED=2021 --output="$HOME/gowalla-seed2021-%A_%a.log" "$HOME/lightgcn-normalization-20261109/scripts/train_seed_comparison_arc.sh"
```

The [submission manifest](../provenance/seed2021-submission.json) records
exact run names. Logs are in ARC home as `gowalla-seed2021-333336_<task>.log`.
The launcher evaluates each selected checkpoint and exports small artifacts
to `$HOME/lightgcn-normalization-20261109/results/gowalla/`. Checkpoints stay
in the corresponding BeeGFS project. Seed 2022 remains unsubmitted by us;
coordinate with teammates before launching it. The authors' test set remains
reserved for final evaluation.


Seed-2021 K=1 completed at epoch 158 and selected epoch 138. Verified
validation metrics: Recall@20 0.17911, NDCG@20 0.09441, tail share 14.56%,
coverage 51.08%, exposure Gini 0.89477 and tail Recall@20 0.05268.
Configuration/source/split checks passed. The full history reproduces strict
checkpoint selection and patience-20 stopping; reloaded accuracy agrees
with history, and item counts reproduce exposure metrics. Tail recall is
the evaluator measurement. Compared with seed 2020, this run stopped earlier
and has lower recall and tail recall. Do not change stopping rules after
seeing the result; complete the remaining seeds before interpreting model
rankings. Other seed-2021 models remain pending at this checkpoint.


Seed-2021 progress checked November 10, 2026: MF completed at epoch 282
(best 262), and K=1 completed at epoch 158 (best 138). Their configurations,
source/split hashes, strict-best checkpoint selection, stopping history and
accuracy agreement were verified. Tail share, coverage and Gini were
recomputed from per-item exposure counts; tail recall is the evaluator's
measurement. See [the partial six-metric table](../results/gowalla/seed2021-comparison.csv).
Other rows remain pending until completed checkpoint evaluation is verified.
At this check K=2 and K=3 were training, with K=4 and APPNP beta 0.5 queued.

MF Recall@20 is 0.15532 (seed 2020: 0.15508); K=1 Recall@20 is 0.17911
(seed 2020: 0.18513). No training settings or stopping rules were changed
in response to these results. The earlier single-seed model ordering should
not be treated as established until the repeated-seed comparison is complete.


### Completed seed-2021 batch

All six jobs completed and were verified on November 10, 2026. K=2 stopped
at epoch 347 (best 327), K=3 at 394 (best 374), K=4 at 337 (best 317), and
APPNP beta 0.5 at 325 (best 305). All use the fixed split, full catalog and
original stopping/selection protocol. Configurations and source hashes match
the planned runs; full histories reproduce checkpoint selection and early
stopping. Reloaded accuracy agrees with history. Exposure counts reproduce
tail share, coverage and Gini. Tail recall is the evaluator measurement.

The [seed-2021 table](../results/gowalla/seed2021-comparison.csv) now contains
all six metrics. The [two-seed summary](../results/gowalla/two-seed-comparison.csv)
reports each seed's value, arithmetic mean and sample standard deviation
(denominator n-1) for seeds 2020 and 2021, with explicit source run IDs.
K=3 uses the fresh seed-2020 control. Two-seed summaries are preliminary;
seed 2022 has not been run by us and statistical significance is not claimed.

Across both seeds, tail share and catalog coverage decrease and exposure
Gini increases from one to four standard LightGCN layers. Three layers has
the highest Recall@20 in each seed, with two layers close in seed 2021.
Tail recall does not decline monotonically: K=2 exceeds K=1 in seed 2021.
APPNP beta 0.5 improves tail recall/exposure and coverage, and reduces Gini,
relative to K=3 in both seeds, at lower overall Recall/NDCG. The seed-2020
observation that K=1 outperforms APPNP on every metric does not repeat:
APPNP has higher recall, NDCG and tail recall and lower Gini than K=1 in
seed 2021, while K=1 has slightly higher coverage. APPNP's tail share is also
higher in seed 2021. Avoid claiming a robust winner from one run.

No jobs remain active in this batch. Coordinate seed 2022 with teammates.
Other APPNP weights and normalization variants still have only seed-2020
results. The authors' test set remains untouched.

## Seed-2022 replication batch

On November 10, 2026, ARC array 334447 was submitted after checking the
latest team main branch and our ARC queue/artifacts for existing seed-2022
runs. This repeats exactly the six-model seed-2021 batch: MF, standard
LightGCN K=1–4, and APPNP beta 0.5 with three layers. The same verified
launcher runs at most two tasks simultaneously. Only the training seed
changes; the split remains fixed at split seed 2020. Every model starts
fresh with maximum 500 epochs and validation Recall@20 patience 20.

| Array task | Model | Training seed | Final metrics |
| --- | --- | --- | --- |
| 334447_0 | MF, zero layers | 2022 | verified; selected epoch 352 |
| 334447_1 | LightGCN, one layer | 2022 | verified; selected epoch 164 |
| 334447_2 | LightGCN, two layers | 2022 | verified; selected epoch 283 |
| 334447_3 | LightGCN, three layers | 2022 | verified; selected epoch 288 |
| 334447_4 | LightGCN, four layers | 2022 | verified; selected epoch 477 |
| 334447_5 | APPNP beta 0.5, three layers | 2022 | verified; selected epoch 194 |

```bash
sbatch --job-name=gowalla-s2022 \
  --export=ALL,TRAINING_SEED=2022 \
  --output="$HOME/gowalla-seed2022-%A_%a.log" \
  "$HOME/lightgcn-normalization-20261109/scripts/train_seed_comparison_arc.sh"
```

The [submission manifest](../provenance/seed2022-submission.json) records
all run names. Logs are in ARC home as `gowalla-seed2022-334447_<task>.log`.
Each task evaluates its selected checkpoint and exports configuration,
history, stopping summary, six metrics and item exposure counts to
`$HOME/lightgcn-normalization-20261109/results/gowalla/`. Checkpoints remain
on BeeGFS. Results are pending, and no three-seed summary is claimed yet.
After verification, report the mean and sample standard deviation across
2020, 2021 and 2022; other weighting/normalization variants remain single-seed
results. The authors' test set is still reserved for final evaluation.


Seed-2022 progress checked November 10, 2026: K=1 completed at epoch 184
and selected epoch 164. Verified validation metrics: Recall@20 0.18018,
NDCG@20 0.09516, tail share 14.77%, coverage 52.07%, exposure Gini 0.89054
and tail Recall@20 0.05462. Configuration/source/split checks passed;
full history reproduces strict-best selection and patience-20 stopping.
Reloaded accuracy matches history; item exposure counts reproduce tail share,
coverage and Gini. Tail recall is the evaluator measurement. At this check,
MF and K=2 were training; K=3, K=4 and APPNP beta 0.5 were queued. Final
metrics for those models remain pending. The full three-seed summary awaits
completion and verification of this batch.


Seed-2022 progress on November 10, 2026: K=1 completed at epoch 184 and
selected epoch 164. Verified Recall@20 0.18018, NDCG@20 0.09516, tail share
14.77%, coverage 52.07%, exposure Gini 0.89054 and tail Recall@20 0.05462.
Configuration/source/split hashes, strict checkpoint selection and patience-20
stopping were checked. Reloaded accuracy agrees with history; exposure
counts reproduce tail share, coverage and Gini. Tail recall is the evaluator
measurement. At this checkpoint MF and K=2 were still training; K=3, K=4
and APPNP were queued. Other final metrics remain pending.


### Completed seed-2022 batch and three-seed validation summary

All six seed-2022 jobs completed and were verified on November 11, 2026.
Selected/stopped epochs were MF 352/372, K=1 164/184, K=2 283/303,
K=3 288/308, K=4 477/497, and APPNP beta 0.5 194/214. All runs stopped
under the unchanged patience-20 rule; no settings were adjusted after
observing outcomes. K=4 finished close to the 500-epoch cap; satisfying
early stopping does not establish mathematical convergence.

The [seed-2022 table](../results/gowalla/seed2022-comparison.csv) contains
all six metrics. The [three-seed table](../results/gowalla/three-seed-comparison.md)
reports means and sample standard deviations (n-1 denominator) across
2020, 2021 and 2022; the [CSV](../results/gowalla/three-seed-comparison.csv)
also contains each seed's value and source run IDs. K=3 uses the fresh
seed-2020 reference. Historical two-seed files remain preserved.

Verification covered configuration, script and split hashes; complete epoch
histories; first-strict-maximum Recall@20 checkpoint selection; and the first
patience-20 stopping point. Reloaded Recall/NDCG agree with history. Item
counts reproduce tail share, coverage and Gini over the complete catalog.
Tail recall is the evaluator's measured output, not reconstructed from
aggregate exposure counts.

Across all three seeds, tail share and coverage decrease and exposure Gini
increases from MF through standard LightGCN depths 1–4. Mean tail share falls
from 19.36% for MF to 14.96%, 12.81%, 11.18% and 9.59% for K=1–4. These
measurements support propagation-related concentration under this protocol.
They do not imply that all tail exposure is useful or establish fairness.
Tail recall is not monotonic with every depth increment.

K=3 has the highest mean Recall@20 (0.1957), closely followed by K=2
(0.1950). K=2 has higher mean NDCG (0.1027 vs 0.1025), tail recall
(0.0553 vs 0.0507), tail share and coverage, with lower Gini. K=4 has the
highest recall in seed 2022, so K=3 is not the best for every seed. The small
K=2/K=3 mean recall difference should not be presented as statistically
established superiority.

APPNP beta 0.5 improves tail share, coverage, Gini and tail recall over K=3
in every seed, with lower Recall/NDCG in every seed. The mean tail recall is
0.0566 vs 0.0507, at mean Recall 0.1864 vs 0.1957. K=1 does not consistently
outperform APPNP; the earlier seed-2020 dominance statement was specific
to that run. Beta 0.5 was chosen for replication using seed-2020 validation,
so the replicated set is not an unbiased full-grid comparison.

No jobs remain active in this batch. Other APPNP weights and normalization
variants remain single-seed results. Three training seeds measure variation
on one fixed split, not robustness across splits or statistical significance.
Next: freeze the final comparison/evaluation protocol with the team before
using the reserved authors' test set. No final test measurements exist yet.
