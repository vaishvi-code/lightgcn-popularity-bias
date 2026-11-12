# Results

**Current status:** the [frozen final-test comparison](gowalla/final-test-job337725/README.md)
is complete for all 23 entries. It includes three-seed main results and
separate single-seed exploratory results. Earlier sections below preserve
the chronological validation milestones; their pending/one-seed statements
describe those earlier stages.

The `gowalla/` directory contains completed data-preparation, MostPopular, original and continued LightGCN, completed 1-, 2-, and 4-layer runs, and BPR-MF validation measurements, plus historical training snapshots. The original ARC artifact export has been transferred and checked against the earlier terminal-output snapshots. `metrics-template.csv` remains a header-only format template.

| File | Contents |
| --- | --- |
| `gowalla/split-metadata.json` | Fixed fitting/validation split counts and original-training hash |
| `gowalla/popularity-summary.json` | Popular/tail catalog and interaction shares |
| `gowalla/mostpopular-validation.json` | Completed MostPopular validation evaluation |
| `gowalla/lightgcn-k3-best-validation.json` | Accuracy and exposure of epoch 192, selected after the 200-epoch run; one seed |
| `gowalla/comparison.csv` | MostPopular, MF, and all completed LightGCN depths, including the original historical run, with run IDs, stopping settings, and selected epochs |
| `gowalla/archive/lightgcn-k3-training-snapshot.json` | Historical epoch-16 snapshot, superseded by the completed evaluation |
| `gowalla/archive/lightgcn-k3-history-excerpt.csv` | Historical excerpts from training; use the run directory for the complete history |

The following original ARC artifacts are now included:

- `gowalla/popularity-distribution.png`: fitting-data popularity figure.
- `gowalla/item-popularity.csv`: fixed item degree and popular/tail labels.
- `gowalla/mostpopular-exposure.csv`: recommendation counts for every item.
- `gowalla/archive/lightgcn-k3-seed2020-20261102-164221/`: two-epoch check configuration and history.
- `gowalla/lightgcn-k3-seed2020-20261102-164455/`: full 200-epoch history, configuration, best-checkpoint metrics, and item-level exposure.
- `provenance/environment-freeze.txt` and `provenance/upstream-commit.txt`: actual ARC provenance exports.

The exported history confirms epoch 192 is the selected checkpoint. Item-level tables reproduce the reported coverage, tail share, and exposure Gini. Checkpoints and datasets remain excluded from Git.

Null exposure values in the historical epoch-16 snapshot mean they had not been measured at that point. Use `gowalla/lightgcn-k3-best-validation.json` for the completed evaluation. Finishing 200 epochs does not establish convergence; all reported LightGCN findings use one seed and validation data. The final test set remains untouched.

Scripts recovered from the shared terminal logs are under `scripts/`. Their base path was adjusted for the repository's `scripts/` directory. The batch script is adapted to accept `PROJECT_ROOT`; the ARC training and evaluation artifacts have not been modified.

Store small, reviewable result tables and final plots here. Raw outputs under `raw/` are excluded from Git, as are model checkpoints and training logs.

## Matrix factorization verification

The [two-epoch zero-layer check](gowalla/archive/mf-k0-seed2020-job327066/) completed successfully on ARC as job 327066. It trained with the baseline's initialization and BPR settings, saved checkpoints, and evaluated relevance and exposure over all 29,858 validation users. Its best-checkpoint recall matches the training history. The archived configuration, history, completion summary, metrics, and item exposure table document the check. These two-epoch measurements verify the pipeline and are excluded from the main model comparison table.

The [full BPR-MF comparison](gowalla/mf-k0-seed2020-job327067/) completed as job 327067. It stopped at epoch 233 and selected epoch 213. The [continued LightGCN run](gowalla/lightgcn-k3-seed2020-20261105-001600-153197/) completed as job 327058, stopped at epoch 266, and selected epoch 246. Its selected checkpoint was evaluated by job 327360. Both used seed 2020, a 500-epoch maximum, and patience 20.

Each completed directory includes `config.json`, `history.csv`, `run-summary.json`, `best-validation-metrics.json`, and `best-validation-exposure.csv`. The LightGCN history includes the original 200 epochs and 66 continued epochs. Its configuration records `resume_rng=reseeded_legacy_checkpoint` because the original checkpoint did not save random-generator states. Checkpoint selection, history lengths, item exposure totals, coverage counts, Gini, and tail share were checked against the exported files.

MF has higher tail exposure and tail recall while LightGCN has higher overall recall in this one-seed comparison. Additional seeds and model settings are still required.

For each run, record the model/configuration, dataset, evaluation split, split seed, training seed, selected epoch, metric values, runtime, and code commit. Leave inapplicable fields such as beta blank for models that do not use them.

Report the actual seed count and mean/standard deviation for repeated runs. Label preliminary or unconverged measurements and keep validation and test results distinct. Published reference scores must be separately attributed.

## Completed layer-depth comparison

All three additional depth jobs completed successfully, including best-checkpoint accuracy and exposure evaluation. They used the same fitting and validation hashes, seed 2020, embedding size, optimizer, batch size, 500-epoch maximum, and patience 20 as the current three-layer reference.

| Layers | Job | Best epoch | Stopped epoch | Artifacts |
| --- | --- | ---: | ---: | --- |
| 1 | 327365 | 265 | 285 | [Run files](gowalla/lightgcn-k1-seed2020-job327365/) |
| 2 | 327366 | 194 | 214 | [Run files](gowalla/lightgcn-k2-seed2020-job327366/) |
| 4 | 327367 | 300 | 320 | [Run files](gowalla/lightgcn-k4-seed2020-job327367/) |

Each folder contains its configuration, full history, stopping summary, selected-checkpoint metrics, and recommendation counts for all 40,981 items. The exported histories reproduce checkpoint selection and the stopping epoch. Item counts reproduce coverage, tail share, and Gini.

Tail share and coverage decrease from one to four layers, while Gini increases. Three layers has the highest overall recall, with two layers close behind. One layer has the highest tail recall. The comparison uses one seed and validation data. The three-layer reference was resumed from a legacy checkpoint and has the random-state limitation recorded above.

## Navigation

- [Supplemental embedding diagnostics for Q4](gowalla/embedding-diagnostics-job340254/README.md)
- [Supplemental LightGCN-single protocol and job status](../docs/supplemental-analysis.md)
- [Current comparison table](gowalla/comparison.csv)
- [Current selected LightGCN metrics](gowalla/lightgcn-k3-seed2020-20261105-001600-153197/best-validation-metrics.json)
- [Selected MF metrics](gowalla/mf-k0-seed2020-job327067/best-validation-metrics.json)
- [Full training run](gowalla/lightgcn-k3-seed2020-20261102-164455/)
- [Historical artifacts](gowalla/archive/)
- [Experiment protocol](../docs/experiments.md)

## Completed APPNP weighting comparison (preliminary)

[Comparison CSV](gowalla/appnp-weighting-comparison.csv) contains the fresh
standard K=3 control, APPNP beta 0.1, 0.2, 0.5, and the team's K=1 result.
All are validation results for training seed 2020, maximum 500 epochs and
patience 20, with the same fitting/validation split and metric definitions.

| Configuration | ARC job | Selected epoch | Stopped epoch |
| --- | --- | ---: | ---: |
| Fresh standard K=3 | 329225 | 266 | 286 |
| APPNP beta 0.1 | 329524 | 359 | 379 |
| APPNP beta 0.2 | 329226 | 269 | 289 |
| APPNP beta 0.5 | 328675 | 191 | 211 |

Each run directory includes configuration with source hashes, full history,
summary, selected-checkpoint metrics and item exposure counts. The environment
is recorded in `../provenance/appnp-environment-freeze.txt`. The two-epoch
beta-0.5 run (job 327901) is a pipeline check, excluded from the comparison.
`appnp-beta05-comparison.csv` preserves the earlier comparison with the
continued K=3 run; use the fresh-control table above for current conclusions.

Beta 0.1 and 0.2 underperform the fresh K=3 control on all six reported
metrics. Beta 0.5 improves tail exposure, coverage, Gini and tail recall but
reduces overall accuracy. The team's K=1 model outperforms beta 0.5 on all six
metrics in these single-seed results. Greater tail exposure alone does not
establish better relevance or fairness. Additional seeds are required to
assess consistency. The authors' test set remains untouched.

Configuration/split hashes, strict-best checkpoint selection and stopping
history were checked; tail share, coverage and Gini were recomputed from item
counts. Recall and NDCG from reloaded checkpoints agree with training history.
Checkpoints remain on ARC and are excluded from Git.

## Completed normalization comparison (preliminary)

The two-epoch normalization check (a=0.5, b=0.75), job 332024, completed
successfully. Its artifacts are in
`gowalla/lightgcn-norm-a0.5-b0.75-k3-seed2020-job332024/`. It verifies the
pipeline and is excluded from final comparison tables. Full seed-2020 job
332026 uses three layers, uniform layer averaging, maximum 500 epochs and
patience 20. It stopped at epoch 307 and selected epoch 287. The
[normalization comparison](gowalla/normalization-comparison.csv) reports all
six validation metrics alongside the fresh K=3 reference. This setting
underperformed the reference on all six metrics: lower accuracy and tail
recall, lower tail share and coverage, and higher exposure Gini. These are
preliminary single-seed results. Full artifacts are in
`gowalla/lightgcn-norm-a0.5-b0.75-k3-seed2020-job332026/`. The b=1.0 setting also completed as ARC job 332568, stopping at epoch 232
and selecting epoch 212. Its artifacts are in
`gowalla/lightgcn-norm-a0.5-b1.0-k3-seed2020-job332568/` and the comparison
CSV includes all six metrics. Both variants underperform the fresh K=3
control on all six metrics in these seed-2020 results. Additional seeds
remain planned; the authors' test set remains untouched.
See the [normalization protocol](../docs/experiments.md#normalization-experiment-first-setting).


## Completed seed-2021 replication

[Seed-2021 results](gowalla/seed2021-comparison.csv) cover MF, LightGCN
K=1–4, and APPNP beta 0.5. All six selected-checkpoint evaluations are
complete and verified. Their raw configurations, histories, stopping
summaries and exposure counts are included in the corresponding
`seed2021-job333336_*` folders.

[Two-seed summary](gowalla/two-seed-comparison.csv) includes values for
2020/2021, their means and sample standard deviations, with source run IDs.
It uses the fresh K=3 seed-2020 control and does not claim significance.
The decrease in tail exposure with depth and APPNP beta 0.5's accuracy/tail
relevance tradeoff against K=3 repeat across both seeds. K=1's earlier
advantage over APPNP on all metrics does not repeat. Seed 2022 is pending;
other weighting/normalization variants remain single-seed results.


## Completed three-seed validation comparison

The [three-seed table](gowalla/three-seed-comparison.md) reports all six
metrics as mean and sample standard deviation for MF, LightGCN K=1–4,
and APPNP beta 0.5. The [CSV](gowalla/three-seed-comparison.csv) includes
individual values and source run IDs. Seed-2022 measurements are also in
[their own table](gowalla/seed2022-comparison.csv). The historical two-seed
summary remains available.

All six seed-2022 runs completed and passed artifact verification. Tail
exposure and coverage decline with standard propagation depth in all three
seeds. K=3 has the highest mean recall, while K=2 has similar recall and
better mean tail recall, coverage and Gini. APPNP beta 0.5 consistently
trades overall accuracy for improved tail relevance/exposure against K=3.
No statistical significance is claimed from these summaries. Other variants
remain single-seed experiments; final test evaluation is pending.


## Final-test status

**Complete:** ARC job 337725 reproduced all six validation metrics and exact
item exposures for all 23 frozen entries, then evaluated the original test
set. The four previously inaccessible checkpoints were shared through the
[seed-2020 release](https://github.com/vaishvi-code/lightgcn-popularity-bias/releases/tag/seed2020-checkpoints)
and verified against its hashes and the frozen validation references.

The [test report](gowalla/final-test-job337725/README.md) includes the full
main and exploratory tables and interpretation. [Individual results](gowalla/final-test-job337725/test-per-run.csv),
[grouped statistics](gowalla/final-test-job337725/test-summary.csv), per-item
test exposure tables and verification audits are preserved. The initial
incomplete [job-336563 audit](../provenance/final-validation-verification-job336563.json)
remains as provenance. No plan, checkpoint or hyperparameter was changed
in response to test results. The next step is final reporting, not tuning
on the opened test set.
