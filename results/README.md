# Results

The `gowalla/` directory contains completed data-preparation, MostPopular, original and continued LightGCN, and BPR-MF validation measurements, plus historical training snapshots. The original ARC artifact export has been transferred and checked against the earlier terminal-output snapshots. `metrics-template.csv` remains a header-only format template.

| File | Contents |
| --- | --- |
| `gowalla/split-metadata.json` | Fixed fitting/validation split counts and original-training hash |
| `gowalla/popularity-summary.json` | Popular/tail catalog and interaction shares |
| `gowalla/mostpopular-validation.json` | Completed MostPopular validation evaluation |
| `gowalla/lightgcn-k3-best-validation.json` | Accuracy and exposure of epoch 192, selected after the 200-epoch run; one seed |
| `gowalla/comparison.csv` | MostPopular, original and continued LightGCN, and MF, with run IDs, stopping settings, and selected epochs |
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

## Navigation

- [Current comparison table](gowalla/comparison.csv)
- [Current selected LightGCN metrics](gowalla/lightgcn-k3-seed2020-20261105-001600-153197/best-validation-metrics.json)
- [Selected MF metrics](gowalla/mf-k0-seed2020-job327067/best-validation-metrics.json)
- [Full training run](gowalla/lightgcn-k3-seed2020-20261102-164455/)
- [Historical artifacts](gowalla/archive/)
- [Experiment protocol](../docs/experiments.md)
