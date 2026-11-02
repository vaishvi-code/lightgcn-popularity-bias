# Results

The `gowalla/` directory contains completed data-preparation, MostPopular, and best-checkpoint LightGCN validation measurements, plus historical training snapshots. These were transcribed from the ARC terminal output supplied by the team. `metrics-template.csv` remains a header-only format template.

| File | Contents |
| --- | --- |
| `gowalla/split-metadata.json` | Fixed fitting/validation split counts and original-training hash |
| `gowalla/popularity-summary.json` | Popular/tail catalog and interaction shares |
| `gowalla/mostpopular-validation.json` | Completed MostPopular validation evaluation |
| `gowalla/lightgcn-k3-best-validation.json` | Accuracy and exposure of epoch 192, selected after the 200-epoch run; one seed |
| `gowalla/comparison.csv` | MostPopular versus the selected LightGCN checkpoint |
| `gowalla/lightgcn-k3-training-snapshot.json` | Historical epoch-16 snapshot, superseded by the completed evaluation |
| `gowalla/lightgcn-k3-history-excerpt.csv` | Observed excerpts from training, with gaps; not a complete learning curve |

The full ARC history, saved plots, item-level popularity/exposure tables, environment freeze, and upstream commit file have not been transferred. Null exposure values in the historical epoch-16 snapshot mean they had not been measured at that point. Use `lightgcn-k3-best-validation.json` for the completed evaluation. Finishing 200 epochs does not establish convergence; all reported LightGCN findings use one seed and validation data. The final test set remains untouched.

Scripts recovered from the shared terminal logs are under `scripts/`. Their base path was adjusted for the repository's `scripts/` directory. The batch script is adapted to accept `PROJECT_ROOT`; the ARC training and evaluation artifacts have not been modified.

Store small, reviewable result tables and final plots here. Raw outputs under `raw/` are excluded from Git, as are model checkpoints and training logs.

For each run, record the model/configuration, dataset, evaluation split, split seed, training seed, selected epoch, metric values, runtime, and code commit. Leave inapplicable fields such as beta blank for models that do not use them.

Report the actual seed count and mean/standard deviation for repeated runs. Label preliminary or unconverged measurements and keep validation and test results distinct. Published reference scores must be separately attributed.
