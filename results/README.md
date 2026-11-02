# Results

The `gowalla/` directory contains completed data-preparation and MostPopular measurements, plus a preliminary LightGCN training snapshot. These were transcribed from the ARC terminal output supplied by the team. `metrics-template.csv` remains a header-only format template.

| File | Contents |
| --- | --- |
| `gowalla/split-metadata.json` | Fixed fitting/validation split counts and original-training hash |
| `gowalla/popularity-summary.json` | Popular/tail catalog and interaction shares |
| `gowalla/mostpopular-validation.json` | Completed MostPopular validation evaluation |
| `gowalla/lightgcn-k3-training-snapshot.json` | Last supplied observation at epoch 16 of a requested 200; one seed |
| `gowalla/lightgcn-k3-history-excerpt.csv` | Observed epochs 12–16 only, not a complete learning curve |

The full ARC history, saved plots, item-level popularity/exposure tables, environment freeze, and upstream commit file have not been transferred. Null LightGCN exposure values mean not yet measured. The epoch-16 snapshot is not a final or best-checkpoint result, nor a live job monitor.

Scripts recovered from the shared terminal logs are under `scripts/`. Their base path was adjusted for the repository's `scripts/` directory. The batch script is adapted to accept `PROJECT_ROOT`; the currently running ARC job has not been modified.

Store small, reviewable result tables and final plots here. Raw outputs under `raw/` are excluded from Git, as are model checkpoints and training logs.

For each run, record the model/configuration, dataset, evaluation split, split seed, training seed, selected epoch, metric values, runtime, and code commit. Leave inapplicable fields such as beta blank for models that do not use them.

Report the actual seed count and mean/standard deviation for repeated runs. Label preliminary or unconverged measurements and keep validation and test results distinct. Published reference scores must be separately attributed.
