# Three-seed Gowalla validation comparison

Mean ± sample standard deviation across training seeds 2020, 2021 and 2022. The fitting/validation split is fixed. Tail share and coverage are percentages; their standard deviations are percentage points. These are validation results; final test evaluation remains pending.

| Model | Recall@20 | NDCG@20 | Tail share | Coverage | Exposure Gini | Tail Recall@20 |
| --- | --- | --- | --- | --- | --- | --- |
| BPR-MF | 0.1577 ± 0.0044 | 0.0799 ± 0.0024 | 19.36 ± 0.38% | 57.19 ± 0.35% | 0.8575 ± 0.0015 | 0.0562 ± 0.0023 |
| LightGCN K=1 | 0.1815 ± 0.0032 | 0.0957 ± 0.0016 | 14.96 ± 0.52% | 52.31 ± 1.37% | 0.8890 ± 0.0067 | 0.0552 ± 0.0029 |
| LightGCN K=2 | 0.1950 ± 0.0034 | 0.1027 ± 0.0019 | 12.81 ± 0.27% | 45.83 ± 1.38% | 0.9074 ± 0.0045 | 0.0553 ± 0.0027 |
| LightGCN K=3 | 0.1957 ± 0.0025 | 0.1025 ± 0.0014 | 11.18 ± 0.13% | 38.45 ± 0.55% | 0.9261 ± 0.0015 | 0.0507 ± 0.0012 |
| LightGCN K=4 | 0.1932 ± 0.0027 | 0.1004 ± 0.0014 | 9.59 ± 0.18% | 32.24 ± 0.89% | 0.9398 ± 0.0019 | 0.0432 ± 0.0016 |
| APPNP beta=0.5 | 0.1864 ± 0.0044 | 0.0979 ± 0.0018 | 14.49 ± 0.34% | 49.92 ± 0.91% | 0.8936 ± 0.0048 | 0.0566 ± 0.0029 |

K=3 uses the fresh seed-2020 control, excluding the continued legacy checkpoint. Other APPNP weights and the two normalization variants remain single-seed experiments. The APPNP beta=0.5 replication was selected after observing seed-2020 validation results; it is not a three-seed comparison of the full variant grid.

Configuration/source/split hashes, checkpoint selection and stopping histories were verified. Reloaded accuracy agrees with training history. Exposure counts independently reproduce tail share, coverage and Gini; tail recall is the evaluator measurement. Three seeds quantify variation across these runs, not statistical significance or robustness to different data splits.
