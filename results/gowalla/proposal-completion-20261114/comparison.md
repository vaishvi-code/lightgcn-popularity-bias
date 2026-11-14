# Expanded three-seed validation comparison

All values are mean ± sample SD across training seeds 2020–2022 on the fixed validation split. Tail share and coverage use percentages, with SD in percentage points. These are not new test results.

| Model | Recall@20 | NDCG@20 | Tail share | Coverage | Gini | Tail Recall@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BPR-MF | 0.1577 ± 0.0044 | 0.0799 ± 0.0024 | 19.36% ± 0.38 | 57.19% ± 0.35 | 0.8575 ± 0.0015 | 0.0562 ± 0.0023 |
| LightGCN K=1 | 0.1815 ± 0.0032 | 0.0957 ± 0.0016 | 14.96% ± 0.52 | 52.31% ± 1.37 | 0.8890 ± 0.0067 | 0.0552 ± 0.0029 |
| LightGCN K=2 | 0.1950 ± 0.0034 | 0.1027 ± 0.0019 | 12.81% ± 0.27 | 45.83% ± 1.38 | 0.9074 ± 0.0045 | 0.0553 ± 0.0027 |
| LightGCN K=3 | 0.1957 ± 0.0025 | 0.1025 ± 0.0014 | 11.18% ± 0.13 | 38.45% ± 0.55 | 0.9261 ± 0.0015 | 0.0507 ± 0.0012 |
| LightGCN K=4 | 0.1932 ± 0.0027 | 0.1004 ± 0.0014 | 9.59% ± 0.18 | 32.24% ± 0.89 | 0.9398 ± 0.0019 | 0.0432 ± 0.0016 |
| APPNP beta=0.5 | 0.1864 ± 0.0044 | 0.0979 ± 0.0018 | 14.49% ± 0.34 | 49.92% ± 0.91 | 0.8936 ± 0.0048 | 0.0566 ± 0.0029 |
| LightGCN-single K=1 | 0.1927 ± 0.0007 | 0.1009 ± 0.0004 | 11.12% ± 0.05 | 38.18% ± 0.46 | 0.9210 ± 0.0011 | 0.0491 ± 0.0009 |
| LightGCN-single K=2 | 0.1848 ± 0.0032 | 0.0955 ± 0.0019 | 9.12% ± 0.39 | 24.04% ± 1.47 | 0.9532 ± 0.0029 | 0.0392 ± 0.0018 |
| APPNP beta=0.1 | 0.1826 ± 0.0017 | 0.0949 ± 0.0011 | 7.94% ± 0.29 | 25.64% ± 1.02 | 0.9544 ± 0.0023 | 0.0351 ± 0.0012 |
| APPNP beta=0.2 | 0.1915 ± 0.0023 | 0.0998 ± 0.0015 | 9.57% ± 0.19 | 32.66% ± 1.12 | 0.9401 ± 0.0024 | 0.0437 ± 0.0011 |
| LightGCN-single K=3 | 0.1582 ± 0.0009 | 0.0794 ± 0.0004 | 6.75% ± 0.14 | 15.31% ± 0.21 | 0.9715 ± 0.0004 | 0.0250 ± 0.0007 |
| LightGCN-single K=4 | 0.1367 ± 0.0001 | 0.0674 ± 0.0003 | 5.51% ± 0.15 | 11.33% ± 0.12 | 0.9793 ± 0.0003 | 0.0165 ± 0.0004 |
| Normalization a=0.25 b=0.75 | 0.1866 ± 0.0017 | 0.0961 ± 0.0010 | 9.36% ± 0.15 | 36.81% ± 0.89 | 0.9374 ± 0.0020 | 0.0443 ± 0.0012 |
| Normalization a=0.25 b=1 | 0.1800 ± 0.0026 | 0.0925 ± 0.0016 | 9.05% ± 0.27 | 33.66% ± 1.33 | 0.9447 ± 0.0032 | 0.0384 ± 0.0013 |

This combines the six original main settings with eight additional settings. Standard K=3 is reused rather than counted twice. Selected normalizations use the same a at all seeds, chosen on seed-2020 validation. APPNP beta=0.1 seed2021 reached the 500-epoch cap (best epoch 498); all other new runs stopped by patience. Sample SD is not a significance test.
