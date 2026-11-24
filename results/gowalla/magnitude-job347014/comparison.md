# Item magnitude intervention: validation results

Three training seeds, fixed existing checkpoints. Primary comparison: raw dot product versus unit-length item vectors with the same user vectors. Sigmoid is a separate numerical control, not the intervention.

| Model | Scoring | Recall@20 | NDCG@20 | Tail Recall@20 | Tail share (%) | Coverage (%) | Gini |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MF | Raw dot baseline | 0.15771 +/- 0.00434 | 0.07991 +/- 0.00235 | 0.05621 +/- 0.00224 | 19.37571 +/- 0.38300 | 57.20700 +/- 0.34586 | 0.85738 +/- 0.00145 |
| MF | Equal-length items | 0.11954 +/- 0.00362 | 0.05754 +/- 0.00163 | 0.10038 +/- 0.00446 | 59.19240 +/- 1.37381 | 96.46340 +/- 0.21393 | 0.56441 +/- 0.00670 |
| MF | Legacy sigmoid check | 0.15772 +/- 0.00437 | 0.07992 +/- 0.00235 | 0.05620 +/- 0.00226 | 19.36382 +/- 0.38326 | 57.19236 +/- 0.34986 | 0.85746 +/- 0.00146 |
| LightGCN K=3 | Raw dot baseline | 0.19578 +/- 0.00245 | 0.10268 +/- 0.00139 | 0.05070 +/- 0.00119 | 11.15882 +/- 0.13291 | 38.02575 +/- 0.58387 | 0.92659 +/- 0.00158 |
| LightGCN K=3 | Equal-length items | 0.14807 +/- 0.00398 | 0.07105 +/- 0.00194 | 0.12639 +/- 0.00397 | 57.58725 +/- 0.24096 | 97.12794 +/- 0.10373 | 0.51305 +/- 0.00209 |
| LightGCN K=3 | Legacy sigmoid check | 0.19569 +/- 0.00248 | 0.10250 +/- 0.00140 | 0.05075 +/- 0.00122 | 11.18210 +/- 0.13092 | 38.44790 +/- 0.54854 | 0.92609 +/- 0.00154 |
| LightGCN-single K=3 | Raw dot baseline | 0.16004 +/- 0.00072 | 0.08314 +/- 0.00036 | 0.02554 +/- 0.00055 | 6.62826 +/- 0.14759 | 11.85021 +/- 0.22909 | 0.97373 +/- 0.00052 |
| LightGCN-single K=3 | Equal-length items | 0.06279 +/- 0.00054 | 0.02587 +/- 0.00041 | 0.09236 +/- 0.00130 | 94.33725 +/- 0.06479 | 46.51017 +/- 0.37495 | 0.86016 +/- 0.00176 |
| LightGCN-single K=3 | Legacy sigmoid check | 0.15818 +/- 0.00089 | 0.07936 +/- 0.00044 | 0.02496 +/- 0.00071 | 6.75319 +/- 0.14022 | 15.31279 +/- 0.20841 | 0.97154 +/- 0.00045 |
