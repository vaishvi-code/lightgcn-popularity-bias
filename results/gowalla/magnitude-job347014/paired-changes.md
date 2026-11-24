# Paired changes from removing item magnitude

Mean +/- sample SD of within-checkpoint changes across three training seeds. Recall, tail recall and tail-share changes are percentage points. Top-20 overlap is the percentage of original recommended items retained, averaged over validation users. Group-level ranking changes are in ranking-changes.json.

| Model | Recall change | Tail recall change | Tail-share change | Top-20 retained (%) | Top-1 changed (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MF | -3.816 +/- 0.116 | 4.417 +/- 0.331 | 39.817 +/- 1.124 | 45.960 +/- 0.327 | 77.469 +/- 0.744 |
| LightGCN K=3 | -4.772 +/- 0.164 | 7.569 +/- 0.283 | 46.428 +/- 0.216 | 33.053 +/- 0.616 | 96.354 +/- 0.113 |
| LightGCN-single K=3 | -9.725 +/- 0.045 | 6.682 +/- 0.090 | 87.709 +/- 0.095 | 4.628 +/- 0.101 | 99.860 +/- 0.029 |
