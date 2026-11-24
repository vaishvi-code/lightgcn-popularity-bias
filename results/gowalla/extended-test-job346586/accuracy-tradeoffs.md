# Accuracy cost and tail recovery

Changes relative to standard LightGCN K=3. Recall and exposure changes below are percentage points, computed from matched three-seed test means. These are descriptive comparisons, not significance results or new model selection.

| Model | Overall recall change (points) | Tail recall change (points) | Tail-share change (points) |
| --- | ---: | ---: | ---: |
| BPR-MF | -4.256 | +0.243 | +8.202 |
| LightGCN K=1 | -1.837 | +0.281 | +3.801 |
| LightGCN K=2 | -0.420 | +0.257 | +1.641 |
| LightGCN K=3 | +0.000 | +0.000 | +0.000 |
| LightGCN K=4 | +0.047 | -0.376 | -1.599 |
| LightGCN-single K=1 | -0.760 | -0.211 | -0.070 |
| LightGCN-single K=2 | -0.612 | -0.779 | -2.089 |
| LightGCN-single K=3 | -2.168 | -1.778 | -4.494 |
| LightGCN-single K=4 | -3.827 | -2.426 | -5.764 |
| APPNP beta=0.1 | -0.619 | -0.980 | -3.273 |
| APPNP beta=0.2 | -0.175 | -0.441 | -1.616 |
| APPNP beta=0.5 | -1.380 | +0.353 | +3.323 |
| Normalization a=0.25 b=0.75 | -0.661 | -0.445 | -1.840 |
| Normalization a=0.25 b=1 | -0.757 | -0.795 | -2.153 |
