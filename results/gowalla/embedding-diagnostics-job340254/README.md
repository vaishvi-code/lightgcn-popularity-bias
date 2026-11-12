# Supplemental embedding analysis for Q4

ARC job 340254 completed the analysis for all 15 frozen MF and standard LightGCN checkpoints at seeds 2020, 2021, and 2022. Checkpoint SHA-256 hashes and saved validation-exposure fingerprints matched the frozen references. The fresh uninterrupted K=3 seed-2020 control is used.

This analysis was conducted after the final test comparison. It uses fitting degrees, existing validation-selected checkpoints, and saved validation exposures. It opens no test-label file and changes no model selection.

## Findings

Means below summarize the three training seeds. Sample standard deviations are in [summary.csv](summary.csv).

| Layers | Final averaged norm vs sqrt(degree), Spearman | Learned layer-0 norm vs sqrt(degree), Spearman | Last propagated norm vs sqrt(degree), Spearman | Degree vs validation exposure, Spearman |
| --- | ---: | ---: | ---: | ---: |
| 0, MF | 0.741 | 0.741 | 0.741 | 0.691 |
| 1 | 0.751 | 0.319 | 0.676 | 0.656 |
| 2 | 0.757 | 0.182 | 0.571 | 0.643 |
| 3 | 0.753 | 0.085 | 0.698 | 0.606 |
| 4 | 0.739 | 0.030 | 0.565 | 0.574 |

Final averaged embedding norms are positively associated with degree at every depth. The association is already present in MF and does not increase monotonically as layers increase. The proposal's stronger claim that the relationship becomes progressively closer with depth is therefore unsupported by these measurements.

The learned layer-0 norms become less associated with degree in deeper models, while the combined embeddings retain a strong association. These are trained layer-0 embeddings rather than the original random initialization. The last-layer diagnostic comes from the standard model's learned embeddings. It is separate from training a LightGCN-single model.

Final embedding norm and validation exposure have positive rank correlations around 0.70 across depths. Pearson correlation between final norm and square-root degree is approximately 0.68 for MF and 0.80–0.82 for the graph models, also without a monotonic depth trend.

Degree–exposure rank correlation decreases with depth even though the completed recommendation audit shows greater head concentration. Spearman correlation compares item ordering. Tail share and Gini measure how recommendation slots are distributed. These different summaries can move in different directions. Use the exposure metrics to quantify popularity concentration.

These correlations describe association. Dot-product ranking also depends on user embeddings and their alignment with items, so the analysis does not prove that item norm causes exposure. Degree and its square root have identical ranks, making their Spearman correlations with a given norm equivalent.

## Presentation figures

- [Norm–degree correlations](figures/norm-degree-correlation.png)
- [Norm, popularity, and exposure correlations](figures/norm-exposure-correlation.png)
- [Degree-bin norm profiles](figures/degree-norm-profile.png)

Each figure also has an SVG version. Error bars in correlation figures are sample standard deviations across three training seeds. Degree-bin profiles average the per-seed median norm within fixed fitting-degree bins.

## Artifacts and reproduction

- [Per-checkpoint diagnostics and hashes](per-run.csv)
- [Grouped means and sample standard deviations](summary.csv)
- [Degree-bin summaries](degree-profiles.csv)
- [Method and source metadata](analysis-metadata.json)
- [Supplemental protocol and commands](../../../docs/supplemental-analysis.md)

Full per-item compressed tables remain on ARC under `/mnt/beegfs/vpatel34/lightgcn-supplemental-20261111/results/gowalla/embedding-diagnostics-job340254/`. The versioned summary files and analysis scripts reproduce the reported figures.
