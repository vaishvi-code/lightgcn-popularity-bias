# Supplemental proposal work

The frozen 23-entry final-test comparison is complete. The work below addresses two remaining proposal items after those test results were available. It is reported separately from that comparison.

## Q4 embedding diagnostics

Analyze the same 15 frozen checkpoints for MF and standard LightGCN depths 1–4 at seeds 2020, 2021, and 2022. Use their saved validation exposure counts. Each checkpoint and exposure fingerprint must match the frozen records before analysis.

Measure item-level Spearman correlations for degree versus exposure, layer-0 norm versus square-root degree, final averaged norm versus square-root degree, last propagated-layer norm versus square-root degree, and final norm versus exposure. Also report Pearson correlation between final norm and square-root degree. Save per-item measurements, degree-bin summaries, and means with sample standard deviations over seeds.

Degree and square-root degree have identical rankings. Their Spearman correlations with a given norm are therefore equivalent. The Pearson diagnostic addresses the proposed square-root scaling separately. Correlation identifies an association rather than proving that embedding magnitude causes recommendation exposure. Dot-product scores also depend on user embeddings and alignment.

The analysis uses existing checkpoints and validation exposures. It opens no test-label file and changes no model or checkpoint selection.

The analysis completed successfully in ARC job 340254. [Results and presentation figures](../results/gowalla/embedding-diagnostics-job340254/README.md) show positive norm–degree associations but no monotonic increase with propagation depth.

```bash
python scripts/analyze_embeddings.py --project-root /path/to/training/workspace \
  --repository-root /path/to/repository --checkpoint-root /path/to/run/directories \
  --output /path/to/new/diagnostic/directory
python scripts/plot_embedding_diagnostics.py --input /path/to/diagnostic/directory
```

Repeat `--checkpoint-root` for additional readable checkpoint locations. The analysis requires all 15 standard checkpoints. The ARC verification script lists the team's current locations.

## LightGCN-single validation ablation

Train depths 1–4 with only the final propagated embedding contributing to recommendation scores. Keep fitting graph, validation split, seed 2020, embedding size 64, BPR loss, Adam learning rate 0.001, batch size 1024, regularization 0.0001, standard normalization, maximum 500 epochs, and early-stopping patience 20 fixed.

The model is implemented in `single_lightgcn.py`. It inherits the original parameter initialization and BPR regularization. Only the embedding combination changes. Its combination weights are zero for layers 0 through K−1 and one for layer K. A zero-layer instance is equivalent to the ordinary zero-layer model.

The training command adds `--layer-combination single`. The setting is saved in checkpoints and reconstructed by validation evaluation. Resume rejects a change in combination. Existing checkpoints without that field default to the original uniform combination.

This is a seed-2020 exploratory validation study conducted after the original final test was opened. It is excluded from the frozen final-test roster. Further seeds and any new test evaluation would require a separately documented protocol.

## Verification and ARC jobs

The numerical checks compare final-layer embeddings and gradients to dense graph propagation at depths 0–4, confirm matching initialization and regularization, and verify checkpoint reloads. The end-to-end check trains two epochs and reloads its best checkpoint through the validation evaluator before the full runs are submitted.

```bash
python tests/check_embedding_metrics.py
python tests/check_single_lightgcn.py LightGCN-PyTorch/code
sbatch --output="$HOME/supplemental-check-%j.out" scripts/check_supplemental_arc.sh
# After the verification job completes successfully:
sbatch --output="$HOME/lightgcn-single-%A_%a.out" scripts/train_single_arc.sh
```

The ARC scripts use a separate `lightgcn-supplemental-20261111` workspace. The training array runs at most two depths concurrently. Its outputs are validation measurements and include the study phase in metadata.

Verification job 340254 passed the numerical tests, two-epoch training, and best-checkpoint validation reload. Array job 340255 then submitted the full seed-2020 depths as tasks 340255_1 through 340255_4. Their trained results are pending. The short check is stored separately as `results/gowalla/lightgcn-single-k1-seed2020-check-job340254/` and is excluded from model-performance comparisons.
