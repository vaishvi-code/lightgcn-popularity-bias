# Frozen final-test protocol — November 11, 2026

This protocol is fixed before inspecting final test metrics. It evaluates
validation-selected checkpoints as they are, without retraining, changing
hyperparameters, choosing new checkpoints, or selecting a lucky seed.
Models were trained on the fitting subset, not on the entire original train
set. The original author test split remains reserved until verification passes.

## Frozen roster

- Main comparison: MF, LightGCN K=1–4 and APPNP beta 0.5, each at seeds
  2020, 2021 and 2022 (18 checkpoint evaluations).
- Exploratory comparison: APPNP beta 0.1/0.2 and normalization (a,b)=(0.5,0.75)
  and (0.5,1.0), seed 2020 only (4 checkpoint evaluations).
- MostPopular: one deterministic evaluation using fitting degrees.

The [machine-readable plan](../provenance/final-test-plan.json) pins all 23
entries, source run IDs, configurations, selected epochs and expected validation
metrics/exposure fingerprints. K=3 seed 2020 uses the fresh control. APPNP
beta 0.5 was selected for replication after seed-2020 validation; disclose this.
All trained entries use their existing best.pt, selected by validation Recall@20.

## Ranking and metrics

Use the fixed full catalog of 40,981 items; top K=20. Score using each
checkpoint's saved graph normalization and layer weighting. Construct graphs
from fitting interactions only, exactly as during training. Use the original
upstream test positives as relevance labels. Rank for every user with at least
one test positive; assert that every such user/item is supported by the model.
Do not silently filter unsupported users/items or overlapping positives.

Mask the union of fitting and validation positives for test ranking. Preserve
fitting-only popularity groups: top ceil(20%) by fitting degree, breaking ties
by item ID. MostPopular ranks by fitting degree, then ascending item ID.
Neural models retain the existing sigmoid scoring and torch.topk behavior.

Report per-user macro Recall@20 and NDCG@20, tail recommendation-slot share,
full-catalog coverage, exposure Gini including zero-exposure items, and tail
Recall@20 averaged only over users with at least one test tail positive.
Save both evaluation-user counts, per-item recommendation counts, checkpoint
hashes, split/test-file hashes, and evaluator/plan hashes.

## Verification and execution gate

Hand-calculated tests cover accuracy, tail-user denominators, coverage/Gini,
and the difference between validation and test exclusions. Next reproduce
all six saved validation metrics (absolute tolerance 1e-6) and exact item
exposure counts for every frozen entry. The verification phase never opens
test labels. Missing checkpoints are recorded as unavailable, not replaced.
Only an all-passed gate with unchanged evaluator, plan and checkpoint hashes
permits the test phase. Outputs use new job-specific directories, preserving
all validation artifacts. A discrepancy stops execution for investigation;
it is not a reason to tune models against test results.

Four historical seed-2020 files are owned by vpatel34 and not readable from
stpatel4: mf-k0-seed2020-job327067, lightgcn-k1-seed2020-job327365,
lightgcn-k2-seed2020-job327366, lightgcn-k4-seed2020-job327367. Place their
original best.pt files in the final-evaluation checkpoint store, each under
its exact run ID directory, or supply an accessible shared location. No
other files from that account are required. Existing saved validation
metrics/configurations will be used to verify identity and reconstruction.

## Reporting and interpretation

Report means and sample standard deviations over all three seeds for the
main comparison, single-run values for exploratory variants, and one value
for MostPopular. Preserve all individual run values. Do not combine unequal
seed counts into an apparent three-seed ranking. Three seeds measure training
variation on this split, not significance or generalization across splits.

The planned questions are whether depth still concentrates exposure, whether
K=2 retains a useful accuracy/tail-relevance compromise, and whether APPNP
beta 0.5 retains its tradeoff against K=3. Report differences from validation
without further test-driven tuning. Test metrics are pending until the full
verification gate passes and final evaluation completes.

## ARC commands and current verification

Use the existing portable environment and frozen model helpers, with new
outputs under `$HOME/lightgcn-final-test-20261111/results/`.

```bash
# Validation-only reproduction (does not open test labels):
sbatch --export=ALL,RUN_FINAL_TEST=0 --output="$HOME/final-verify-%j.log" "$HOME/lightgcn-final-test-20261111/scripts/evaluate_final_arc.sh"
# Once all four missing checkpoint files have been supplied:
# Reverify the complete roster, then evaluate test only if every check passes.
sbatch --job-name=final-test --export=ALL,RUN_FINAL_TEST=1 --output="$HOME/final-test-%j.log" "$HOME/lightgcn-final-test-20261111/scripts/evaluate_final_arc.sh"
```

Validation-only job 336563 was submitted on November 11, 2026. The evaluator
records missing historical checkpoints and checks the available models.
Local checks for metric arithmetic and exclusion rules passed. A synthetic
summary check also verifies sample standard deviations, separate single-seed
reporting, and rejection of incomplete evaluation reports. This synthetic
check uses no real test outcomes.


Verification job 336563 finished: 19 entries passed all six metric checks
and exact exposure-count reproduction (18 trained checkpoints plus
MostPopular). Four entries are missing only their original checkpoint files.
The [saved audit](../provenance/final-validation-verification-job336563.json)
records plan/source/checkpoint hashes and per-entry status. The launcher
exited with status 2 as designed for an incomplete roster; there were no
metric mismatches. No test labels were opened and no final-test results
were produced. The final-run command above is ready after the missing files
are supplied, and will reverify all 23 entries before opening test labels.
