# Frozen final-test results — November 11, 2026

**Measured and verified:** ARC job 337725 completed successfully (exit 0,
1 minute 57 seconds). All 23 frozen entries reproduced their six saved
validation metrics and exact item exposures before the authors' test set
was opened. The same 23 entries then completed final-test evaluation.
No model was retrained, retuned or reselected after viewing test results.

## Main comparison

Trained models report mean ± sample standard deviation over seeds 2020,
2021 and 2022. MostPopular is one deterministic evaluation. For tail share
and coverage, standard deviations are in percentage points. All metrics
use K=20. Higher recall/NDCG means more relevance hits; more tail exposure
or coverage alone does not establish relevance or fairness. Lower Gini
means less concentrated item exposure.

| Model | Recall@20 | NDCG@20 | Tail share | Coverage | Exposure Gini | Tail Recall@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MostPopular | 0.0416 | 0.0318 | 0.00% | 0.09% | 0.9995 | 0.0000 |
| BPR-MF | 0.1319 ± 0.0041 | 0.1046 ± 0.0034 | 19.56% ± 0.39 | 57.40% ± 0.36 | 0.8562 ± 0.0015 | 0.0400 ± 0.0022 |
| LightGCN K=1 | 0.1561 ± 0.0010 | 0.1323 ± 0.0006 | 15.16% ± 0.53 | 52.56% ± 1.37 | 0.8876 ± 0.0068 | 0.0404 ± 0.0018 |
| LightGCN K=2 | 0.1703 ± 0.0013 | 0.1458 ± 0.0009 | 13.00% ± 0.27 | 46.14% ± 1.38 | 0.9061 ± 0.0046 | 0.0402 ± 0.0011 |
| LightGCN K=3 | 0.1745 ± 0.0010 | 0.1486 ± 0.0007 | 11.36% ± 0.14 | 38.78% ± 0.56 | 0.9249 ± 0.0016 | 0.0376 ± 0.0009 |
| LightGCN K=4 | 0.1750 ± 0.0020 | 0.1484 ± 0.0016 | 9.76% ± 0.18 | 32.57% ± 0.87 | 0.9388 ± 0.0020 | 0.0338 ± 0.0011 |
| APPNP beta=0.5 | 0.1607 ± 0.0011 | 0.1370 ± 0.0006 | 14.68% ± 0.35 | 50.22% ± 0.94 | 0.8923 ± 0.0049 | 0.0411 ± 0.0015 |

## What the final test shows

- Tail share and coverage decrease, and exposure Gini increases, from MF
  through LightGCN K=1–4 in **each of the three seeds**. Mean tail share
  falls from 19.56% (MF) to 11.36% (K=3) and 9.76% (K=4). MF already
  concentrates 80.44% of slots on the popular 20% of items, compared with
  their 53.09% share of fitting interactions. Propagation adds concentration
  under this controlled training and evaluation protocol; it is not the
  sole source of concentration.
- Graph propagation improves overall recall over MF. K=4 has the highest
  numerical mean test Recall@20 (0.1750), narrowly above K=3 (0.1745).
  K=3 leads K=4 in two seeds and has slightly higher mean NDCG; K=4 leads
  recall in seed 2022. This differs from the validation mean ordering.
  These small differences are not an established statistically significant
  advantage, and we do not use them to select a new model.
- K=2 remains a useful accuracy/exposure tradeoff: mean Recall@20 0.1703
  versus 0.1745 for K=3, with higher tail recall (0.0402 versus 0.0376),
  more tail exposure and coverage, and lower Gini. Its recall gap from K=3
  is larger on test than on validation. Tail recall is not monotonic across
  every depth increment, so exposure alone cannot answer whether the
  recommendations for less popular items are useful.
- APPNP-style beta 0.5 improves tail share, coverage, Gini and tail recall
  against standard K=3 **in every seed**, while reducing overall Recall
  and NDCG in every seed. Mean tail share rises from 11.36% to 14.68%,
  tail recall from 0.0376 to 0.0411, and overall recall falls from 0.1745
  to 0.1607. The weighting experiment produced a repeatable tradeoff,
  rather than an improvement on every metric.

## Exploratory variants: seed 2020 only

These variants were not replicated across seeds. The same seed-2020 fresh
K=3 control is included below for a like-for-like comparison; it is also
part of the main three-seed aggregate above. Do not compare one-seed
variants with three-seed means as if their evidence were equally strong.

| Model | Recall@20 | NDCG@20 | Tail share | Coverage | Exposure Gini | Tail Recall@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGCN K=3 | 0.1735 | 0.1479 | 11.41% | 38.32% | 0.9260 | 0.0371 |
| APPNP beta=0.1 | 0.1675 | 0.1410 | 7.89% | 25.41% | 0.9548 | 0.0273 |
| APPNP beta=0.2 | 0.1717 | 0.1460 | 9.65% | 32.29% | 0.9407 | 0.0320 |
| Normalization a=0.5 b=0.75 | 0.1650 | 0.1423 | 10.89% | 35.05% | 0.9347 | 0.0334 |
| Normalization a=0.5 b=1.0 | 0.1544 | 0.1318 | 10.94% | 35.57% | 0.9352 | 0.0317 |

Both smaller APPNP weights and both stronger-normalization variants have
lower Recall/NDCG, tail share, coverage and tail recall, and higher Gini
than their seed-2020 standard K=3 control. This is a finding about these
specific settings on one seed, not a general rejection of weighting or
normalization methods. APPNP beta 0.5 was chosen for replication after
seed-2020 validation; disclose that selection when presenting the study.

## Evaluation and verification details

- Frozen roster: 18 main checkpoint evaluations, 4 exploratory checkpoint
  evaluations and MostPopular. Checkpoints were selected by validation
  Recall@20 during training, with a 500-epoch maximum and patience 20.
  K=3 seed 2020 uses the fresh uninterrupted control.
- Original upstream Gowalla test labels; 29,858 evaluation users and
  26,839 users with test tail positives. There are 597,160 recommendation
  slots per evaluation and 40,981 catalog items. Tail recall averages only
  over users with at least one tail positive.
- Fitting-only graphs and popularity groups. Test ranking excludes the
  union of fitting and validation positives. No unsupported user/item or
  overlapping relevance label was silently removed.
- Frozen plan SHA256:
  `5f459b7479142af789861eb0179fde02a0ac0d76b3b7627dae3a189f3b1fa709`.
  Test-file SHA256:
  `099a2a73924e4b754dc6efd730d764f492321b53c38eb93298850c83acf57be3`.
- All four shared seed-2020 checkpoints match the release manifest hashes.
  Plan, evaluator sources and checkpoint hashes are unchanged between
  validation verification and test evaluation.
- A separate local audit reproduced tail share, coverage and Gini from all
  46 saved exposure tables, checked total slots and user denominators, and
  recomputed summary means and sample standard deviations. Recall, NDCG
  and tail recall are measured by the evaluator; the artifact audit does
  not independently rerank test users.

Three seeds measure training variation on one fixed split. They do not
establish significance, fairness, causality beyond this experimental
setting, or generalization to other datasets. Validation and test relevance
sets, exclusions and tail-user denominators differ; their absolute scores
should not be interpreted as a before/after performance change.

## Artifacts and next step

- [All 23 individual results](test-per-run.csv)
- [Grouped means and sample standard deviations](test-summary.csv)
- [Validation verification gate](validation-verification.json)
- [Final-test completion record](test-completion.json)
- [Independent artifact audit](artifact-audit.json); reproduce it with
  `python results/gowalla/final-test-job337725/audit_artifacts.py` from the repository root
- [ARC completion status](arc-job-status.json) and [execution log](arc-job.log)
- Per-run metrics and full item-exposure counts: `verify/` and `test/`
- [Frozen protocol](../../../docs/final-test-plan.md) and
  [checkpoint release receipt](../../../provenance/seed2020-release-receipt.json)

The frozen final-test plan is complete. Next prepare the final report and
slides using both validation and test findings, retaining the single-seed
labels on exploratory variants. Further model development would need a
new evaluation plan; do not tune these settings on the opened test set.
