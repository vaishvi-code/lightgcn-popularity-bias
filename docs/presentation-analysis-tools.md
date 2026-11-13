# Local analysis figures and illustrative example

`scripts/build_presentation_analysis.py` reads the frozen main test summaries and the supplemental single-output test comparison. It creates an overall-recall versus tail-recall chart with sample SD, a matched-seed normalization comparison, and a matched-seed layer-averaging comparison. It does not retrain, tune, or modify the original results.

Run in an environment with NumPy and Matplotlib:

```bash
python scripts/build_presentation_analysis.py --repository . --output output/presentation-strengthening
```

The plotting code was executed on ARC. Presentation assets and speaker notes stay under ignored `output/`, outside GitHub.

## Optional illustrative case

`scripts/extract_recommendation_example.py` performs inference for one deterministic user selected from fitting histories. Among tail-oriented users under the frozen grouping rule, it minimizes distance to the group's median fitting-history length, then distance to its median tail fraction, then user ID. The selection record is written before opening test labels. The user must not be replaced based on recommendations or hits.

It uses frozen seed-2020 K=2, K=4 and APPNP beta=0.5 checkpoints, verifies their identities against existing test records, masks fitting plus validation positives, and ranks the full catalog. To retain the main evaluator's matrix multiplication batch shape, it evaluates the selected user's existing 256-user test batch and extracts that user's row. This is an illustrative post-hoc case, not a new aggregate model comparison or evidence of representativeness.

Arguments are `--repository`, `--project-root`, one or more `--checkpoint-root` directories, and a new `--output` directory. It requires the pinned upstream checkout, original prepared data, PyTorch and a GPU allocation. Set the output to an ignored location such as `.local/recommendation-example`. It writes raw dataset IDs and mappings: these are private local/ARC artifacts and must not be added to the public repo.

Pass the resulting private JSON as `--example .local/recommendation-example/example-private.json` to the plotting script to create local top-10/top-20 figures. Figure labels use consistent aliases, not actual venue names. Hits represent held-out check-ins; an unobserved recommendation is not necessarily disliked. The case can fail to recover any held-out visits, and must be reported honestly if that happens.

No individual example outputs or presentation files are published here. The main and grouped aggregate results remain the evidence for population-level conclusions.
