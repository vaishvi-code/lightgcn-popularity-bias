"""Rebuild expanded validation comparisons from frozen references and follow-up runs."""
import argparse
import csv
import json
import statistics
from pathlib import Path

METRICS = ("recall_at_20", "ndcg_at_20", "tail_share_at_20", "coverage_at_20", "exposure_gini", "tail_recall_at_20")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/gowalla/proposal-completion-20261114"))
    args = parser.parse_args()
    out = args.results.resolve()
    repo = out.parents[2]
    original = json.loads((repo / "provenance/final-test-plan.json").read_text())
    followup = json.loads((repo / "provenance/proposal-completion-plan.json").read_text())
    selection = json.loads((out / "normalization-selection.json").read_text())
    records, seen = [], {}

    def add(model, run, seed, metrics):
        if run in seen:
            assert all(seen[run][metric] == metrics[metric] for metric in METRICS)
            return
        seen[run] = metrics
        records.append(dict(model=model, run_id=run, training_seed=seed, evaluation_split="validation", **{metric: metrics[metric] for metric in METRICS}))

    for entry in original["evaluations"]:
        if entry["comparison_group"] == "main_three_seed":
            add(entry["model"], entry["run_id"], entry["training_seed"], entry["validation_metrics"])
    extra = [json.loads(path.read_text()) for group in ["replication", "selected"] for path in sorted((out / "state" / group).glob("*.json"))]
    extra += [entry for entry in followup["references"] if entry["kind"] in ["single", "appnp"]]
    extra += selection["selected_seed2020"] + selection["reused_replications"]
    for entry in extra:
        if entry["kind"] == "single":
            label = f"LightGCN-single K={entry['layers']}"
        elif entry["kind"] == "appnp":
            label = f"APPNP beta={entry['beta']:g}"
        elif entry["a"] == entry["b"] == .5:
            label = "LightGCN K=3"
        else:
            label = f"Normalization a={entry['a']:g} b={entry['b']:g}"
        add(label, entry["run_id"], entry["seed"], entry["metrics"])
    aggregate = []
    for model in dict.fromkeys(row["model"] for row in records):
        rows = [row for row in records if row["model"] == model]
        assert sorted(row["training_seed"] for row in rows) == [2020, 2021, 2022]
        for metric in METRICS:
            values = [row[metric] for row in rows]
            aggregate.append(dict(model=model, metric=metric, evaluation_split="validation", seed_count=3, mean=statistics.mean(values), sample_sd=statistics.stdev(values)))
    assert len(records) == 42 and len(aggregate) == 84
    for name, rows in [("expanded-validation-per-run.csv", records), ("expanded-validation-summary.csv", aggregate)]:
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    grid = [json.loads(path.read_text()) for path in sorted((out / "state/grid").glob("*.json"))]
    grid += [entry for entry in followup["references"] if entry["kind"] == "normalization" and entry["seed"] == 2020]
    with (out / "normalization-grid.csv").open("w", newline="") as stream:
        rows = [dict(a=entry["a"], b=entry["b"], training_seed=2020, run_id=entry["run_id"], selected=entry in selection["selected_seed2020"], **{metric: entry["metrics"][metric] for metric in METRICS}) for entry in sorted(grid, key=lambda entry: (entry["b"], entry["a"]))]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# Expanded three-seed validation comparison", "", "All values are mean ± sample SD across training seeds 2020–2022 on the fixed validation split. Tail share and coverage use percentages, with SD in percentage points. These are not new test results.", "", "| Model | Recall@20 | NDCG@20 | Tail share | Coverage | Gini | Tail Recall@20 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model in dict.fromkeys(row["model"] for row in records):
        cells = []
        for metric in METRICS:
            row = next(row for row in aggregate if row["model"] == model and row["metric"] == metric)
            if metric in ["tail_share_at_20", "coverage_at_20"]:
                cells.append(f"{100 * row['mean']:.2f}% ± {100 * row['sample_sd']:.2f}")
            else:
                cells.append(f"{row['mean']:.4f} ± {row['sample_sd']:.4f}")
        lines.append("| " + model + " | " + " | ".join(cells) + " |")
    lines += ["", "This combines the six original main settings with eight additional settings. Standard K=3 is reused rather than counted twice. Selected normalizations use the same a at all seeds, chosen on seed-2020 validation. APPNP beta=0.1 seed2021 reached the 500-epoch cap (best epoch 498); all other new runs stopped by patience. Sample SD is not a significance test.", ""]
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    index = ["# Inspect every new run without ARC", "", "All links below point to files stored in GitHub. There are 22 new runs: six search fits, twelve independent seed replications and four selected-normalization replications. Original reused runs remain in their existing result directories. Checkpoint weights are not part of this export.", ""]
    for group, heading in [("grid", "Normalization search"), ("replication", "Independent additional seeds"), ("selected", "Selected-normalization additional seeds")]:
        index += ["## " + heading, "", "| Variant | Seed | Best / completed epoch | Stop reason | Validation Recall@20 | Files |", "| --- | ---: | ---: | --- | ---: | --- |"]
        for state_path in sorted((out / "state" / group).glob("*.json"), key=lambda path: int(path.stem)):
            row = json.loads(state_path.read_text())
            run = out / "runs" / row["run_id"]
            history = json.loads((run / "run-summary.json").read_text())
            variant = f"{row['kind']}, K={row['layers']}, a={row['a']:g}, b={row['b']:g}"
            if row["beta"] is not None:
                variant += f", beta={row['beta']:g}"
            links = [f"[{label}](runs/{row['run_id']}/{name})" for label, name in [("config", "config.json"), ("history", "history.csv"), ("metrics", "best-validation-metrics.json"), ("exposure", "best-validation-exposure.csv"), ("provenance", "followup-provenance.json")]]
            logs = list((out / "logs").glob(group + "-*_" + state_path.stem + ".out"))
            assert len(logs) == 1
            links.append(f"[log](logs/{logs[0].name})")
            index.append(f"| {variant} | {row['seed']} | {history['best_epoch']} / {history['completed_epochs']} | {history['stop_reason']} | {row['metrics']['recall_at_20']:.6f} | " + " · ".join(links) + " |")
        index.append("")
    index += ["The reused seed-2020 and standard-control run IDs and exact validation metrics are recorded in the frozen [plan](../../../provenance/proposal-completion-plan.json). Search selection and final summaries are linked from the [result report](README.md).", ""]
    (out / "runs.md").write_text("\n".join(index), encoding="utf-8")
    print("Rebuilt 42 matched validation runs, 84 summary rows and 9 grid candidates.")


if __name__ == "__main__":
    main()
