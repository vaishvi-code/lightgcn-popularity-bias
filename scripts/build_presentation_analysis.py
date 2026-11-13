"""Build local presentation figures from saved results; never publish individual cases."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

NAVY, CORAL, GREEN, GRAY = "#4F608F", "#FC795F", "#7A9B76", "#EDF0F6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def save(fig, root, name):
    for extension in ["png", "svg"]:
        path = root / (name + "." + extension)
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--example", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = args.repository / "results/gowalla"
    summary = read(results / "final-test-job337725/test-summary.csv")
    main_rows = [row for row in summary if row["comparison_group"] == "main_three_seed"]
    models = list(dict.fromkeys(row["model"] for row in main_rows))
    labels = ["MF", "LightGCN K=1", "LightGCN K=2", "LightGCN K=3", "LightGCN K=4", "APPNP β=0.5"]
    offsets = [(-12, -28), (-72, 18), (-20, 24), (-98, -20), (12, -8), (12, 18)]
    colors = ["#8791A7", "#7697BD", NAVY, "#353D65", "#9083A4", CORAL]
    fig, axis = plt.subplots(figsize=(11.5, 6.8))
    table = []
    for model, label, offset, color in zip(models, labels, offsets, colors):
        x = next(row for row in main_rows if row["model"] == model and row["metric"] == "recall_at_20")
        y = next(row for row in main_rows if row["model"] == model and row["metric"] == "tail_recall_at_20")
        vx, vy = 100 * float(x["mean"]), 100 * float(y["mean"])
        sx, sy = 100 * float(x["sample_std"]), 100 * float(y["sample_std"])
        axis.errorbar(vx, vy, xerr=sx, yerr=sy, fmt="o", ms=9, color=color, capsize=4, elinewidth=1.5)
        axis.annotate(label, (vx, vy), xytext=offset, textcoords="offset points", color=color, weight="bold", fontsize=12,
                      arrowprops=dict(arrowstyle="-", color=color, alpha=.5))
        table.append(dict(model=model, recall_percent=vx, recall_sd_percentage_points=sx, tail_recall_percent=vy, tail_recall_sd_percentage_points=sy))
    axis.set(xlabel="Overall Recall@20 (%) → higher is better", ylabel="Tail Recall@20 (%) → higher is better", title="Better overall recall does not guarantee better tail recall")
    axis.grid(alpha=.2)
    axis.margins(x=.18, y=.35)
    fig.text(.5, .015, "Gowalla test • three training seeds • error bars: sample SD • same split and users", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, 1))
    save(fig, args.output, "accuracy-tail-tradeoff")
    with (args.output / "accuracy-tail-tradeoff.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)

    per_run = read(results / "final-test-job337725/test-per-run.csv")
    reference = next(row for row in per_run if row["run_id"] == "lightgcn-k3-fresh-seed2020-job329225")
    variants = [reference] + [next(row for row in per_run if row["model"] == name) for name in ["Normalization a=0.5 b=0.75", "Normalization a=0.5 b=1.0"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for axis, metric, title in zip(axes, ["recall_at_20", "tail_recall_at_20"], ["Overall Recall@20 (%)", "Tail Recall@20 (%)"]):
        bars = axis.bar(["Standard\nb=0.5", "Stronger\nb=0.75", "Stronger\nb=1.0"], [100 * float(row[metric]) for row in variants], color=[NAVY, CORAL, "#E4AB9D"], width=.6)
        axis.bar_label(bars, fmt="%.2f", padding=5)
        axis.set_title(title); axis.set_ylim(0, max(bar.get_height() for bar in bars) * 1.2)
        axis.grid(axis="y", alpha=.15); axis.set_axisbelow(True)
    fig.suptitle("Stronger neighbor down-weighting did not improve this comparison", fontsize=16)
    fig.text(.5, .015, "Gowalla test • K=3 • a fixed at 0.5 • matched seed 2020 only • no error bars", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .93))
    save(fig, args.output, "normalization-result")

    single = read(results / "supplemental-single-test-job340785/comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for axis, metric, title in zip(axes, ["recall_at_20", "tail_recall_at_20"], ["Overall Recall@20 (%)", "Tail Recall@20 (%)"]):
        for combination, label, color, offset in [("uniform", "Average all layers", NAVY, -.18), ("single", "Use final layer only", CORAL, .18)]:
            values = [100 * float(next(row for row in single if row["layer_combination"] == combination and int(row["layers"]) == k)[metric]) for k in range(1, 5)]
            axis.bar(np.arange(1, 5) + offset, values, width=.36, label=label, color=color)
        axis.set_title(title); axis.set_xticks(range(1, 5)); axis.set_xlabel("Propagation layers")
        axis.grid(axis="y", alpha=.15); axis.set_axisbelow(True)
    fig.suptitle("Keeping earlier layers helps tail recovery at every tested depth", fontsize=16)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.text(.5, .065, "Matched seed 2020 • supplemental post-hoc test • single-output models trained separately", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .13, 1, .93))
    save(fig, args.output, "layer-averaging-result")

    if args.example:
        case = json.loads(args.example.read_text())
        rows = [dict(model=model["model"], hits=model["hits"], tail_slots=model["tail_slots"], tail_hits=model["tail_hits"], recall_at_20=model["recall_at_20"], tail_recall_at_20=model["tail_recall_at_20"]) for model in case["models"]]
        with (args.output / "illustrative-example-summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        for limit in [10, 20]:
            fig, axes = plt.subplots(1, 3, figsize=(14, 6.8 if limit == 10 else 10.5))
            for axis, model in zip(axes, case["models"]):
                axis.set(xlim=(0, 1), ylim=(0, limit + 2)); axis.axis("off")
                title = {"LightGCN K=2": "LightGCN: 2 layers", "LightGCN K=4": "LightGCN: 4 layers", "APPNP beta=0.5": "APPNP weights: β=0.5"}[model["model"]]
                axis.text(.02, limit + 1.6, title, fontsize=15, weight="bold", color="#303550")
                axis.text(.02, limit + .85, f"Top 20: {model['tail_slots']} tail places; {model['hits']} hits; {model['tail_hits']} tail hits", fontsize=10)
                for row in model["rankings"][:limit]:
                    y = limit - row["rank"] + .15
                    box = FancyBboxPatch((.01, y), .97, .75, boxstyle="round,pad=0.015", facecolor="#E4EAF7" if row["tail"] else "#F0F1F4", edgecolor=CORAL if row["test_hit"] else "white", linewidth=2)
                    axis.add_patch(box)
                    text = f"{row['rank']:02d}   {row['display_name']}   {'TAIL' if row['tail'] else 'POPULAR'}"
                    axis.text(.04, y + .38, text, va="center", fontsize=12, color="#303550")
                    if row["test_hit"]:
                        axis.text(.95, y + .38, "★", va="center", ha="right", color=CORAL, fontsize=16)
            fig.suptitle("One user: how do the recommended places change?", fontsize=19, color="#303550")
            fig.text(.5, .055, "Blue = less-popular place • gray = popular place • coral outline / ★ = held-out visit recovered", ha="center", fontsize=11)
            fig.text(.5, .018, f"Fitting history: {case['selection']['fitting_interactions']} places, {100 * case['selection']['fitting_tail_fraction']:.0f}% tail • illustrative seed 2020 • top {limit} shown • aliases, not venue names", ha="center", fontsize=10)
            fig.tight_layout(rect=(0, .09, 1, .94))
            save(fig, args.output, "recommendation-example-top" + str(limit))


if __name__ == "__main__":
    main()
