"""Scientific figures for the complete matched three-seed test extension."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from build_extended_test_report import ORDER


def main(root):
    output = root / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    def save(fig, name):
        fig.tight_layout(rect=(0, .055, 1, .95))
        fig.savefig(output / (name + ".png"), dpi=220, facecolor="white")
        path = output / (name + ".svg")
        fig.savefig(path, facecolor="white")
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        plt.close(fig)
    with (root / "comparison-summary.csv").open() as stream:
        values = {(r["model"], r["metric"]): r for r in csv.DictReader(stream)}
    labels = [m.replace("LightGCN-single", "Final layer").replace("LightGCN", "Layer average").replace("Normalization a=0.25 ", "Norm ") for m in ORDER]
    def color(model):
        if model.startswith("LightGCN-single"): return "#e07a5f"
        if model.startswith("APPNP"): return "#81b29a"
        if model.startswith("Normalization"): return "#a16fbb"
        return "#888888" if model == "BPR-MF" else "#4a5a8c"
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    y = np.arange(len(ORDER))
    for ax, metric, title in zip(axes, ["recall_at_20", "tail_recall_at_20"], ["Overall relevance", "Relevant tail recovery"]):
        points = [values[m, metric] for m in ORDER]
        ax.barh(y, [100*float(p["mean"]) for p in points],
                xerr=[100*float(p["sample_sd"]) for p in points], capsize=3,
                color=[color(m) for m in ORDER], alpha=.9)
        ax.set(title=title, xlabel=("Recall@20 (%)" if metric == "recall_at_20" else "Tail Recall@20 (%)"), yticks=y, yticklabels=labels)
        ax.grid(axis="x", alpha=.25)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle("Complete matched test comparison")
    fig.text(.5, .015, "Gowalla test · 3 training seeds · mean +/- sample SD · original results preserved; supplemental evaluations are post-hoc", ha="center", fontsize=10)
    save(fig, "complete-test-comparison")
    with (root / "user-groups-summary.csv").open() as stream:
        groups = {(r["model"], r["group"], r["metric"]): r for r in csv.DictReader(stream)}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for col, (group, title) in enumerate([("head_oriented", "Mostly popular"), ("mixed", "Mixed"), ("tail_oriented", "Mostly tail")]):
        for row, metric in enumerate(["recall_at_20", "tail_recall_at_20"]):
            ax = axes[row, col]
            for prefix, label, c in [("LightGCN", "Layer average", "#4a5a8c"), ("LightGCN-single", "Final layer only", "#e07a5f")]:
                points = [groups[f"{prefix} K={k}", group, metric] for k in range(1, 5)]
                ax.errorbar(range(1, 5), [100*float(p["mean"]) for p in points],
                            yerr=[100*float(p["sample_sd"]) for p in points], marker="o", capsize=3, color=c, label=label)
            ax.set(title=title if row == 0 else "", ylabel="Overall Recall@20 (%)" if row == 0 else "Tail Recall@20 (%)", xticks=range(1, 5), xlabel="Layers" if row else "")
            ax.grid(axis="y", alpha=.25)
    axes[0, 0].legend()
    fig.suptitle("Layer aggregation across user histories: test outcomes")
    fig.text(.5, .015, "Gowalla test · mean +/- sample SD across 3 seeds · fitting-history groups · post-hoc analysis", ha="center", fontsize=10)
    save(fig, "user-history-test-comparison")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    main(parser.parse_args().results)
