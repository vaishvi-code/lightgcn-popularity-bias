"""Figures for completed three-seed validation follow-up results."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, required=True)
args = parser.parse_args()
with (args.results / "expanded-validation-summary.csv").open() as stream:
    rows = list(csv.DictReader(stream))
destination = args.results / "figures"
destination.mkdir(exist_ok=True)
navy, coral = "#4F608F", "#FC795F"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def point(model, metric):
    row = next(row for row in rows if row["model"] == model and row["metric"] == metric)
    return 100 * float(row["mean"]), 100 * float(row["sample_sd"])


def save(fig, name):
    for extension in ["png", "svg"]:
        path = destination / (name + "." + extension)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
for axis, metric, title in zip(axes, ["recall_at_20", "tail_recall_at_20", "tail_share_at_20"], ["Overall Recall@20", "Tail Recall@20", "Tail share@20"]):
    for prefix, label, color in [("LightGCN K=", "Average all layers", navy), ("LightGCN-single K=", "Final layer only", coral)]:
        values = [point(prefix + str(k), metric) for k in range(1, 5)]
        axis.errorbar(range(1, 5), [value[0] for value in values], yerr=[value[1] for value in values], marker="o", capsize=3, label=label, color=color)
    axis.set(title=title, xlabel="Propagation layers", ylabel="Percent", xticks=range(1, 5))
    axis.grid(axis="y", alpha=.2)
fig.suptitle("Layer averaging preserves more tail relevance across three seeds")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
fig.text(.5, .075, "Validation only • seeds 2020–2022 • bars: sample SD • post-hoc follow-up", ha="center", fontsize=9)
fig.tight_layout(rect=(0, .13, 1, .94))
save(fig, "single-versus-averaging")

for name, models, labels, title in [
    ("appnp-three-seed", ["LightGCN K=3", "APPNP beta=0.1", "APPNP beta=0.2", "APPNP beta=0.5"], ["Uniform", "β=0.1", "β=0.2", "β=0.5"], "APPNP-style weights: the trade-off depends on beta"),
    ("selected-normalization-three-seed", ["LightGCN K=3", "Normalization a=0.25 b=0.75", "Normalization a=0.25 b=1"], ["a=0.5\nb=0.5", "a=0.25\nb=0.75", "a=0.25\nb=1.0"], "Selected normalizations still trail the standard model"),
]:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for axis, metric, heading in zip(axes, ["recall_at_20", "tail_recall_at_20", "tail_share_at_20"], ["Overall Recall@20", "Tail Recall@20", "Tail share@20"]):
        values = [point(model, metric) for model in models]
        axis.bar(range(len(models)), [value[0] for value in values], yerr=[value[1] for value in values], capsize=4, color=[navy] + [coral] * (len(models) - 1), width=.6)
        axis.set(title=heading, ylabel="Percent", xticks=range(len(models)), xticklabels=labels)
        axis.grid(axis="y", alpha=.2); axis.set_axisbelow(True)
    fig.suptitle(title)
    note = "a selected on seed-2020 validation" if name.startswith("selected") else "post-hoc replication"
    fig.text(.5, .025, "Validation only • K=3 • seeds 2020–2022 • bars: sample SD • " + note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .94))
    save(fig, name)
