"""Plot grouped test metrics with sample SD across three training seeds."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("results", type=Path)
args = parser.parse_args()
with (args.results / "summary.csv").open() as stream:
    rows = list(csv.DictReader(stream))
models = list(dict.fromkeys(row["model"] for row in rows))
groups = ["head_oriented", "mixed", "tail_oriented"]
labels = ["Mostly popular histories", "Mixed histories", "Mostly tail histories"]
colors = ["#4F608F", "#FC795F", "#7A9B76"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for axis, metric, title in zip(axes, ["recall_at_20", "tail_recall_at_20", "tail_share_at_20"], ["Overall relevance: Recall@20", "Tail relevance: Tail Recall@20", "Exposure: Tail share@20"]):
    for index, (group, label, color) in enumerate(zip(groups, labels, colors)):
        records = [next(row for row in rows if row["model"] == model and row["group"] == group and row["metric"] == metric) for model in models]
        axis.errorbar(np.arange(len(models)), [100 * float(row["mean"]) for row in records], yerr=[100 * float(row["sample_sd"]) for row in records], marker="o", capsize=3, label=label, color=color)
    axis.set_xticks(np.arange(len(models)), ["MF", "K1", "K2", "K3", "K4", "APPNP\nβ=0.5"])
    axis.set_title(title, fontsize=11)
    axis.set_ylabel("Percent")
    axis.grid(axis="y", alpha=.2)
    axis.spines[["top", "right"]].set_visible(False)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
fig.suptitle("Who benefits? Frozen models evaluated by fitting-history group", fontsize=14)
fig.text(.5, .08, "Test set • 3 training seeds • bars: sample SD • post-hoc descriptive analysis", ha="center", fontsize=9)
fig.tight_layout(rect=(0, .12, 1, .93))
destination = args.results / "figures"
destination.mkdir(exist_ok=True)
for extension in ["png", "svg"]:
    path = destination / ("user-group-comparison." + extension)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    if extension == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
