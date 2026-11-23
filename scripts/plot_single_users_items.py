"""Plot matched validation user groups and item correlations for layer aggregation."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(root):
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    def save(fig, name):
        fig.tight_layout(rect=(0, .05, 1, .95))
        fig.savefig(figures / (name + ".png"), dpi=220, facecolor="white")
        fig.savefig(figures / (name + ".svg"), facecolor="white")
        svg = figures / (name + ".svg")
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        plt.close(fig)
    with (root / "user-groups-summary.csv").open() as stream:
        grouped = {(r["family"], int(r["layers"]), r["group"], r["metric"]): r for r in csv.DictReader(stream)}
    groups = [("head_oriented", "Mostly popular"), ("mixed", "Mixed"), ("tail_oriented", "Mostly tail")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for column, (group, title) in enumerate(groups):
        for row, (metric, label) in enumerate([("recall_at_20", "Overall Recall@20 (%)"), ("tail_recall_at_20", "Tail Recall@20 (%)")]):
            ax = axes[row, column]
            for family, name, color in [("uniform", "Layer average", "#4a5a8c"), ("single", "Final layer only", "#e07a5f")]:
                values = [grouped[(family, k, group, metric)] for k in range(1, 5)]
                ax.errorbar(range(1, 5), [100 * float(v["mean"]) for v in values],
                            yerr=[100 * float(v["sample_sd"]) for v in values], marker="o", capsize=3, color=color, label=name)
            ax.set(title=title if row == 0 else "", ylabel=label, xticks=range(1, 5), xlabel="Layers" if row else "")
            ax.grid(axis="y", alpha=.25)
    axes[0, 0].legend()
    fig.suptitle("Layer aggregation and user history: matched validation comparison")
    fig.text(.5, .01, "Gowalla validation · mean ± sample SD across 3 training seeds · groups use fitting history only", ha="center")
    save(fig, "user-history-comparison")
    with (root / "item-correlations-summary.csv").open() as stream:
        items = {(r["family"], int(r["layers"]), r["metric"]): r for r in csv.DictReader(stream)}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (metric, title) in zip(axes, [("rho_combined_norm_sqrt_degree", "Final norm vs. popularity"),
                                         ("rho_raw_norm_sqrt_degree", "Learned pre-propagation norm vs. popularity"),
                                         ("rho_degree_exposure", "Popularity vs. recommendation frequency")]):
        for family, name, color in [("uniform", "Layer average", "#4a5a8c"), ("single", "Final layer only", "#e07a5f")]:
            values = [items[(family, k, metric)] for k in range(1, 5)]
            ax.errorbar(range(1, 5), [float(v["mean"]) for v in values], yerr=[float(v["sample_sd"]) for v in values], marker="o", capsize=3, color=color, label=name)
        ax.set(title=title, xlabel="Layers", ylabel="Spearman correlation", xticks=range(1, 5), ylim=(-1.05, 1.05))
        ax.grid(axis="y", alpha=.25)
    axes[0].legend()
    fig.text(.5, .01, "Gowalla validation exposure · all catalog items · mean ± sample SD across 3 training seeds · associations, not causes", ha="center", fontsize=10)
    save(fig, "item-correlation-comparison")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    main(parser.parse_args().results)
