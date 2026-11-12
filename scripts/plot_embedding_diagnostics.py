"""Create presentation figures from the supplemental embedding diagnostics."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    output = args.input / "figures"
    output.mkdir(exist_ok=True)
    with (args.input / "summary.csv").open() as stream:
        summary = {(int(row["layers"]), row["metric"]): row for row in csv.DictReader(stream)}
    depths = list(range(5))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

    def save(figure, filename):
        figure.tight_layout()
        figure.savefig(output / (filename + ".png"), dpi=240, facecolor="white")
        svg_path = output / (filename + ".svg")
        figure.savefig(svg_path, facecolor="white")
        svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
        plt.close(figure)

    def correlation_plot(series, title, filename):
        figure, axis = plt.subplots(figsize=(10, 5.6))
        for metric, label, color in series:
            means = [float(summary[(depth, metric)]["mean"]) for depth in depths]
            deviations = [float(summary[(depth, metric)]["sample_std"]) for depth in depths]
            axis.errorbar(depths, means, yerr=deviations, marker="o", capsize=4, linewidth=2, color=color, label=label)
        axis.set(xlabel="Propagation layers (0 = MF)", ylabel="Spearman rank correlation", title=title, xticks=depths, ylim=(-1.05, 1.05))
        axis.axhline(0, color="#999999", linewidth=0.8)
        axis.grid(axis="y", color="#dddddd")
        axis.legend(loc="best")
        figure.text(0.5, 0.01, "Gowalla · 3 training seeds · mean ± sample SD · all 40,981 items", ha="center", fontsize=10, color="#666666")
        figure.subplots_adjust(bottom=0.17)
        save(figure, filename)

    correlation_plot([
        ("rho_raw_norm_sqrt_degree", "Learned layer-0 norm", "#888888"),
        ("rho_combined_norm_sqrt_degree", "Final averaged norm", "#4a5a8c"),
        ("rho_last_norm_sqrt_degree", "Last propagated-layer norm", "#e07a5f"),
    ], "How embedding norms relate to item degree", "norm-degree-correlation")
    correlation_plot([
        ("rho_degree_exposure", "Degree vs. recommendation frequency", "#4a5a8c"),
        ("rho_combined_norm_exposure", "Final norm vs. recommendation frequency", "#e07a5f"),
    ], "Popularity, embedding norm, and validation exposure", "norm-exposure-correlation")

    profiles = defaultdict(list)
    with (args.input / "degree-profiles.csv").open() as stream:
        for row in csv.DictReader(stream):
            profiles[(int(row["layers"]), float(row["degree_lower"]))].append(row)
    figure, axis = plt.subplots(figsize=(10, 5.6))
    colors = ["#888888", "#e07a5f", "#81b29a", "#4a5a8c", "#a16fbb"]
    labels = ["MF", "1 layer", "2 layers", "3 layers", "4 layers"]
    for depth in depths:
        bins = sorted(key[1] for key in profiles if key[0] == depth)
        x = [float(profiles[(depth, lower)][0]["median_degree"]) for lower in bins]
        y = [np.mean([float(row["median_combined_norm"]) for row in profiles[(depth, lower)]]) for lower in bins]
        axis.plot(x, y, marker="o", linewidth=2, color=colors[depth], label=labels[depth])
    axis.set(xlabel="Fitting item degree (log scale)", ylabel="Mean of per-seed median final norms", title="Final embedding magnitude by item popularity")
    axis.set_xscale("symlog", linthresh=1)
    axis.grid(axis="y", color="#dddddd")
    axis.legend()
    figure.text(0.5, 0.01, "Fixed degree bins · validation-selected checkpoints · 3 seeds", ha="center", fontsize=10, color="#666666")
    save(figure, "degree-norm-profile")
    print("Figures saved to", output)


if __name__ == "__main__":
    main()
