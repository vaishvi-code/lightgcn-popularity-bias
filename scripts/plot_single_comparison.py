"""Plot a completed single-seed supplemental validation or test comparison."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-split", choices=["validation", "test"], default="validation")
    args = parser.parse_args()
    with args.comparison.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert all(row["evaluation_split"] == args.evaluation_split and row["training_seed"] == "2020" for row in rows)
    values = {(int(row["layers"]), row["layer_combination"]): row for row in rows}
    assert set(values) == {(depth, combination) for depth in range(1, 5) for combination in ["uniform", "single"]}
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("recall_at_20", "Overall Recall@20", False),
        ("tail_share_at_20", "Tail recommendation share", True),
        ("tail_recall_at_20", "Tail Recall@20", False),
        ("coverage_at_20", "Catalog coverage", True),
    ]
    depths = [1, 2, 3, 4]
    for axis, (metric, title, percentage) in zip(axes.flat, metrics):
        for combination, label, color in [
            ("uniform", "Standard LightGCN, averaged layers", "#4a5a8c"),
            ("single", "LightGCN-single, final layer only", "#e07a5f"),
        ]:
            axis.plot(depths, [float(values[(depth, combination)][metric]) for depth in depths], marker="o", linewidth=2, color=color, label=label)
        axis.set(title=title, xlabel="Propagation layers", xticks=depths, ylim=(0, None))
        axis.grid(axis="y", color="#dddddd")
        if percentage:
            axis.yaxis.set_major_formatter(PercentFormatter(1))
    figure.suptitle("What changes when only the final layer is used?", fontsize=18, y=0.99)
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2, fontsize=11)
    figure.text(0.5, 0.02, f"Supplemental {args.evaluation_split} · seed 2020 · best checkpoints · fresh three-layer control", ha="center", color="#666666", fontsize=10)
    figure.tight_layout(rect=(0, 0.05, 1, 0.9))
    figure.savefig(args.output / "single-versus-uniform.png", dpi=240, facecolor="white")
    svg = args.output / "single-versus-uniform.svg"
    figure.savefig(svg, facecolor="white")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(figure)
    print(f"Saved supplemental {args.evaluation_split} comparison figure to", args.output)


if __name__ == "__main__":
    main()
