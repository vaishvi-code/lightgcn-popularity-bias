"""Scientific validation figures for the fixed item-magnitude intervention."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from build_magnitude_report import MODELS


def main(root):
    output=root/"figures";output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    def save(fig,name):
        fig.tight_layout(rect=(0,.07,1,.92));fig.savefig(output/(name+".png"),dpi=220,facecolor="white")
        path=output/(name+".svg");fig.savefig(path,facecolor="white");path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n");plt.close(fig)
    with (root/"comparison-summary.csv").open() as stream:values={(r["model"],r["mode"],r["metric"]):r for r in csv.DictReader(stream)}
    fig,axes=plt.subplots(1,3,figsize=(14,5));x=np.arange(3);labels=["MF","LightGCN\nK=3","Final layer\nK=3"]
    for ax,metric,title in zip(axes,["recall_at_20","tail_recall_at_20","tail_share_at_20"],["Overall Recall@20 (%)","Tail Recall@20 (%)","Tail recommendation share (%)"]):
        for mode,label,color,offset in [("raw_dot","Raw-dot baseline","#4a5a8c",-.18),("unit_item","Equal-length items","#e07a5f",.18)]:
            points=[values[m,mode,metric] for m in MODELS]
            ax.bar(x+offset,[100*float(p["mean"]) for p in points],width=.34,yerr=[100*float(p["sample_sd"]) for p in points],capsize=3,color=color,label=label)
        ax.set(title=title,xticks=x,xticklabels=labels);ax.grid(axis="y",alpha=.25);ax.set_axisbelow(True)
    axes[0].legend();fig.suptitle("Sensitivity to item embedding magnitude")
    fig.text(.5,.02,"Gowalla validation · fixed checkpoints · 3 seeds · mean +/- sample SD · scoring intervention, no retraining",ha="center",fontsize=10)
    save(fig,"magnitude-relevance-exposure")
    with (root/"paired-summary.csv").open() as stream:effects={(r["model"],r["comparison"],r["metric"]):r for r in csv.DictReader(stream)}
    fig,ax=plt.subplots(figsize=(8,5))
    for comparison,label,color,offset in [("unit_item_vs_raw_dot","Equal-length items","#e07a5f",-.18),("legacy_sigmoid_vs_raw_dot","Sigmoid numerical control","#81b29a",.18)]:
        points=[effects[m,comparison,"mean_top20_overlap"] for m in MODELS]
        ax.bar(x+offset,[100*float(p["mean"]) for p in points],width=.34,yerr=[100*float(p["sample_sd"]) for p in points],capsize=3,color=color,label=label)
    ax.set(xticks=x,xticklabels=labels,ylabel="Original top-20 items retained (%)",ylim=(0,105),title="How much do recommendation lists change?")
    ax.legend();ax.grid(axis="y",alpha=.25);ax.set_axisbelow(True)
    fig.text(.5,.02,"Validation-user mean overlap · 3 seeds · error bars show training-seed SD",ha="center",fontsize=10)
    save(fig,"magnitude-ranking-overlap")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("results",type=Path);main(parser.parse_args().results)
