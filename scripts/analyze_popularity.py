import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

base = Path(__file__).resolve().parent.parent
source = base / "prepared/gowalla/train.txt"
output = base / "results/gowalla"
output.mkdir(parents=True, exist_ok=True)

degree = Counter()
for line in source.read_text().splitlines():
    values = list(map(int, line.split()))
    if values:
        degree.update(values[1:])

# Highest degree first; item ID breaks ties deterministically.
ranked = sorted(degree, key=lambda item: (-degree[item], item))
n_items = len(ranked)
assert n_items > 0, "No fitting items found"

n_head = math.ceil(0.20 * n_items)
head = set(ranked[:n_head])
total = sum(degree.values())
head_interactions = sum(degree[item] for item in head)

summary = {
    "dataset": "gowalla",
    "popularity_source": "fitting interactions only",
    "fitting_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "head_rule": "top ceil(20% of items); degree descending, item ID ascending",
    "items": n_items,
    "popular_items": n_head,
    "tail_items": n_items - n_head,
    "fitting_interactions": total,
    "popular_item_catalog_share": n_head / n_items,
    "popular_item_interaction_share": head_interactions / total,
    "tail_item_interaction_share": 1 - head_interactions / total,
}

with (output / "item-popularity.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["item_id", "fitting_degree", "group"])
    for item in sorted(degree):
        writer.writerow([
            item, degree[item],
            "popular" if item in head else "tail"
        ])

(output / "popularity-summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ranks = np.arange(1, n_items + 1)
counts = np.array([degree[item] for item in ranked])
axes[0].loglog(ranks, counts, color="#2563eb")
axes[0].axvline(n_head, color="#dc2626", linestyle="--",
               label="Popular/tail boundary")
axes[0].set_xlabel("Item rank by fitting degree")
axes[0].set_ylabel("Fitting interaction count")
axes[0].set_title("Gowalla item popularity")
axes[0].legend()

positions = np.arange(2)
catalog = [n_head / n_items, 1 - n_head / n_items]
interactions = [head_interactions / total, 1 - head_interactions / total]

axes[1].bar(positions - 0.18, np.array(catalog) * 100,
            width=0.36, label="Catalog share", color="#64748b")
axes[1].bar(positions + 0.18, np.array(interactions) * 100,
            width=0.36, label="Interaction share", color="#2563eb")
axes[1].set_xticks(positions, ["Popular", "Tail"])
axes[1].set_ylabel("Percent")
axes[1].set_ylim(0, 100)
axes[1].set_title("Catalog size versus fitting interactions")
axes[1].legend()

fig.tight_layout()
fig.savefig(output / "popularity-distribution.png", dpi=200)
plt.close(fig)

print(json.dumps(summary, indent=2))
print("Saved figure:", output / "popularity-distribution.png")
print("Saved item labels:", output / "item-popularity.csv")
