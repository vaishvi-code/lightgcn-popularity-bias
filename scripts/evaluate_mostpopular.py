import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

base = Path(__file__).resolve().parent.parent
data = base / "prepared/gowalla"
output = base / "results/gowalla"
k = 20

def load(path):
    result = {}
    for line in path.read_text().splitlines():
        values = list(map(int, line.split()))
        if values:
            result[values[0]] = set(values[1:])
    return result

def accuracy(recommendations, relevant):
    hits = [int(item in relevant) for item in recommendations]
    recall = sum(hits) / len(relevant)
    dcg = sum(hit / math.log2(rank + 2)
              for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2)
                for rank in range(min(len(relevant), len(recommendations))))
    return recall, dcg / ideal
# Hand-computed metric check before evaluating the dataset.
recall, ndcg = accuracy([10, 20, 30], {20, 40})
assert math.isclose(recall, 0.5)
expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
assert math.isclose(ndcg, expected)

fit = load(data / "train.txt")
validation = load(data / "val.txt")

summary = json.loads((output / "popularity-summary.json").read_text())
fit_hash = hashlib.sha256((data / "train.txt").read_bytes()).hexdigest()
assert fit_hash == summary["fitting_file_sha256"], "Popularity labels are stale"

with (output / "item-popularity.csv").open() as file:
    rows = list(csv.DictReader(file))

degrees = {int(row["item_id"]): int(row["fitting_degree"]) for row in rows}
tail = {int(row["item_id"]) for row in rows if row["group"] == "tail"}
ranked = sorted(degrees, key=lambda item: (-degrees[item], item))

recalls, ndcgs, tail_recalls = [], [], []
exposure = Counter()
tail_slots = 0
start = time.perf_counter()

for user in sorted(validation):
    relevant = validation[user]
    seen = fit.get(user, set())
    assert relevant.isdisjoint(seen)

    recommendations = []
    for item in ranked:
        if item not in seen:
            recommendations.append(item)
            if len(recommendations) == k:
                break

    assert len(recommendations) == k
    assert len(set(recommendations)) == k
    assert seen.isdisjoint(recommendations)

    recall, ndcg = accuracy(recommendations, relevant)
    recalls.append(recall)
    ndcgs.append(ndcg)

    exposure.update(recommendations)
    tail_slots += sum(item in tail for item in recommendations)

    relevant_tail = relevant & tail
    if relevant_tail:
        tail_recalls.append(
            len(set(recommendations) & relevant_tail) / len(relevant_tail)
        )

counts = sorted(exposure[item] for item in ranked)
n = len(counts)
slots = sum(counts)
gini = (
    2 * sum(index * count for index, count in enumerate(counts, 1))
    / (n * slots)
    - (n + 1) / n
)

results = {
    "model": "MostPopular",
    "dataset": "gowalla",
    "evaluation_split": "validation",
    "top_k": k,
    "split_seed": 2020,
    "training": "none; deterministic degree ranking",
    "fitting_file_sha256": fit_hash,
    "evaluation_users": len(recalls),
    "recall_at_20": sum(recalls) / len(recalls),
    "ndcg_at_20": sum(ndcgs) / len(ndcgs),
    "tail_share_at_20": tail_slots / slots,
    "coverage_at_20": len(exposure) / n,
    "exposure_gini": gini,
    "tail_recall_at_20": (
        sum(tail_recalls) / len(tail_recalls) if tail_recalls else None
    ),
    "tail_evaluation_users": len(tail_recalls),
    "evaluation_seconds": time.perf_counter() - start,
}

(output / "mostpopular-validation.json").write_text(
    json.dumps(results, indent=2) + "\n"
)

with (output / "mostpopular-exposure.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["item_id", "recommendation_count"])
    for item in sorted(degrees):
        writer.writerow([item, exposure[item]])

print(json.dumps(results, indent=2))
print("Metric checks passed. Results saved to:", output)
