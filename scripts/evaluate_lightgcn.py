import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

base = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description="Evaluate a trusted local LightGCN checkpoint on validation data")
parser.add_argument("--run-dir", type=Path,
                    default=base / "results/gowalla/lightgcn-k3-seed2020-20261102-164455")
args = parser.parse_args()
run = args.run_dir.resolve()
data = base / "prepared/gowalla"
output = base / "results/gowalla"

# This is our own locally generated checkpoint.
checkpoint = torch.load(
    run / "best.pt", map_location="cpu", weights_only=False
)
metadata = checkpoint["metadata"]

fit_hash = hashlib.sha256((data / "train.txt").read_bytes()).hexdigest()
val_hash = hashlib.sha256((data / "val.txt").read_bytes()).hexdigest()
assert fit_hash == metadata["fitting_file_sha256"]
assert val_hash == metadata["validation_file_sha256"]

summary = json.loads((output / "popularity-summary.json").read_text())
assert fit_hash == summary["fitting_file_sha256"]

loader_data = base / "prepared" / (
    "gowalla-validation-" + fit_hash[:12] + "-" + val_hash[:12]
)

# The upstream parser reads sys.argv when world is imported.
sys.argv = ["evaluate_lightgcn.py"]
sys.path.insert(0, str(base / "LightGCN-PyTorch/code"))

import world
world.config.update(metadata["config"])
world.tensorboard = 0
torch.set_num_threads(4)

from dataloader import Loader
from model import LightGCN

assert torch.cuda.is_available(), "Run inside a GPU allocation"

dataset = Loader(config=world.config, path=str(loader_data))
network = LightGCN(world.config, dataset).to(world.device)
network.load_state_dict(checkpoint["model_state_dict"])
network.eval()

with (output / "item-popularity.csv").open() as file:
    rows = list(csv.DictReader(file))

catalog = {int(row["item_id"]) for row in rows}
assert catalog == set(range(dataset.m_items)), "Catalog mismatch"
tail = {int(row["item_id"]) for row in rows if row["group"] == "tail"}

k = 20
users = sorted(dataset.testDict)
exposure = np.zeros(dataset.m_items, dtype=np.int64)
recalls, ndcgs, tail_recalls = [], [], []
tail_slots = 0

with torch.no_grad():
    for start in range(0, len(users), 256):
        batch = users[start:start + 256]
        user_tensor = torch.tensor(batch, device=world.device)
        scores = network.getUsersRating(user_tensor)

        assert torch.isfinite(scores).all(), "Non-finite model scores"

        for row_index, user in enumerate(batch):
            seen = dataset.allPos[user]
            scores[row_index, seen] = -torch.inf

        recommendations = torch.topk(scores, k=k, dim=1).indices.cpu().numpy()

        for user, items in zip(batch, recommendations):
            relevant = set(dataset.testDict[user])
            seen = set(dataset.allPos[user])
            assert relevant.isdisjoint(seen)
            assert seen.isdisjoint(items.tolist())

            hits = [int(item in relevant) for item in items]
            recalls.append(sum(hits) / len(relevant))

            dcg = sum(hit / math.log2(rank + 2)
                      for rank, hit in enumerate(hits))
            ideal = sum(1 / math.log2(rank + 2)
                        for rank in range(min(k, len(relevant))))
            ndcgs.append(dcg / ideal)

            np.add.at(exposure, items, 1)
            tail_slots += sum(int(item) in tail for item in items)

            relevant_tail = relevant & tail
            if relevant_tail:
                tail_recalls.append(
                    len(set(items.tolist()) & relevant_tail)
                    / len(relevant_tail)
                )

        if start % 5120 == 0:
            print(f"Evaluated {start + len(batch)}/{len(users)} users",
                  flush=True)

counts = np.sort(exposure).astype(np.float64)
n = len(counts)
slots = int(exposure.sum())
gini = (
    2 * np.dot(np.arange(1, n + 1), counts) / (n * slots)
    - (n + 1) / n
)

results = {
    "model": metadata.get("model", "LightGCN"),
    "dataset": "gowalla",
    "evaluation_split": "validation",
    "checkpoint": "best.pt; selected by validation Recall@20",
    "checkpoint_epoch": checkpoint["epoch"],
    "layers": metadata["config"]["lightGCN_n_layers"],
    "training_seed": metadata["training_seed"],
    "seed_count": 1,
    "top_k": k,
    "evaluation_users": len(users),
    "fitting_file_sha256": fit_hash,
    "validation_file_sha256": val_hash,
    "recall_at_20": float(np.mean(recalls)),
    "ndcg_at_20": float(np.mean(ndcgs)),
    "tail_share_at_20": tail_slots / slots,
    "coverage_at_20": float(np.count_nonzero(exposure) / n),
    "distinct_recommended_items": int(np.count_nonzero(exposure)),
    "exposure_gini": float(gini),
    "tail_recall_at_20": (
        float(np.mean(tail_recalls)) if tail_recalls else None
    ),
    "tail_evaluation_users": len(tail_recalls),
}

(run / "best-validation-metrics.json").write_text(
    json.dumps(results, indent=2) + "\n"
)

with (run / "best-validation-exposure.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["item_id", "recommendation_count"])
    writer.writerows(enumerate(exposure.tolist()))

print(json.dumps(results, indent=2))
