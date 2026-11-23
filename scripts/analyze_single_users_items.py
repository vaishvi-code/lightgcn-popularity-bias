"""Post-hoc validation user/item diagnostics for frozen single and averaged models."""
import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from embedding_metrics import embedding_diagnostics
from final_metrics import METRICS, RankingMetrics, load_interactions
from user_groups import GROUPS, assign_group


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, keys, metrics):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in keys)].append(row)
    result = []
    for identity, members in sorted(buckets.items()):
        assert sorted(row["training_seed"] for row in members) == [2020, 2021, 2022]
        for metric in metrics:
            values = [row[metric] for row in members]
            valid = all(value is not None for value in values)
            result.append(dict(zip(keys, identity), metric=metric, seed_count=3,
                               mean=statistics.mean(values) if valid else None,
                               sample_sd=statistics.stdev(values) if valid else None))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, base = args.repository, args.project_root
    plan_path = repo / "provenance/single-users-items-plan.json"
    plan = json.loads(plan_path.read_text())
    assert plan["evaluation_split"] == "validation" and not plan["new_test_evaluation"]
    entries = plan["evaluations"]
    assert len(entries) == 24
    assert {(e["family"], e["layers"], e["training_seed"]) for e in entries} == {
        (f, k, seed) for f in ["single", "uniform"] for k in range(1, 5)
        for seed in [2020, 2021, 2022]}
    # Preflight every checkpoint before writing an incomplete output directory.
    paths = {}
    for entry in entries:
        candidates = [root / entry["run_id"] / "best.pt" for root in args.checkpoint_root]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError("Missing checkpoint: " + entry["run_id"])
        assert digest(path) == entry["checkpoint_sha256"], entry["run_id"]
        paths[entry["run_id"]] = path
    upstream = base / "LightGCN-PyTorch"
    assert subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip() == plan["upstream_commit"]
    subprocess.run(["git", "-C", str(upstream), "diff", "--exit-code", "HEAD", "--",
                    "code/model.py", "code/dataloader.py"], check=True)
    prepared = base / "prepared/gowalla"
    for name, key in [("train.txt", "fitting_file_sha256"), ("val.txt", "validation_file_sha256")]:
        assert digest(prepared / name) == plan[key]
    fit, validation = (load_interactions(prepared / name) for name in ["train.txt", "val.txt"])
    popularity = repo / "results/gowalla/item-popularity.csv"
    assert digest(popularity) == plan["popularity_file_sha256"]
    with popularity.open() as stream:
        items = list(csv.DictReader(stream))
    n = plan["catalog_size"]
    assert [int(row["item_id"]) for row in items] == list(range(n))
    degree = np.array([int(row["fitting_degree"]) for row in items])
    tail = {int(row["item_id"]) for row in items if row["group"] == "tail"}
    assert tail == set(range(n)) - set(sorted(range(n), key=lambda i: (-degree[i], i))[:math.ceil(n * .2)])
    groups = {u: assign_group(len(history & tail), len(history)) for u, history in fit.items()}
    users = sorted(u for u, truth in validation.items() if truth)
    assert set(users) <= groups.keys()
    assert all(validation[u].isdisjoint(fit[u]) for u in users)
    profiles = []
    for group in GROUPS:
        chosen = [u for u in users if groups[u] == group]
        profiles.append(dict(group=group, evaluation_users=len(chosen),
                             tail_evaluation_users=sum(bool(validation[u] & tail) for u in chosen),
                             mean_fitting_tail_fraction=statistics.mean(len(fit[u] & tail) / len(fit[u]) for u in chosen)))
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(upstream / "code"))
    import world
    from dataloader import Loader
    from model import LightGCN
    from single_lightgcn import SingleLayerOutputLightGCN
    assert torch.cuda.is_available(), "Run within a GPU allocation"
    world.device = torch.device("cuda")
    world.tensorboard = 0
    torch.set_num_threads(4)
    loader_path = base / "prepared" / ("gowalla-validation-" + plan["fitting_file_sha256"][:12] + "-" + plan["validation_file_sha256"][:12])
    assert digest(loader_path / "train.txt") == plan["fitting_file_sha256"]
    assert digest(loader_path / "test.txt") == plan["validation_file_sha256"]
    dataset = Loader(config=entries[0]["config"], path=str(loader_path))
    assert np.array_equal(np.asarray(dataset.UserItemNet.sum(axis=0)).ravel(), degree)
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "group-profiles.csv", profiles)
    grouped_rows, item_rows, audits = [], [], []
    for entry in entries:
        run = entry["run_id"]
        checkpoint = torch.load(paths[run], map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == entry["checkpoint_epoch"]
        assert checkpoint["metadata"]["config"] == entry["config"]
        assert checkpoint["metadata"]["training_seed"] == entry["training_seed"]
        for key in ["fitting_file_sha256", "validation_file_sha256"]:
            assert checkpoint["metadata"][key] == plan[key]
        world.config.update(entry["config"])
        cls = SingleLayerOutputLightGCN if entry["family"] == "single" else LightGCN
        model = cls(world.config, dataset).to(world.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        total = RankingMetrics(n, tail)
        accumulators = {group: RankingMetrics(n, tail) for group in GROUPS}
        with torch.no_grad():
            final_users, final_items = model.computer()
            raw_norm = torch.linalg.vector_norm(model.embedding_item.weight, dim=1).cpu().numpy()
            final_norm = torch.linalg.vector_norm(final_items, dim=1).cpu().numpy()
            propagated = None
            if entry["family"] == "single":
                # This architecture's final output IS the last propagated layer.
                # Reuse that tensor rather than performing a second CUDA sparse
                # propagation, which can differ slightly through reduction order.
                last_norm = final_norm.copy()
            else:
                propagated = torch.cat([model.embedding_user.weight, model.embedding_item.weight])
                for _ in range(model.n_layers):
                    propagated = torch.sparse.mm(model.Graph, propagated)
                last_norm = torch.linalg.vector_norm(propagated[dataset.n_users:], dim=1).cpu().numpy()
            for offset in range(0, len(users), 256):
                batch = users[offset:offset + 256]
                scores = torch.sigmoid(final_users[torch.tensor(batch, device=world.device)] @ final_items.T)
                assert torch.isfinite(scores).all()
                for row, user in enumerate(batch):
                    scores[row, list(fit[user])] = -torch.inf
                recommendations = torch.topk(scores, k=20, dim=1).indices.cpu().numpy()
                for user, recs in zip(batch, recommendations):
                    total.add(recs, validation[user], fit[user])
                    accumulators[groups[user]].add(recs, validation[user], fit[user])
        overall = total.finish()
        reference = entry["reference_metrics"]
        deltas = {key: overall[key] - reference[key] for key in METRICS}
        assert all(abs(value) <= plan["metric_absolute_tolerance"] for value in deltas.values()), (run, deltas)
        with (repo / entry["reference_exposure"]).open() as stream:
            exposure_rows = list(csv.DictReader(stream))
        assert [int(row["item_id"]) for row in exposure_rows] == list(range(n))
        expected = np.array([int(row["recommendation_count"]) for row in exposure_rows])
        difference = int(np.abs(expected - total.exposure).sum())
        assert difference <= plan["exposure_l1_tolerance"], (run, difference)
        assert np.array_equal(sum(acc.exposure for acc in accumulators.values()), total.exposure)
        identity = {key: entry[key] for key in ["family", "layers", "training_seed", "run_id"]}
        grouped = {group: acc.finish() for group, acc in accumulators.items()}
        for metric, denominator in [("recall_at_20", "evaluation_users"), ("ndcg_at_20", "evaluation_users"), ("tail_recall_at_20", "tail_evaluation_users"), ("tail_share_at_20", "recommendation_slots")]:
            assert sum(row[denominator] for row in grouped.values()) == overall[denominator]
            pooled = sum(row[metric] * row[denominator] for row in grouped.values()) / overall[denominator]
            assert math.isclose(pooled, overall[metric], abs_tol=1e-12)
        for group, metrics in grouped.items():
            grouped_rows.append(dict(identity, group=group, **metrics))
        diagnostics = embedding_diagnostics(degree, raw_norm, final_norm, last_norm, total.exposure)
        item_rows.append(dict(identity, **diagnostics))
        with gzip.open(args.output / (run + "-items.csv.gz"), "wt", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["item_id", "fitting_degree", "raw_norm", "final_norm", "last_norm", "validation_exposure"])
            writer.writerows(zip(range(n), degree, raw_norm, final_norm, last_norm, total.exposure))
        record = dict(identity, checkpoint_sha256=entry["checkpoint_sha256"], metric_deltas=deltas,
                      exposure_l1_difference=difference, measured=overall, reference=reference,
                      grouped=grouped, group_exposures_reconcile=True)
        (args.output / (run + "-verification.json")).write_text(json.dumps(record, indent=2) + "\n")
        audits.append(record)
        print("PASS " + run, flush=True)
        del model, checkpoint, final_users, final_items, propagated
        gc.collect()
        torch.cuda.empty_cache()
    write_csv(args.output / "user-groups-per-run.csv", grouped_rows)
    write_csv(args.output / "user-groups-summary.csv", summarize(grouped_rows, ["family", "layers", "group"], METRICS))
    write_csv(args.output / "item-correlations-per-run.csv", item_rows)
    write_csv(args.output / "item-correlations-summary.csv", summarize(item_rows, ["family", "layers"], diagnostics))
    (args.output / "audit.json").write_text(json.dumps(dict(all_passed=len(audits) == 24,
        evaluation_split="validation", test_labels_opened=False, post_hoc=True,
        plan_sha256=digest(plan_path), evaluation_count=len(audits),
        source_sha256={name: digest(repo / "scripts" / name) for name in
                       ["analyze_single_users_items.py", "single_lightgcn.py", "final_metrics.py", "embedding_metrics.py", "user_groups.py"]},
        evaluations=audits), indent=2) + "\n")


if __name__ == "__main__":
    main()
