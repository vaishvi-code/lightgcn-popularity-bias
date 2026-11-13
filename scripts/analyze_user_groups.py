"""Post-hoc grouped test evaluation of the frozen three-seed main comparison."""
import argparse
import csv
import gc
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
from final_metrics import METRICS, RankingMetrics, exclusion_sets, load_interactions
from user_groups import GROUPS, assign_group


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, base = args.repository, args.project_root
    plan_path = repo / "provenance/final-test-plan.json"
    group_plan = repo / "provenance/user-group-plan.json"
    plan = json.loads(plan_path.read_text())
    assert json.loads(group_plan.read_text())["original_test_already_viewed"] is True
    frozen = repo / "results/gowalla/final-test-job337725"
    verification = json.loads((frozen / "validation-verification.json").read_text())
    assert verification["all_passed"] and digest(plan_path) == verification["plan_sha256"]
    for name in ["final_metrics.py", "weighted_lightgcn.py"]:
        assert digest(repo / "scripts" / name) == verification["source_sha256"][name]
    entries = [entry for entry in plan["evaluations"] if entry["comparison_group"] == "main_three_seed"]
    assert len(entries) == 18 and all(e["kind"] == "checkpoint" for e in entries)
    upstream = base / "LightGCN-PyTorch"
    assert subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip() == plan["upstream_commit"]
    subprocess.run(["git", "-C", str(upstream), "diff", "--exit-code", "HEAD", "--", "code/model.py", "code/dataloader.py"], check=True)
    prepared = base / "prepared/gowalla"
    for filename, key in [("train.txt", "fitting_file_sha256"), ("val.txt", "validation_file_sha256")]:
        assert digest(prepared / filename) == plan[key]
    fitting, validation = (load_interactions(prepared / filename) for filename in ["train.txt", "val.txt"])
    popularity = repo / "results/gowalla/item-popularity.csv"
    assert digest(popularity) == plan["popularity_file_sha256"]
    with popularity.open() as stream:
        items = list(csv.DictReader(stream))
    n = plan["catalog_size"]
    assert [int(row["item_id"]) for row in items] == list(range(n))
    tail = {int(row["item_id"]) for row in items if row["group"] == "tail"}
    groups = {}
    assignments = []
    for user, history in sorted(fitting.items()):
        count = len(history & tail)
        group = assign_group(count, len(history))
        groups[user] = group
        assignments.append(dict(user_id=user, group=group, fitting_interactions=len(history), fitting_tail_interactions=count, fitting_tail_fraction=count / len(history)))
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "user-groups.csv", assignments)
    # Persist assignments before reading held-out labels or running models.
    assignment_hash = digest(args.output / "user-groups.csv")
    relevant_path = upstream / "data/gowalla/test.txt"
    expected_test_hash = json.loads(next((frozen / "test").glob("*/metrics.json")).read_text())["test_file_sha256"]
    assert digest(relevant_path) == expected_test_hash
    relevant = load_interactions(relevant_path)
    users = sorted(user for user, values in relevant.items() if values)
    assert set(users) <= groups.keys()
    seen = exclusion_sets(fitting, validation, "test")
    profiles = []
    for group in GROUPS:
        selected = [row for row in assignments if row["group"] == group and row["user_id"] in relevant]
        assert selected
        profiles.append(dict(group=group, evaluation_users=len(selected), mean_fitting_tail_fraction=statistics.mean(row["fitting_tail_fraction"] for row in selected), median_fitting_interactions=statistics.median(row["fitting_interactions"] for row in selected), tail_evaluation_users=sum(bool(relevant[row["user_id"]] & tail) for row in selected)))
    write_csv(args.output / "group-profiles.csv", profiles)
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(upstream / "code"))
    import world
    from dataloader import Loader
    from weighted_lightgcn import build_model
    assert torch.cuda.is_available()
    world.device = torch.device("cuda")
    world.tensorboard = 0
    torch.set_num_threads(4)
    loader = base / "prepared" / ("gowalla-validation-" + plan["fitting_file_sha256"][:12] + "-" + plan["validation_file_sha256"][:12])
    assert digest(loader / "train.txt") == plan["fitting_file_sha256"]
    assert digest(loader / "test.txt") == plan["validation_file_sha256"]
    dataset = Loader(config=entries[0]["config"], path=str(loader))
    degree = np.asarray(dataset.UserItemNet.sum(axis=0)).ravel()
    assert np.array_equal(degree, [int(row["fitting_degree"]) for row in items])
    assert tail == set(range(n)) - set(sorted(range(n), key=lambda i: (-degree[i], i))[:math.ceil(n * .2)])
    measured, audits = [], []
    for entry in entries:
        run = entry["run_id"]
        reference = json.loads((frozen / "test" / run / "metrics.json").read_text())
        checkpoint_path = next((root / run / "best.pt" for root in args.checkpoint_root if (root / run / "best.pt").is_file()), None)
        assert checkpoint_path is not None, run
        checkpoint_hash = digest(checkpoint_path)
        assert checkpoint_hash == reference["checkpoint_sha256"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == entry["checkpoint_epoch"]
        assert checkpoint["metadata"]["config"] == entry["config"]
        world.config.update(entry["config"])
        network = build_model(world.config, dataset, entry["appnp_beta"]).to(world.device)
        network.load_state_dict(checkpoint["model_state_dict"])
        network.eval()
        all_users = RankingMetrics(n, tail)
        accumulators = {group: RankingMetrics(n, tail) for group in GROUPS}
        with torch.no_grad():
            for offset in range(0, len(users), 256):
                batch = users[offset:offset + 256]
                scores = network.getUsersRating(torch.tensor(batch, device=world.device))
                assert torch.isfinite(scores).all()
                for row, user in enumerate(batch):
                    scores[row, list(seen.get(user, set()))] = -torch.inf
                recommendations = torch.topk(scores, k=20, dim=1).indices.cpu().numpy()
                for user, recs in zip(batch, recommendations):
                    all_users.add(recs, relevant[user], seen.get(user, set()))
                    accumulators[groups[user]].add(recs, relevant[user], seen.get(user, set()))
        overall = all_users.finish()
        for metric in METRICS:
            assert math.isclose(overall[metric], reference[metric], rel_tol=0, abs_tol=1e-6), (run, metric)
        with (frozen / "test" / run / "exposure.csv").open() as stream:
            expected_exposure = np.array([int(row["recommendation_count"]) for row in csv.DictReader(stream)])
        assert np.array_equal(expected_exposure, all_users.exposure), run
        assert np.array_equal(sum(acc.exposure for acc in accumulators.values()), all_users.exposure)
        grouped = {group: accumulator.finish() for group, accumulator in accumulators.items()}
        for metric, denominator in [("recall_at_20", "evaluation_users"), ("ndcg_at_20", "evaluation_users"), ("tail_share_at_20", "recommendation_slots"), ("tail_recall_at_20", "tail_evaluation_users")]:
            assert sum(row[denominator] for row in grouped.values()) == overall[denominator]
            pooled = sum(row[metric] * row[denominator] for row in grouped.values()) / overall[denominator]
            assert math.isclose(pooled, overall[metric], abs_tol=1e-12)
        for group, metrics in grouped.items():
            measured.append(dict(model=entry["model"], run_id=run, training_seed=entry["training_seed"], group=group, **metrics))
        audits.append(dict(run_id=run, checkpoint_sha256=checkpoint_hash, overall_metrics_reproduced=True, exact_overall_exposure=True, group_totals_reconcile=True))
        print("PASS " + run, flush=True)
        del network, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    write_csv(args.output / "per-run.csv", measured)
    buckets = defaultdict(list)
    for row in measured:
        buckets[(row["model"], row["group"])].append(row)
    summary = []
    for (model, group), rows in buckets.items():
        assert sorted(row["training_seed"] for row in rows) == [2020, 2021, 2022]
        for metric in METRICS:
            values = [row[metric] for row in rows]
            summary.append(dict(model=model, group=group, metric=metric, seed_count=3, mean=statistics.mean(values), sample_sd=statistics.stdev(values)))
    write_csv(args.output / "summary.csv", summary)
    (args.output / "audit.json").write_text(json.dumps(dict(study_phase="post_hoc_user_group_analysis", group_plan_sha256=digest(group_plan), final_plan_sha256=digest(plan_path), assignment_sha256=assignment_hash, test_file_sha256=expected_test_hash, source_sha256={name: digest(repo / "scripts" / name) for name in ["analyze_user_groups.py", "user_groups.py", "final_metrics.py", "weighted_lightgcn.py"]}, all_passed=len(audits) == 18, evaluations=audits), indent=2) + "\n")


if __name__ == "__main__":
    main()
