"""Separate, post-hoc test evaluation of frozen LightGCN-single checkpoints."""

import argparse
import csv
import gc
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from final_metrics import METRICS, RankingMetrics, exclusion_sets, load_interactions


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def require_gate(gate, plan_hash, sources, checkpoints):
    if not gate.get("all_passed") or gate.get("plan_sha256") != plan_hash:
        raise ValueError("Complete matching validation verification is required")
    if gate.get("source_sha256") != sources:
        raise ValueError("Evaluator sources changed after validation verification")
    records = gate.get("evaluations", [])
    if len(records) != len(checkpoints) or len({row["run_id"] for row in records}) != len(records):
        raise ValueError("Verification roster differs from the frozen roster")
    observed = {row["run_id"]: row["checkpoint_sha256"] for row in records if row["status"] == "passed"}
    if observed != checkpoints:
        raise ValueError("Checkpoint identities changed or validation verification was incomplete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["verify", "test"], required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    plan_hash = digest(args.plan)
    assert plan["study_phase"] == "supplemental_single_post_original_test"
    assert plan["original_final_test_already_viewed"] is True
    assert len(plan["evaluations"]) == 4
    sources = {name: digest(Path(__file__).parent / name) for name in [
        "evaluate_supplemental_single.py", "final_metrics.py", "single_lightgcn.py"
    ]}
    base = args.project_root.resolve()
    upstream = base / "LightGCN-PyTorch"
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    assert commit == plan["upstream_commit"]
    subprocess.run(["git", "-C", str(upstream), "diff", "--exit-code", "HEAD", "--", "code/model.py", "code/dataloader.py"], check=True)
    prepared = base / "prepared/gowalla"
    for name, key in [("train.txt", "fitting_file_sha256"), ("val.txt", "validation_file_sha256")]:
        assert digest(prepared / name) == plan[key]
    popularity = base / "results/gowalla/item-popularity.csv"
    assert digest(popularity) == plan["popularity_file_sha256"]
    with popularity.open() as stream:
        items = list(csv.DictReader(stream))
    n = plan["catalog_size"]
    assert [int(row["item_id"]) for row in items] == list(range(n))
    tail = {int(row["item_id"]) for row in items if row["group"] == "tail"}
    fitting = load_interactions(prepared / "train.txt")
    validation = load_interactions(prepared / "val.txt")
    assert all(values.isdisjoint(fitting.get(user, set())) for user, values in validation.items())
    original_train = upstream / "data/gowalla/train.txt"
    assert digest(original_train) == plan["original_train_sha256"]
    known = exclusion_sets(fitting, validation, "test")
    assert {user: positives for user, positives in known.items() if positives} == load_interactions(original_train)
    checkpoints = {}
    for entry in plan["evaluations"]:
        path = base / "results/gowalla" / entry["run_id"] / "best.pt"
        checkpoints[entry["run_id"]] = digest(path)
        assert checkpoints[entry["run_id"]] == entry["checkpoint_sha256"]
    args.output.mkdir(parents=True, exist_ok=True)
    gate_path = args.output / "validation-verification.json"
    test_hash = None
    if args.phase == "test":
        require_gate(json.loads(gate_path.read_text()), plan_hash, sources, checkpoints)
        # The new evaluator opens test labels only after all four validation checks pass.
        test_path = upstream / "data/gowalla/test.txt"
        test_hash = digest(test_path)
        assert test_hash == plan["test_file_sha256"]
        relevant = load_interactions(test_path)
    else:
        relevant = validation
    seen = exclusion_sets(fitting, validation, "test" if args.phase == "test" else "validation")
    users = sorted(user for user, positives in relevant.items() if positives)
    assert len(users) == 29858
    for user in users:
        assert relevant[user].isdisjoint(seen.get(user, set()))
        assert all(0 <= item < n for item in relevant[user])
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(upstream / "code"))
    import world
    from dataloader import Loader
    from single_lightgcn import SingleLayerOutputLightGCN

    assert torch.cuda.is_available(), "Use a GPU allocation"
    torch.set_num_threads(4)
    world.tensorboard = 0
    loader = base / "prepared" / (
        "gowalla-validation-" + plan["fitting_file_sha256"][:12] + "-" + plan["validation_file_sha256"][:12]
    )
    assert digest(loader / "train.txt") == plan["fitting_file_sha256"]
    assert digest(loader / "test.txt") == plan["validation_file_sha256"]
    dataset = Loader(config=plan["evaluations"][0]["config"], path=str(loader))
    assert dataset.m_items == n and all(0 <= user < dataset.n_users for user in users)
    measured_degrees = np.asarray(dataset.UserItemNet.sum(axis=0)).ravel()
    assert np.array_equal(measured_degrees, [int(row["fitting_degree"]) for row in items])
    head = set(sorted(range(n), key=lambda item: (-measured_degrees[item], item))[:math.ceil(0.2 * n)])
    assert tail == set(range(n)) - head
    records, measured = [], []
    for entry in plan["evaluations"]:
        run_id = entry["run_id"]
        checkpoint = torch.load(base / "results/gowalla" / run_id / "best.pt", map_location="cpu", weights_only=False)
        metadata = checkpoint["metadata"]
        assert metadata["config"] == entry["config"]
        assert metadata["training_seed"] == 2020 and checkpoint["epoch"] == entry["checkpoint_epoch"]
        assert metadata["layer_combination"] == "single" and metadata.get("appnp_beta") is None
        assert metadata["normalization"]["a"] == metadata["normalization"]["b"] == 0.5
        assert metadata["layer_weights"] == [0.0] * entry["layers"] + [1.0]
        for key in ["fitting_file_sha256", "validation_file_sha256"]:
            assert metadata[key] == plan[key]
        world.config.update(entry["config"])
        network = SingleLayerOutputLightGCN(world.config, dataset).to(world.device)
        network.load_state_dict(checkpoint["model_state_dict"])
        network.eval()
        accumulator = RankingMetrics(n, tail)
        with torch.no_grad():
            for offset in range(0, len(users), 256):
                batch = users[offset:offset + 256]
                scores = network.getUsersRating(torch.tensor(batch, device=world.device))
                assert torch.isfinite(scores).all()
                for row, user in enumerate(batch):
                    scores[row, list(seen.get(user, set()))] = -torch.inf
                recommendations = torch.topk(scores, k=20, dim=1).indices.cpu().numpy()
                for user, items in zip(batch, recommendations):
                    accumulator.add(items, relevant[user], seen.get(user, set()))
        metrics = accumulator.finish()
        if args.phase == "verify":
            for key in METRICS:
                assert math.isclose(metrics[key], entry["validation_metrics"][key], rel_tol=0, abs_tol=1e-6), (run_id, key)
            for key in ["evaluation_users", "tail_evaluation_users"]:
                assert metrics[key] == entry["validation_metrics"][key]
            assert hashlib.sha256(np.asarray(accumulator.exposure, dtype="<i8").tobytes()).hexdigest() == entry["validation_exposure_sha256"]
        result = {
            "model": entry["model"], "run_id": run_id, "comparison_group": "supplemental_single_seed",
            "study_phase": plan["study_phase"], "training_seed": 2020,
            "checkpoint_epoch": entry["checkpoint_epoch"], "checkpoint_sha256": checkpoints[run_id],
            "evaluation_split": "test" if args.phase == "test" else "validation", "top_k": 20,
            "plan_sha256": plan_hash, "source_sha256": sources,
            "fitting_file_sha256": plan["fitting_file_sha256"], "validation_file_sha256": plan["validation_file_sha256"],
            "popularity_file_sha256": plan["popularity_file_sha256"], "test_file_sha256": test_hash,
            "graph_interactions": "fitting only", "excluded_interactions": "fitting + validation" if args.phase == "test" else "fitting",
            **metrics,
        }
        destination = args.output / args.phase / run_id
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        with (destination / "exposure.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["item_id", "recommendation_count"])
            writer.writerows(enumerate(accumulator.exposure.tolist()))
        measured.append(result)
        records.append({"run_id": run_id, "status": "passed", "checkpoint_sha256": checkpoints[run_id]})
        print(f"PASS {args.phase} {run_id} Recall@20={metrics['recall_at_20']:.8f}", flush=True)
        del network, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    report = {
        "phase": args.phase, "study_phase": plan["study_phase"], "plan_sha256": plan_hash,
        "source_sha256": sources, "all_passed": len(records) == 4,
        "evaluations": records, "test_file_sha256": test_hash,
    }
    report_path = gate_path if args.phase == "verify" else args.output / "test-completion.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.phase == "test":
        with (args.output / "test-per-run.csv").open("w", newline="") as stream:
            fields = ["model", "run_id", "training_seed", "checkpoint_epoch", "evaluation_split", *METRICS]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(measured)


if __name__ == "__main__":
    main()
