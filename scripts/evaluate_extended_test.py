"""Gated post-hoc evaluation of selected single, APPNP and normalization checkpoints."""
import argparse
import csv
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from final_metrics import METRICS, RankingMetrics, exclusion_sets, load_interactions
from supplemental_test_gate import assert_reconciles, plan_digest, require_gate, source_digest
from user_groups import GROUPS, assign_group


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_exposure(path, n):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r["item_id"]) for r in rows] == list(range(n))
    return np.array([int(r["recommendation_count"]) for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["verify", "test"], required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, base = args.repository, args.project_root
    plan = json.loads((repo / "provenance/extended-test-plan.json").read_text())
    assert plan["original_test_already_viewed"] and plan["new_test_evaluation_authorized"]
    assert plan["study_phase"] == "post_hoc_extended_test" and len(plan["evaluations"]) == 24
    sources = {name: source_digest(repo / "scripts" / name) for name in plan["source_sha256"]}
    assert sources == plan["source_sha256"], "Source changed after plan freeze"
    identity = plan_digest(plan)
    paths, hashes = {}, {}
    for entry in plan["evaluations"]:
        run = entry["run_id"]
        path = next((root / run / "best.pt" for root in args.checkpoint_root if (root / run / "best.pt").is_file()), None)
        if path is None:
            raise FileNotFoundError(run)
        paths[run], hashes[run] = path, digest(path)
        assert hashes[run] == entry["checkpoint_sha256"], run
    gate_path = args.output / "validation-verification.json"
    if args.phase == "test":
        require_gate(json.loads(gate_path.read_text()), identity, sources, hashes)
    upstream = base / "LightGCN-PyTorch"
    assert subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip() == plan["upstream_commit"]
    subprocess.run(["git", "-C", str(upstream), "diff", "--exit-code", "HEAD", "--", "code/model.py", "code/dataloader.py"], check=True)
    data = base / "prepared/gowalla"
    for filename, key in [("train.txt", "fitting_file_sha256"), ("val.txt", "validation_file_sha256")]:
        assert digest(data / filename) == plan[key]
    fit, val = (load_interactions(data / f) for f in ["train.txt", "val.txt"])
    assert all(v.isdisjoint(fit.get(u, set())) for u, v in val.items())
    known = exclusion_sets(fit, val, "test")
    assert digest(upstream / "data/gowalla/train.txt") == plan["original_train_sha256"]
    assert {u: v for u, v in known.items() if v} == {u: v for u, v in load_interactions(upstream / "data/gowalla/train.txt").items() if v}
    popularity = repo / "results/gowalla/item-popularity.csv"
    assert digest(popularity) == plan["popularity_file_sha256"]
    with popularity.open() as stream:
        items = list(csv.DictReader(stream))
    n = plan["catalog_size"]
    assert [int(r["item_id"]) for r in items] == list(range(n))
    degree = np.array([int(r["fitting_degree"]) for r in items])
    tail = {int(r["item_id"]) for r in items if r["group"] == "tail"}
    assert tail == set(range(n)) - set(sorted(range(n), key=lambda i: (-degree[i], i))[:math.ceil(n * .2)])
    groups = {u: assign_group(len(history & tail), len(history)) for u, history in fit.items()}
    # Only this block opens the original test labels, after the complete gate.
    test_hash = None
    if args.phase == "test":
        test_path = upstream / "data/gowalla/test.txt"
        test_hash = digest(test_path)
        assert test_hash == plan["test_file_sha256"]
        relevant = load_interactions(test_path)
    else:
        relevant = val
    users = sorted(u for u, values in relevant.items() if values)
    assert len(users) == 29858 and set(users) <= groups.keys()
    seen = exclusion_sets(fit, val, "test" if args.phase == "test" else "validation")
    assert all(relevant[u].isdisjoint(seen[u]) for u in users)
    args.output.mkdir(parents=True, exist_ok=True)
    profiles = [dict(group=g, evaluation_users=sum(groups[u] == g for u in users),
                     tail_evaluation_users=sum(groups[u] == g and bool(relevant[u] & tail) for u in users)) for g in GROUPS]
    write_csv(args.output / (args.phase + "-group-profiles.csv"), profiles)
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(upstream / "code"))
    import world
    from normalized_graph import build_dataset
    from single_lightgcn import SingleLayerOutputLightGCN
    from weighted_lightgcn import build_model
    assert torch.cuda.is_available()
    world.device, world.tensorboard = torch.device("cuda"), 0
    torch.set_num_threads(4)
    loader = base / "prepared" / ("gowalla-validation-" + plan["fitting_file_sha256"][:12] + "-" + plan["validation_file_sha256"][:12])
    assert digest(loader / "train.txt") == plan["fitting_file_sha256"]
    assert digest(loader / "test.txt") == plan["validation_file_sha256"]
    records, results, grouped_rows, datasets = [], [], [], {}
    for entry in plan["evaluations"]:
        started = time.perf_counter()
        run = entry["run_id"]
        checkpoint = torch.load(paths[run], map_location="cpu", weights_only=False)
        meta = checkpoint["metadata"]
        assert meta["config"] == entry["config"] and meta["training_seed"] == entry["training_seed"]
        assert checkpoint["epoch"] == entry["checkpoint_epoch"]
        assert meta.get("appnp_beta") == entry["beta"]
        norm = meta.get("normalization", {"a": .5, "b": .5})
        assert norm["a"] == entry["a"] and norm["b"] == entry["b"]
        combination = {"single": "single", "appnp": "appnp", "normalization": "uniform"}[entry["family"]]
        assert meta.get("layer_combination", combination) == combination
        for key in ["fitting_file_sha256", "validation_file_sha256"]:
            assert meta[key] == plan[key]
        world.config.update(entry["config"])
        ds_key = (entry["a"], entry["b"])
        if ds_key not in datasets:
            datasets[ds_key] = build_dataset(world.config, loader, *ds_key)
        dataset = datasets[ds_key]
        assert np.array_equal(np.asarray(dataset.UserItemNet.sum(axis=0)).ravel(), degree)
        model = (SingleLayerOutputLightGCN(world.config, dataset) if entry["family"] == "single"
                 else build_model(world.config, dataset, entry["beta"])).to(world.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        total = RankingMetrics(n, tail)
        accumulators = {g: RankingMetrics(n, tail) for g in GROUPS}
        with torch.no_grad():
            embeddings_u, embeddings_i = model.computer()
            for offset in range(0, len(users), 256):
                batch = users[offset:offset + 256]
                scores = torch.sigmoid(embeddings_u[torch.tensor(batch, device=world.device)] @ embeddings_i.T)
                assert torch.isfinite(scores).all()
                for row, user in enumerate(batch):
                    scores[row, list(seen[user])] = -torch.inf
                recommendations = torch.topk(scores, k=20, dim=1).indices.cpu().numpy()
                for user, recs in zip(batch, recommendations):
                    total.add(recs, relevant[user], seen[user])
                    accumulators[groups[user]].add(recs, relevant[user], seen[user])
        metrics = total.finish()
        grouped = {g: acc.finish() for g, acc in accumulators.items()}
        assert np.array_equal(sum(acc.exposure for acc in accumulators.values()), total.exposure)
        assert_reconciles(metrics, grouped)
        reference, exposure_ref = None, None
        if args.phase == "verify":
            reference, exposure_ref = entry["validation_metrics"], entry["validation_exposure"]
        elif entry["previous_test_metrics"]:
            reference = json.loads((repo / entry["previous_test_metrics"]).read_text())
            exposure_ref = entry["previous_test_exposure"]
        deltas, difference = {}, None
        if reference is not None:
            deltas = {key: metrics[key] - reference[key] for key in METRICS}
            assert all(abs(v) <= plan["metric_absolute_tolerance"] for v in deltas.values()), (run, deltas)
            assert metrics["catalog_size"] == n
            for key in ["evaluation_users", "tail_evaluation_users"]:
                assert metrics[key] == reference[key]
            difference = int(np.abs(read_exposure(repo / exposure_ref, n) - total.exposure).sum())
            assert difference <= plan["exposure_l1_tolerance"], (run, difference)
        result = dict(model=entry["model"], run_id=run, training_seed=entry["training_seed"],
            family=entry["family"], layers=entry["layers"], beta=entry["beta"], a=entry["a"], b=entry["b"],
            evaluation_split="validation" if args.phase == "verify" else "test", checkpoint_epoch=entry["checkpoint_epoch"],
            checkpoint_sha256=hashes[run], study_phase=plan["study_phase"], original_test_already_viewed=True,
            first_test_evaluation=entry["previous_test_metrics"] is None if args.phase == "test" else False,
            plan_sha256=identity, source_sha256=sources, top_k=20,
            fitting_file_sha256=plan["fitting_file_sha256"], validation_file_sha256=plan["validation_file_sha256"],
            test_file_sha256=test_hash, popularity_file_sha256=plan["popularity_file_sha256"],
            graph_interactions="fitting only", excluded_interactions="fitting plus validation" if args.phase == "test" else "fitting",
            reproduction_metric_deltas=deltas, reproduction_exposure_l1=difference,
            grouped=grouped, elapsed_seconds=time.perf_counter() - started, **metrics)
        destination = args.output / args.phase / run
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        with (destination / "exposure.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["item_id", "recommendation_count"])
            writer.writerows(enumerate(total.exposure.tolist()))
        results.append(result)
        grouped_rows.extend(dict(model=entry["model"], run_id=run, training_seed=entry["training_seed"], group=g, **m) for g, m in grouped.items())
        records.append(dict(run_id=run, checkpoint_sha256=hashes[run], status="passed"))
        print("PASS " + args.phase + " " + run, flush=True)
        del model, checkpoint, embeddings_u, embeddings_i
        gc.collect()
        torch.cuda.empty_cache()
    write_csv(args.output / (args.phase + "-per-run.csv"), [{k: r[k] for k in ["model", "run_id", "training_seed", "evaluation_split", *METRICS]} for r in results])
    write_csv(args.output / (args.phase + "-user-groups.csv"), grouped_rows)
    report = dict(phase=args.phase, all_passed=len(records) == 24, plan_sha256=identity,
        source_sha256=sources, evaluations=records, test_file_sha256=test_hash,
        test_labels_opened=args.phase == "test", original_test_already_viewed=True,
        torch_version=torch.__version__, numpy_version=np.__version__, gpu=torch.cuda.get_device_name(0))
    (gate_path if args.phase == "verify" else args.output / "test-completion.json").write_text(json.dumps(report, indent=2) + "\n")
    print("COMPLETE " + args.phase + " 24 checkpoints", flush=True)


if __name__ == "__main__":
    main()
