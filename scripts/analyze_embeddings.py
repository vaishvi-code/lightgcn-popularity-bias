"""Supplemental Q4 diagnostics from frozen checkpoints and saved validation exposures."""

import argparse
import csv
import gc
import gzip
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
import torch

from embedding_metrics import embedding_diagnostics


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    base, repository = args.project_root.resolve(), args.repository_root.resolve()
    plan_path = repository / "provenance/final-test-plan.json"
    plan = json.loads(plan_path.read_text())
    verification_root = repository / "results/gowalla/final-test-job337725"
    verification = json.loads((verification_root / "validation-verification.json").read_text())
    assert verification["all_passed"] and digest(plan_path) == verification["plan_sha256"]
    expected_hash = {row["run_id"]: row["checkpoint_sha256"] for row in verification["evaluations"]}
    entries = [
        entry for entry in plan["evaluations"]
        if entry["comparison_group"] == "main_three_seed"
        and entry["kind"] == "checkpoint" and entry["appnp_beta"] is None
    ]
    assert len(entries) == 15
    paths = {}
    for entry in entries:
        candidates = [root / entry["run_id"] / "best.pt" for root in args.checkpoint_root]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError("Missing frozen checkpoint " + entry["run_id"])
        paths[entry["run_id"]] = path
        assert digest(path) == expected_hash[entry["run_id"]]

    prepared = base / "prepared/gowalla"
    assert digest(prepared / "train.txt") == plan["fitting_file_sha256"]
    assert digest(prepared / "val.txt") == plan["validation_file_sha256"]
    popularity_path = repository / "results/gowalla/item-popularity.csv"
    assert digest(popularity_path) == plan["popularity_file_sha256"]
    with popularity_path.open() as stream:
        items = list(csv.DictReader(stream))
    assert [int(item["item_id"]) for item in items] == list(range(plan["catalog_size"]))
    degree = np.array([int(item["fitting_degree"]) for item in items])
    head = np.array([item["group"] == "popular" for item in items])
    args.output.mkdir(parents=True, exist_ok=False)
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(base / "LightGCN-PyTorch/code"))
    import world
    from dataloader import Loader
    from model import LightGCN

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested without a GPU allocation")
    world.device = torch.device(args.device)
    world.tensorboard = 0
    torch.set_num_threads(4)
    loader_path = base / "prepared" / (
        "gowalla-validation-" + plan["fitting_file_sha256"][:12]
        + "-" + plan["validation_file_sha256"][:12]
    )
    assert digest(loader_path / "train.txt") == plan["fitting_file_sha256"]
    assert digest(loader_path / "test.txt") == plan["validation_file_sha256"]
    dataset = Loader(config=entries[0]["config"], path=str(loader_path))
    summaries, profiles = [], []
    boundaries = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, float("inf")]
    for entry in entries:
        run_id = entry["run_id"]
        checkpoint = torch.load(paths[run_id], map_location="cpu", weights_only=False)
        metadata = checkpoint["metadata"]
        assert metadata["config"] == entry["config"]
        assert metadata["training_seed"] == entry["training_seed"]
        assert checkpoint["epoch"] == entry["checkpoint_epoch"]
        assert metadata["fitting_file_sha256"] == plan["fitting_file_sha256"]
        assert metadata["validation_file_sha256"] == plan["validation_file_sha256"]
        with (verification_root / "verify" / run_id / "exposure.csv").open() as stream:
            exposure_rows = list(csv.DictReader(stream))
        assert [int(row["item_id"]) for row in exposure_rows] == list(range(len(degree)))
        exposure = np.array([int(row["recommendation_count"]) for row in exposure_rows], dtype=np.int64)
        assert hashlib.sha256(exposure.astype("<i8").tobytes()).hexdigest() == entry["validation_exposure_sha256"]
        world.config.update(entry["config"])
        network = LightGCN(world.config, dataset).to(world.device)
        network.load_state_dict(checkpoint["model_state_dict"])
        network.eval()
        with torch.no_grad():
            combined_items = network.computer()[1]
            embeddings = torch.cat([network.embedding_user.weight, network.embedding_item.weight])
            raw_norm = torch.linalg.vector_norm(network.embedding_item.weight, dim=1).cpu().numpy()
            for _ in range(network.n_layers):
                embeddings = torch.sparse.mm(network.Graph, embeddings)
            last_norm = torch.linalg.vector_norm(embeddings[dataset.n_users:], dim=1).cpu().numpy()
            combined_norm = torch.linalg.vector_norm(combined_items, dim=1).cpu().numpy()
        identity = {"run_id": run_id, "model": entry["model"], "layers": network.n_layers, "training_seed": entry["training_seed"]}
        summary = {
            **identity, "checkpoint_epoch": checkpoint["epoch"],
            "checkpoint_sha256": expected_hash[run_id], "item_count": len(degree),
            "exposure_source": "frozen validation exposure",
            **embedding_diagnostics(degree, raw_norm, combined_norm, last_norm, exposure),
            "mean_head_combined_norm": float(combined_norm[head].mean()),
            "mean_tail_combined_norm": float(combined_norm[~head].mean()),
        }
        summaries.append(summary)
        for lower, upper in zip(boundaries, boundaries[1:]):
            mask = (degree >= lower) & (degree < upper)
            if mask.any():
                profiles.append({
                    **identity, "degree_lower": lower, "degree_upper": upper,
                    "item_count": int(mask.sum()), "median_degree": float(np.median(degree[mask])),
                    "median_raw_norm": float(np.median(raw_norm[mask])),
                    "median_combined_norm": float(np.median(combined_norm[mask])),
                    "median_last_norm": float(np.median(last_norm[mask])),
                })
        with gzip.open(args.output / (run_id + "-item-diagnostics.csv.gz"), "wt", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["item_id", "fitting_degree", "raw_norm", "combined_norm", "last_layer_norm", "validation_exposure"])
            writer.writerows(zip(range(len(degree)), degree, raw_norm, combined_norm, last_norm, exposure))
        print(json.dumps(summary), flush=True)
        del network, checkpoint, embeddings, combined_items
        gc.collect()

    write_csv(args.output / "per-run.csv", summaries)
    write_csv(args.output / "degree-profiles.csv", profiles)
    grouped = defaultdict(list)
    for row in summaries:
        grouped[row["layers"]].append(row)
    aggregate = []
    for layers, rows in sorted(grouped.items()):
        assert sorted(row["training_seed"] for row in rows) == [2020, 2021, 2022]
        for metric in embedding_diagnostics(degree, raw_norm, combined_norm, last_norm, exposure):
            values = [row[metric] for row in rows]
            aggregate.append({
                "layers": layers, "metric": metric, "run_count": len(rows),
                "mean": statistics.mean(values) if all(value is not None for value in values) else None,
                "sample_std": statistics.stdev(values) if all(value is not None for value in values) else None,
            })
    write_csv(args.output / "summary.csv", aggregate)
    (args.output / "analysis-metadata.json").write_text(json.dumps({
        "analysis": "supplemental_post_final_test_embedding_diagnostics",
        "plan_sha256": digest(plan_path), "checkpoint_count": len(entries),
        "exposure_split": "validation", "test_labels_opened": False,
        "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
        "sources": {name: digest(Path(__file__).parent / name) for name in ["analyze_embeddings.py", "embedding_metrics.py"]},
        "interpretation": "Item-level correlations describe association and do not prove the scoring mechanism. Degree and its square root have the same ranks.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
