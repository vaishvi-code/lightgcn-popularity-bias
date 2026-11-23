"""Audit exported validation user/item diagnostics without checkpoints or a GPU."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def audit(root):
    record = json.loads((root / "audit.json").read_text())
    assert record["all_passed"] and record["evaluation_count"] == 24
    assert record["evaluation_split"] == "validation" and record["test_labels_opened"] is False
    source_bundle = root / "inference-sources.tar.gz"
    if source_bundle.exists():
        with tarfile.open(source_bundle) as archive:
            hashes = {"scripts/" + name: value for name, value in record["source_sha256"].items()}
            hashes["provenance/single-users-items-plan.json"] = record["plan_sha256"]
            for name, expected_hash in hashes.items():
                assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected_hash
    profiles = {r["group"]: r for r in read_csv(root / "group-profiles.csv")}
    assert len(profiles) == 3
    expected = {(family, depth, seed) for family in ["single", "uniform"]
                for depth in range(1, 5) for seed in [2020, 2021, 2022]}
    assert {(e["family"], e["layers"], e["training_seed"]) for e in record["evaluations"]} == expected
    for entry in record["evaluations"]:
        assert entry["group_exposures_reconcile"] and entry["exposure_l1_difference"] <= 20
        assert all(abs(v) <= 1e-5 for v in entry["metric_deltas"].values())
        overall = entry["measured"]
        assert sum(int(r["evaluation_users"]) for r in profiles.values()) == overall["evaluation_users"]
        for group, metrics in entry["grouped"].items():
            assert metrics["evaluation_users"] == int(profiles[group]["evaluation_users"])
            assert metrics["tail_evaluation_users"] == int(profiles[group]["tail_evaluation_users"])
        for metric, denominator in [("recall_at_20", "evaluation_users"), ("ndcg_at_20", "evaluation_users"),
                                    ("tail_recall_at_20", "tail_evaluation_users"), ("tail_share_at_20", "recommendation_slots")]:
            pooled = sum(r[metric] * r[denominator] for r in entry["grouped"].values()) / overall[denominator]
            assert math.isclose(pooled, overall[metric], abs_tol=1e-12)
        with gzip.open(root / (entry["run_id"] + "-items.csv.gz"), "rt") as stream:
            rows = list(csv.DictReader(stream))
        assert [int(r["item_id"]) for r in rows] == list(range(overall["catalog_size"]))
        assert sum(int(r["validation_exposure"]) for r in rows) == overall["recommendation_slots"]
        for row in rows:
            assert int(row["fitting_degree"]) >= 0 and int(row["validation_exposure"]) >= 0
            assert all(math.isfinite(float(row[k])) and float(row[k]) >= 0 for k in ["raw_norm", "final_norm", "last_norm"])
            if entry["family"] == "single":
                assert row["final_norm"] == row["last_norm"]
    for prefix, keys, count in [("user-groups", ["family", "layers", "group"], 72),
                                 ("item-correlations", ["family", "layers"], 24)]:
        rows = read_csv(root / (prefix + "-per-run.csv"))
        assert len(rows) == count
        buckets = defaultdict(list)
        for row in rows:
            buckets[tuple(row[k] for k in keys)].append(row)
        for row in read_csv(root / (prefix + "-summary.csv")):
            members = buckets[tuple(row[k] for k in keys)]
            assert sorted(int(r["training_seed"]) for r in members) == [2020, 2021, 2022]
            assert int(row["seed_count"]) == 3
            values = [r[row["metric"]] for r in members]
            if all(values):
                values = list(map(float, values))
                assert math.isclose(float(row["mean"]), statistics.mean(values), abs_tol=1e-12)
                assert math.isclose(float(row["sample_sd"]), statistics.stdev(values), abs_tol=1e-12)
            else:
                assert not row["mean"] and not row["sample_sd"]
    print("PASS: 24 frozen comparisons, pooled user metrics, 24 item exports and seed summaries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    audit(parser.parse_args().results)
