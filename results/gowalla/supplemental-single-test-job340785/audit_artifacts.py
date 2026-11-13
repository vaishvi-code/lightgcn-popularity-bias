"""Read-only audit of the separate supplemental test artifacts (standard library only)."""

import csv
import hashlib
import json
import math
import struct
from pathlib import Path

out = Path(__file__).resolve().parent
root = out.parents[2]
plan_path = root / "provenance/supplemental-single-test-plan.json"
plan = json.loads(plan_path.read_text())
plan_hash = hashlib.sha256(plan_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
gate = json.loads((out / "validation-verification.json").read_text())
completion = json.loads((out / "test-completion.json").read_text())
assert gate["all_passed"] and completion["all_passed"]
assert gate["plan_sha256"] == completion["plan_sha256"] == plan_hash
assert gate["source_sha256"] == completion["source_sha256"]
for name, expected in gate["source_sha256"].items():
    assert hashlib.sha256((root / "scripts" / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected
entries = {entry["run_id"]: entry for entry in plan["evaluations"]}
assert len(entries) == 4
for report in [gate, completion]:
    assert {record["run_id"] for record in report["evaluations"]} == set(entries)
    for record in report["evaluations"]:
        assert record["status"] == "passed" and record["checkpoint_sha256"] == entries[record["run_id"]]["checkpoint_sha256"]
assert gate["test_file_sha256"] is None and completion["test_file_sha256"] == plan["test_file_sha256"]
tail = {int(row["item_id"]) for row in csv.DictReader((root / "results/gowalla/item-popularity.csv").open()) if row["group"] == "tail"}
metric_names = ["recall_at_20", "ndcg_at_20", "tail_share_at_20", "coverage_at_20", "exposure_gini", "tail_recall_at_20"]
for phase in ["verify", "test"]:
    for run_id, entry in entries.items():
        metrics = json.loads((out / phase / run_id / "metrics.json").read_text())
        assert metrics["plan_sha256"] == plan_hash and metrics["source_sha256"] == gate["source_sha256"]
        assert metrics["checkpoint_sha256"] == entry["checkpoint_sha256"]
        assert metrics["evaluation_users"] == 29858 and metrics["tail_evaluation_users"] == (18987 if phase == "verify" else 26839)
        assert metrics["graph_interactions"] == "fitting only"
        assert metrics["excluded_interactions"] == ("fitting" if phase == "verify" else "fitting + validation")
        exposure = list(csv.DictReader((out / phase / run_id / "exposure.csv").open()))
        assert [int(row["item_id"]) for row in exposure] == list(range(40981))
        counts = [int(row["recommendation_count"]) for row in exposure]
        slots, n = sum(counts), len(counts)
        assert slots == 597160 and all(value >= 0 for value in counts)
        cumulative = 0
        area = 0
        for value in sorted(counts):
            area += 2 * cumulative + value
            cumulative += value
        recomputed = {
            "tail_share_at_20": sum(counts[item] for item in tail) / slots,
            "coverage_at_20": sum(value > 0 for value in counts) / n,
            "exposure_gini": 1 - area / (n * slots),
        }
        for key, value in recomputed.items():
            assert math.isclose(value, metrics[key], abs_tol=1e-12, rel_tol=0)
        if phase == "verify":
            assert hashlib.sha256(struct.pack("<" + "q" * n, *counts)).hexdigest() == entry["validation_exposure_sha256"]
            for key in metric_names:
                assert math.isclose(metrics[key], entry["validation_metrics"][key], abs_tol=1e-6, rel_tol=0)
for row in csv.DictReader((out / "comparison.csv").open()):
    folder = out / "test" / row["run_id"] if row["layer_combination"] == "single" else root / "results/gowalla/final-test-job337725/test" / row["run_id"]
    source = json.loads((folder / "metrics.json").read_text())
    for key in metric_names:
        assert math.isclose(float(row[key]), source[key], abs_tol=1e-12, rel_tol=0)
    assert source["test_file_sha256"] == plan["test_file_sha256"]
print("PASS: frozen supplemental identities, four validation reproductions, four test results, eight exposure audits, and matched test comparisons.")
