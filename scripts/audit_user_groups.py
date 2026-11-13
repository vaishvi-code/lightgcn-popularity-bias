"""Independently audit saved assignments, denominators and seed summaries."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from user_groups import assign_group

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("results", type=Path)
args = parser.parse_args()


def read(name):
    with (args.results / name).open() as stream:
        return list(csv.DictReader(stream))


audit = json.loads((args.results / "audit.json").read_text())
assert audit["all_passed"] and len(audit["evaluations"]) == 18
profiles = {row["group"]: row for row in read("group-profiles.csv")}
counts = {group: int(profile["evaluation_users"]) for group, profile in profiles.items()}
assert sum(counts.values()) == 29858
assignment_path = args.results / "user-groups.csv"
if assignment_path.exists():
    assert hashlib.sha256(assignment_path.read_bytes()).hexdigest() == audit["assignment_sha256"]
    assignments = read("user-groups.csv")
    assert len(assignments) == len({row["user_id"] for row in assignments}) == 29858
    assert Counter(row["group"] for row in assignments) == counts
    for row in assignments:
        assert row["group"] == assign_group(int(row["fitting_tail_interactions"]), int(row["fitting_interactions"]))
    print("PASS: local-only 29,858 assignments and group boundaries")
else:
    print("Per-user assignments not published; checking aggregate artifacts only")
rows = read("per-run.csv")
assert len(rows) == 54
grouped = defaultdict(list)
for row in rows:
    assert int(row["evaluation_users"]) == counts[row["group"]]
    assert int(row["tail_evaluation_users"]) == int(profiles[row["group"]]["tail_evaluation_users"])
    assert int(row["recommendation_slots"]) == counts[row["group"]] * 20
    grouped[(row["model"], row["group"])].append(row)
summary = read("summary.csv")
assert len(summary) == 108 and len(grouped) == 18
for row in summary:
    matching = grouped[(row["model"], row["group"])]
    assert sorted(int(item["training_seed"]) for item in matching) == [2020, 2021, 2022]
    values = [float(item[row["metric"]]) for item in matching]
    assert math.isclose(float(row["mean"]), statistics.mean(values), abs_tol=1e-12)
    assert math.isclose(float(row["sample_sd"]), statistics.stdev(values), abs_tol=1e-12)
for record in audit["evaluations"]:
    assert record["group_totals_reconcile"] and record["overall_metrics_within_tolerance"]
    assert record["exposure_l1_difference"] <= 20
    assert all(abs(value) <= 1e-5 for value in record["metric_deltas"].values())
print("PASS: group denominators, 54 grouped runs, 108 seed summaries and 18 reproduction records")
