"""Audit the supplemental export and assemble the preserved 42-run test comparison."""
import argparse
import csv
import json
import math
import statistics
import io
import tarfile
from collections import defaultdict
from pathlib import Path
from supplemental_test_gate import assert_reconciles, plan_digest, require_gate, source_digest

METRICS = ["recall_at_20", "ndcg_at_20", "tail_share_at_20", "coverage_at_20", "exposure_gini", "tail_recall_at_20"]
ORDER = ["BPR-MF"] + [f"LightGCN K={k}" for k in range(1, 5)] + [f"LightGCN-single K={k}" for k in range(1, 5)] + [f"APPNP beta={b}" for b in [0.1, 0.2, 0.5]] + ["Normalization a=0.25 b=0.75", "Normalization a=0.25 b=1"]


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exposure_check(path, metrics, tail, n):
    rows = read_csv(path)
    assert [int(r["item_id"]) for r in rows] == list(range(n))
    counts = [int(r["recommendation_count"]) for r in rows]
    assert all(c >= 0 for c in counts)
    slots = sum(counts)
    assert slots == metrics["recommendation_slots"] == metrics["evaluation_users"] * 20
    assert metrics["catalog_size"] == n
    assert metrics["distinct_recommended_items"] == sum(c > 0 for c in counts)
    derived = dict(tail_share_at_20=sum(counts[i] for i in tail) / slots,
                   coverage_at_20=sum(c > 0 for c in counts) / n,
                   exposure_gini=2 * sum(i * c for i, c in enumerate(sorted(counts), 1)) / (n * slots) - (n + 1) / n)
    for key, value in derived.items():
        assert math.isclose(value, metrics[key], abs_tol=1e-12), (path, key)
    return counts


def aggregate(rows, keys):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in keys)].append(row)
    summary = []
    for identity, members in buckets.items():
        assert sorted(int(r["training_seed"]) for r in members) == [2020, 2021, 2022]
        for metric in METRICS:
            values = [float(r[metric]) for r in members]
            summary.append(dict(zip(keys, identity), metric=metric, seed_count=3,
                                mean=statistics.mean(values), sample_sd=statistics.stdev(values)))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root, repo = args.results, args.repository
    plan = json.loads((repo / "provenance/extended-test-plan.json").read_text())
    gate = json.loads((root / "validation-verification.json").read_text())
    completion = json.loads((root / "test-completion.json").read_text())
    hashes = {e["run_id"]: e["checkpoint_sha256"] for e in plan["evaluations"]}
    require_gate(gate, plan_digest(plan), plan["source_sha256"], hashes)
    assert completion["all_passed"] and completion["phase"] == "test" and completion["test_labels_opened"]
    require_gate(dict(completion, phase="verify"), plan_digest(plan), plan["source_sha256"], hashes)
    assert completion["test_file_sha256"] == plan["test_file_sha256"]
    pop = read_csv(repo / "results/gowalla/item-popularity.csv")
    tail = {int(r["item_id"]) for r in pop if r["group"] == "tail"}
    n = plan["catalog_size"]
    verified, test_results, differences = {}, {}, []
    for phase in ["verify", "test"]:
        profiles = {r["group"]: r for r in read_csv(root / (phase + "-group-profiles.csv"))}
        assert len(profiles) == 3 and sum(int(r["evaluation_users"]) for r in profiles.values()) == 29858
        for entry in plan["evaluations"]:
            run = entry["run_id"]
            folder = root / phase / run
            metrics = json.loads((folder / "metrics.json").read_text())
            assert metrics["plan_sha256"] == plan_digest(plan)
            assert metrics["source_sha256"] == plan["source_sha256"] and metrics["checkpoint_sha256"] == entry["checkpoint_sha256"]
            assert metrics["training_seed"] == entry["training_seed"] and metrics["checkpoint_epoch"] == entry["checkpoint_epoch"]
            assert metrics["evaluation_split"] == ("validation" if phase == "verify" else "test")
            assert metrics["test_file_sha256"] == (None if phase == "verify" else plan["test_file_sha256"])
            for key in ["fitting_file_sha256", "validation_file_sha256", "popularity_file_sha256"]:
                assert metrics[key] == plan[key]
            assert_reconciles(metrics, metrics["grouped"])
            for group, values in metrics["grouped"].items():
                assert values["evaluation_users"] == int(profiles[group]["evaluation_users"])
                assert values["tail_evaluation_users"] == int(profiles[group]["tail_evaluation_users"])
            counts = exposure_check(folder / "exposure.csv", metrics, tail, n)
            if phase == "verify" or entry["previous_test_metrics"]:
                reference = entry["validation_metrics"] if phase == "verify" else json.loads((repo / entry["previous_test_metrics"]).read_text())
                ref_path = entry["validation_exposure"] if phase == "verify" else entry["previous_test_exposure"]
                expected = read_csv(repo / ref_path)
                l1 = sum(abs(c - int(r["recommendation_count"])) for c, r in zip(counts, expected))
                assert len(expected) == n and l1 <= plan["exposure_l1_tolerance"]
                delta = {k: metrics[k] - reference[k] for k in METRICS}
                assert all(abs(v) <= plan["metric_absolute_tolerance"] for v in delta.values())
                assert l1 == metrics["reproduction_exposure_l1"]
                differences.append(dict(phase=phase, run_id=run, exposure_l1=l1, metric_deltas=delta))
            if phase == "verify": verified[run] = metrics
            else: test_results[run] = metrics
    combined, user_rows = [], []
    for entry in plan["preserved_main"]:
        measured = json.loads((repo / entry["metrics"]).read_text())
        combined.append(dict(model=entry["model"], run_id=entry["run_id"], training_seed=entry["training_seed"],
            evaluation_split="test", result_origin="preserved_original_main", **{k: measured[k] for k in METRICS}))
    old_groups = read_csv(repo / "results/gowalla/user-groups-job340789/per-run.csv")
    assert len(old_groups) == 54
    for row in old_groups:
        user_rows.append(dict(row, result_origin="preserved_original_user_analysis"))
    for entry in plan["evaluations"]:
        measured = test_results[entry["run_id"]]
        if entry["previous_test_metrics"]:
            reference = json.loads((repo / entry["previous_test_metrics"]).read_text())
            origin = "preserved_previous_test"
        else:
            reference, origin = measured, "new_post_hoc_test"
        combined.append(dict(model=entry["model"], run_id=entry["run_id"], training_seed=entry["training_seed"],
            evaluation_split="test", result_origin=origin, **{k: reference[k] for k in METRICS}))
        for group, values in measured["grouped"].items():
            user_rows.append(dict(model=entry["model"], run_id=entry["run_id"], training_seed=entry["training_seed"],
                                  group=group, result_origin="new_post_hoc_group_inference", **values))
    assert len(combined) == 42 and len({r["run_id"] for r in combined}) == 42 and len(user_rows) == 126
    assert sum(r["result_origin"] == "new_post_hoc_test" for r in combined) == 18
    main_summary = aggregate(combined, ["model"])
    user_summary = aggregate(user_rows, ["model", "group"])
    assert len(main_summary) == 84 and len(user_summary) == 252
    idx = {(r["model"], r["metric"]): r for r in main_summary}
    deltas = []
    for model in ORDER:
        row = dict(model=model)
        for metric in METRICS:
            row[metric + "_change"] = idx[model, metric]["mean"] - idx["LightGCN K=3", metric]["mean"]
        deltas.append(row)
    datasets = {"comparison-per-run.csv": combined, "comparison-summary.csv": main_summary,
                "user-groups-per-run.csv": user_rows, "user-groups-summary.csv": user_summary,
                "changes-versus-k3.csv": deltas}
    if args.check:
        for filename, calculated in datasets.items():
            actual = read_csv(root / filename)
            assert len(actual) == len(calculated)
            for observed, expected in zip(actual, calculated):
                assert set(observed) == set(expected)
                for key, value in expected.items():
                    if isinstance(value, (float, int)):
                        assert math.isclose(float(observed[key]), value, abs_tol=1e-12)
                    else: assert observed[key] == value
        with tarfile.open(root / "inference-sources.tar.gz") as archive:
            import hashlib
            for name, expected_hash in plan["source_sha256"].items():
                raw = archive.extractfile("scripts/" + name).read()
                assert hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == expected_hash
            saved = json.loads(archive.extractfile("provenance/extended-test-plan.json").read())
            assert plan_digest(saved) == plan_digest(plan)
        print("PASS: saved 42-run comparison and all seed/group summaries reproduce from audited inputs")
        return
    for filename, values in datasets.items(): write_csv(root / filename, values)
    lines = ["# Matched three-seed test comparison", "", "Post-hoc supplemental extension. Means +/- sample SD across seeds 2020-2022.",
             "Original main and earlier seed-2020 test values remain preserved; 18 missing evaluations were added. All models use the same fitting graph, validation selection and original test split.", "",
             "| Model | Recall@20 | NDCG@20 | Tail share (%) | Coverage (%) | Gini | Tail Recall@20 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model in ORDER:
        cells = []
        for metric in METRICS:
            value = idx[model, metric]
            scale = 100 if metric in ["tail_share_at_20", "coverage_at_20"] else 1
            cells.append(f"{scale*value['mean']:.4f} +/- {scale*value['sample_sd']:.4f}")
        lines.append("| " + model + " | " + " | ".join(cells) + " |")
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    changes = ["# Accuracy cost and tail recovery", "", "Changes relative to standard LightGCN K=3. Recall and exposure changes below are percentage points, computed from matched three-seed test means. These are descriptive comparisons, not significance results or new model selection.", "",
               "| Model | Overall recall change (points) | Tail recall change (points) | Tail-share change (points) |", "| --- | ---: | ---: | ---: |"]
    for row in deltas:
        changes.append(f"| {row['model']} | {100*row['recall_at_20_change']:+.3f} | {100*row['tail_recall_at_20_change']:+.3f} | {100*row['tail_share_at_20_change']:+.3f} |")
    (root / "accuracy-tradeoffs.md").write_text("\n".join(changes) + "\n", encoding="utf-8")
    (root / "artifact-audit.json").write_text(json.dumps(dict(all_passed=True,
        validation_checks=24, supplemental_test_checks=24, new_test_evaluations=18,
        repeated_test_checks=6, preserved_main=18, combined_runs=42, combined_group_rows=126,
        plan_sha256=plan_digest(plan), numerical_reproduction=differences,
        limits="Recall metrics are measured by the checkpoint evaluator. This artifact audit verifies provenance, exposures, group reconciliation and summaries; it does not independently rerank without weights."), indent=2) + "\n")
    with tarfile.open(root / "inference-sources.tar.gz", "w:gz") as archive:
        names = ["scripts/" + n for n in plan["source_sha256"]] + ["provenance/extended-test-plan.json"]
        for name in names:
            if name.startswith("scripts/"):
                assert source_digest(repo / name) == plan["source_sha256"][Path(name).name]
            raw = (repo / name).read_bytes().replace(b"\r\n", b"\n")
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    print("PASS: 24 validation gates, 18 new + 6 repeated tests, exposure metrics, 42-run means and 126 grouped rows")


if __name__ == "__main__":
    main()
