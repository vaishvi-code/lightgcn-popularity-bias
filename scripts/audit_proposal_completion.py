"""Audit the exported follow-up artifacts without checkpoints or test labels."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ("recall_at_20", "ndcg_at_20", "tail_share_at_20", "coverage_at_20", "exposure_gini", "tail_recall_at_20")


def load(path):
    return json.loads(path.read_text())


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def canonical_hash(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/gowalla/proposal-completion-20261114"))
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    out = args.results.resolve()
    repo = out.parents[2]
    plan_path = repo / "provenance/proposal-completion-plan.json"
    plan = load(plan_path)
    plan_hash = canonical_hash(plan_path)
    assert canonical_hash(out / "provenance/proposal-completion-plan.json") == plan_hash
    assert plan["original_test_already_viewed"] and not plan["test_evaluation_authorized_by_this_plan"]
    assert load(out / "state/submission.json")["plan_sha256"] == plan_hash
    assert load(out / "state/prepared.json")["plan_sha256"] == plan_hash
    for name, expected in plan["sources_sha256"].items():
        assert canonical_hash(repo / "scripts" / name) == expected, name
    for reference in plan["references"]:
        assert load(repo / "results/gowalla" / reference["run_id"] / "best-validation-metrics.json") == reference["metrics"]
    selection = load(out / "state/normalization-selection.json")
    assert selection == load(out / "normalization-selection.json")
    assert selection["plan_sha256"] == plan_hash and not selection["test_labels_opened"]
    grid = [load(out / "state/grid" / (str(index) + ".json")) for index in range(6)]
    candidates = grid + [row for row in plan["references"] if row["kind"] == "normalization" and row["seed"] == 2020]
    assert len(candidates) == 9
    for b, winner in zip(plan["b_grid"], selection["selected_seed2020"]):
        rows = [row for row in candidates if row["b"] == b]
        assert sorted(row["a"] for row in rows) == plan["a_grid"]
        maximum = max(row["metrics"]["recall_at_20"] for row in rows)
        tied = [row for row in rows if row["metrics"]["recall_at_20"] == maximum]
        expected = sorted(tied, key=lambda row: (abs(row["a"] - .5), row["a"]))[0]
        assert winner == expected
    pop = csv_rows(repo / "results/gowalla/item-popularity.csv")
    n = len(pop)
    tail = {int(row["item_id"]) for row in pop if row["group"] == "tail"}
    completed = []
    stop_reasons = defaultdict(int)
    for group, tasks in [("grid", plan["grid_tasks"]), ("replication", plan["replication_tasks"]), ("selected", selection["replication_tasks"])]:
        assert len(list((out / "state" / group).glob("*.json"))) == len(tasks)
        for index, task in enumerate(tasks):
            record = load(out / "state" / group / (str(index) + ".json"))
            assert all(record[key] == value for key, value in task.items())
            run = out / "runs" / record["run_id"]
            config = load(run / "config.json")
            summary = load(run / "run-summary.json")
            metrics = load(run / "best-validation-metrics.json")
            provenance = load(run / "followup-provenance.json")
            history = csv_rows(run / "history.csv")
            assert record["metrics"] == metrics
            assert metrics["evaluation_split"] == config["evaluation_split"] == "validation"
            assert metrics["training_seed"] == config["training_seed"] == task["seed"]
            assert metrics["normalization"] == config["normalization"]
            assert config["normalization"]["a"] == task["a"] and config["normalization"]["b"] == task["b"]
            assert config.get("appnp_beta") == task["beta"]
            assert config["config"]["lightGCN_n_layers"] == task["layers"]
            assert (config["layer_combination"] == "single") == (task["kind"] == "single")
            assert provenance["plan_sha256"] == plan_hash and provenance["task"] == task
            assert provenance["checkpoint_sha256"] == record["checkpoint_sha256"]
            assert not provenance["test_labels_opened"]
            for key in ["fitting_file_sha256", "validation_file_sha256"]:
                assert config[key] == metrics[key] == plan[key]
            for name, expected in config["script_sha256"].items():
                assert expected == plan["sources_sha256"][name]
            assert config["requested_epochs"] == 500 and config["early_stopping"]["patience"] == 20
            assert [int(row["epoch"]) for row in history] == list(range(1, len(history) + 1))
            best, best_epoch, counter = -math.inf, 0, 0
            for row in history:
                recall = float(row["validation_recall_at_20"])
                assert math.isfinite(recall) and math.isfinite(float(row["validation_ndcg_at_20"]))
                if recall > best:
                    best, best_epoch, counter = recall, int(row["epoch"]), 0
                else:
                    counter += 1
                assert counter <= 20
                if counter == 20:
                    assert int(row["epoch"]) == len(history)
            assert summary["best_epoch"] == metrics["checkpoint_epoch"] == best_epoch
            assert summary["completed_epochs"] == len(history) <= 500
            assert math.isclose(best, metrics["recall_at_20"], rel_tol=0, abs_tol=1e-6)
            assert math.isclose(float(history[best_epoch - 1]["validation_ndcg_at_20"]), metrics["ndcg_at_20"], rel_tol=0, abs_tol=1e-6)
            assert (summary["stop_reason"] == "early_stopping" and counter == 20) or (len(history) == 500 and counter < 20)
            stop_reasons[summary["stop_reason"]] += 1
            exposure = csv_rows(run / "best-validation-exposure.csv")
            assert [int(row["item_id"]) for row in exposure] == list(range(n))
            counts = [int(row["recommendation_count"]) for row in exposure]
            assert min(counts) >= 0
            slots = sum(counts)
            assert slots == metrics["evaluation_users"] * 20 == 597160
            distinct = sum(count > 0 for count in counts)
            assert distinct == metrics["distinct_recommended_items"]
            cumulative = area = 0
            for count in sorted(counts):
                area += 2 * cumulative + count
                cumulative += count
            independent = dict(tail_share_at_20=sum(counts[item] for item in tail) / slots, coverage_at_20=distinct / n, exposure_gini=1 - area / (n * slots))
            for key, value in independent.items():
                assert math.isclose(metrics[key], value, rel_tol=0, abs_tol=1e-12)
            assert metrics["tail_evaluation_users"] == 18987
            completed.append(record)
    records = [row for row in completed if row not in grid]
    records += [row for row in plan["references"] if row["kind"] in ["single", "appnp"]]
    records += selection["selected_seed2020"] + selection["reused_replications"]
    assert len(records) == 27
    buckets = defaultdict(list)
    for record in records:
        label = f"{record['kind']}-k{record['layers']}-a{record['a']}-b{record['b']}-beta{record['beta']}"
        buckets[label].append(record)
    table = csv_rows(out / "validation-per-run.csv")
    assert len(table) == 27
    assert {row["run_id"] for row in table} == {row["run_id"] for row in records}
    for row in table:
        original = next(record for record in records if record["run_id"] == row["run_id"])
        for metric in METRICS:
            assert float(row[metric]) == original["metrics"][metric]
    aggregate = csv_rows(out / "validation-summary.csv")
    assert len(aggregate) == 54 and len(buckets) == 9
    for row in aggregate:
        entries = buckets[row["setting"]]
        assert sorted(record["seed"] for record in entries) == [2020, 2021, 2022]
        values = [record["metrics"][row["metric"]] for record in entries]
        assert math.isclose(float(row["mean"]), statistics.mean(values), rel_tol=0, abs_tol=1e-14)
        assert math.isclose(float(row["sample_sd"]), statistics.stdev(values), rel_tol=0, abs_tol=1e-14)
    completion = load(out / "completion.json")
    assert completion["all_passed"] and completion["runs"] == 27 and completion["settings"] == 9
    assert completion["plan_sha256"] == plan_hash and not completion["test_labels_opened"]
    expected = {record["run_id"]: (record["seed"], record["metrics"]) for record in records}
    original = load(repo / "provenance/final-test-plan.json")
    for entry in original["evaluations"]:
        if entry["comparison_group"] == "main_three_seed":
            expected[entry["run_id"]] = (entry["training_seed"], entry["validation_metrics"])
    expanded = csv_rows(out / "expanded-validation-per-run.csv")
    assert len(expanded) == len(expected) == 42
    assert {row["run_id"] for row in expanded} == set(expected)
    model_groups = defaultdict(list)
    for row in expanded:
        seed, metrics = expected[row["run_id"]]
        assert int(row["training_seed"]) == seed and row["evaluation_split"] == "validation"
        assert all(float(row[key]) == metrics[key] for key in METRICS)
        model_groups[row["model"]].append(row)
    expanded_summary = csv_rows(out / "expanded-validation-summary.csv")
    assert len(expanded_summary) == 84 and len(model_groups) == 14
    for row in expanded_summary:
        runs = model_groups[row["model"]]
        assert sorted(int(run["training_seed"]) for run in runs) == [2020, 2021, 2022]
        values = [float(run[row["metric"]]) for run in runs]
        assert math.isclose(float(row["mean"]), statistics.mean(values), rel_tol=0, abs_tol=1e-14)
        assert math.isclose(float(row["sample_sd"]), statistics.stdev(values), rel_tol=0, abs_tol=1e-14)
    report = dict(all_passed=True, plan_sha256=plan_hash, audited_new_runs=len(completed), exposure_tables=len(completed), grid_candidates=9, summarized_settings=9, summarized_runs=27, summary_rows=54, expanded_models=14, expanded_runs=42, expanded_summary_rows=84, stop_reasons=dict(stop_reasons), evaluation_split="validation", test_labels_opened=False,
                  checks=["Frozen plan, committed sources and 11 reused references match", "Complete validation-only grid and deterministic selection", "Run configuration, split identities, history, first-best epoch and patience", "Independent tail share, coverage and full-catalog Gini from each exposure table", "All three-seed mean and sample-SD summaries reproduced"],
                  limitation="Saved hashes link exported artifacts to ARC checkpoints; weights were not downloaded. Recall/NDCG/tail Recall are saved evaluator outputs, not independently reranked by this audit.")
    if args.write_report:
        (out / "artifact-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
