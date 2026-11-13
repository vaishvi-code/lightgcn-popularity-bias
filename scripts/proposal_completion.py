"""Validation-only follow-up orchestration; never opens original test labels."""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

METRICS = ("recall_at_20", "ndcg_at_20", "tail_share_at_20", "coverage_at_20", "exposure_gini", "tail_recall_at_20")
ARTIFACTS = ("config.json", "history.csv", "run-summary.json", "best-validation-metrics.json", "best-validation-exposure.csv", "followup-provenance.json")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp-" + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def choose_a(rows):
    if not rows or any(not math.isfinite(row["metrics"]["recall_at_20"]) for row in rows):
        raise ValueError("Complete finite validation candidates are required")
    return min(rows, key=lambda row: (-row["metrics"]["recall_at_20"], abs(row["a"] - .5), row["a"]))


def verify_metrics(record, plan):
    metrics = record["metrics"]
    assert metrics["evaluation_split"] == "validation"
    assert metrics["training_seed"] == record["seed"] and metrics["layers"] == record["layers"]
    # The three older standard-normalization exports predate this metadata field.
    normalization = metrics.get("normalization", {"a": .5, "b": .5})
    assert normalization["a"] == record["a"] and normalization["b"] == record["b"]
    assert metrics.get("appnp_beta") == record["beta"]
    if record["kind"] == "single":
        assert metrics.get("layer_combination") == "single"
    for key in ["fitting_file_sha256", "validation_file_sha256"]:
        assert metrics[key] == plan[key]
    assert all(math.isfinite(metrics[key]) for key in METRICS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "run", "select", "summarize"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--group", choices=["grid", "replication", "selected"])
    parser.add_argument("--index", type=int)
    parser.add_argument("--replication-job")
    args = parser.parse_args()
    source, project = args.source.resolve(), args.project.resolve()
    plan_path = source / "provenance/proposal-completion-plan.json"
    plan = json.loads(plan_path.read_text())
    plan_hash = digest(plan_path)
    assert plan["study_phase"] == "post_hoc_proposal_completion_validation"
    assert not plan["test_evaluation_authorized_by_this_plan"]
    python = Path("/mnt/beegfs") / os.environ["USER"] / "conda/envs/lightgcn/bin/python"
    state = source / "state"

    if args.phase == "prepare":
        original = Path("/mnt/beegfs") / os.environ["USER"] / "lightgcn-work"
        project.mkdir(parents=True, exist_ok=False)
        shutil.copytree(source / "scripts", project / "scripts")
        shutil.copytree(source / "tests", project / "tests")
        (project / "LightGCN-PyTorch").symlink_to(original / "LightGCN-PyTorch", target_is_directory=True)
        prepared = project / "prepared/gowalla"
        prepared.mkdir(parents=True)
        for filename, key in [("train.txt", "fitting_file_sha256"), ("val.txt", "validation_file_sha256")]:
            assert digest(original / "prepared/gowalla" / filename) == plan[key]
            shutil.copyfile(original / "prepared/gowalla" / filename, prepared / filename)
        results = project / "results/gowalla"
        results.mkdir(parents=True)
        for name in ["item-popularity.csv", "popularity-summary.json"]:
            shutil.copyfile(source / "results/gowalla" / name, results / name)
        assert digest(results / "item-popularity.csv") == plan["popularity_file_sha256"]
        # Preinstall the existing standard graph cache before parallel training.
        cache_name = "gowalla-validation-" + plan["fitting_file_sha256"][:12] + "-" + plan["validation_file_sha256"][:12]
        shutil.copytree(original / "prepared" / cache_name, project / "prepared" / cache_name)
        assert subprocess.check_output(["git", "-C", str(project / "LightGCN-PyTorch"), "rev-parse", "HEAD"], text=True).strip() == plan["upstream_commit"]
        subprocess.run(["git", "-C", str(project / "LightGCN-PyTorch"), "diff", "--exit-code", "HEAD", "--", "code/model.py", "code/dataloader.py"], check=True)
        for script in ["check_normalization.py", "check_single_lightgcn.py"]:
            subprocess.run([str(python), str(project / "tests" / script), str(project / "LightGCN-PyTorch/code")], check=True)
        for entry in plan["references"]:
            verify_metrics(entry, plan)
        write_json(state / "prepared.json", dict(plan_sha256=plan_hash, project=str(project), verified=True))
        print("PREPARED isolated fitting/validation workspace; numerical model checks passed", flush=True)
        return

    assert json.loads((state / "prepared.json").read_text())["plan_sha256"] == plan_hash
    for name, expected in plan["sources_sha256"].items():
        assert digest(project / "scripts" / name) == digest(source / "scripts" / name) == expected

    if args.phase == "run":
        if args.group == "selected":
            selection = json.loads((state / "normalization-selection.json").read_text())
            assert selection["plan_sha256"] == plan_hash
            task = selection["replication_tasks"][args.index]
        else:
            task = plan["grid_tasks" if args.group == "grid" else "replication_tasks"][args.index]
        identity = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ["SLURM_JOB_ID"]) + "_" + str(args.index)
        run_id = f"followup-{args.group}-{task['kind']}-a{task['a']}-b{task['b']}-k{task['layers']}-beta{task['beta']}-seed{task['seed']}-job{identity}"
        run = project / "results/gowalla" / run_id
        command = [str(python), "-u", str(project / "scripts/train_validation.py"), "--run-dir", str(run),
                   "--norm-a", str(task["a"]), "--norm-b", str(task["b"]), "--dataset", "gowalla", "--model", "lgn",
                   "--layer", str(task["layers"]), "--recdim", "64", "--bpr_batch", "1024", "--lr", "0.001", "--decay", "0.0001",
                   "--epochs", "500", "--early-stopping-patience", "20", "--seed", str(task["seed"]), "--topks", "[20]", "--testbatch", "256", "--tensorboard", "0", "--multicore", "0"]
        if task["kind"] == "single":
            command += ["--layer-combination", "single"]
        if task["beta"] is not None:
            command += ["--appnp-beta", str(task["beta"])]
        subprocess.run(command, cwd=project, check=True)
        subprocess.run([str(python), "-u", str(project / "scripts/evaluate_lightgcn.py"), "--run-dir", str(run)], cwd=project, check=True)
        record = dict(task, run_id=run_id, checkpoint_sha256=digest(run / "best.pt"), metrics=json.loads((run / "best-validation-metrics.json").read_text()))
        verify_metrics(record, plan)
        summary = json.loads((run / "run-summary.json").read_text())
        assert summary["best_epoch"] == record["metrics"]["checkpoint_epoch"]
        provenance = dict(plan_sha256=plan_hash, study_phase=plan["study_phase"], task=task, group=args.group, array_index=args.index, job_id=os.environ["SLURM_JOB_ID"], checkpoint_sha256=record["checkpoint_sha256"], test_labels_opened=False)
        write_json(run / "followup-provenance.json", provenance)
        exported = source / "results/gowalla" / run_id
        exported.mkdir(exist_ok=False)
        for name in ARTIFACTS:
            shutil.copyfile(run / name, exported / name)
        write_json(state / args.group / (str(args.index) + ".json"), record)
        print("COMPLETED and exported " + run_id, flush=True)
        return

    if args.phase == "select":
        new = [json.loads((state / "grid" / (str(index) + ".json")).read_text()) for index in range(6)]
        references = [entry for entry in plan["references"] if entry["kind"] == "normalization"]
        candidates = new + [entry for entry in references if entry["seed"] == 2020]
        assert len(candidates) == 9
        for entry in candidates:
            verify_metrics(entry, plan)
        selected = []
        tasks, reused = [], []
        for b in plan["b_grid"]:
            rows = [entry for entry in candidates if entry["b"] == b]
            assert sorted(row["a"] for row in rows) == plan["a_grid"]
            winner = choose_a(rows)
            selected.append(winner)
            for seed in [2021, 2022]:
                existing = next((entry for entry in references if entry["a"] == winner["a"] and entry["b"] == b and entry["seed"] == seed), None)
                if existing:
                    reused.append(existing)
                else:
                    tasks.append(dict(kind="normalization", a=winner["a"], b=b, layers=3, beta=None, seed=seed))
        write_json(state / "normalization-selection.json", dict(plan_sha256=plan_hash, selected_seed2020=selected, replication_tasks=tasks, reused_replications=reused, selection_rule=plan["selection_rule"], test_labels_opened=False))
        dependency = [args.replication_job]
        if tasks:
            job = subprocess.check_output(["sbatch", "--parsable", "--array=0-" + str(len(tasks) - 1) + "%2", "--export=ALL,FOLLOWUP_GROUP=selected", "--output=" + str(source / "logs/selected-%A_%a.out"), str(source / "scripts/proposal_completion_arc.sh")], text=True).strip().split(";")[0]
            dependency.append(job)
            write_json(state / "selected-submission.json", dict(job_id=job, task_count=len(tasks)))
        job = subprocess.check_output(["sbatch", "--parsable", "--partition=normal", "--dependency=afterok:" + ":".join(dependency), "--export=ALL,FOLLOWUP_PHASE=summarize", "--output=" + str(source / "logs/summary-%j.out"), str(source / "scripts/proposal_completion_control_arc.sh")], text=True).strip().split(";")[0]
        write_json(state / "summary-submission.json", dict(job_id=job, dependencies=dependency))
        print("SELECTED normalization settings from validation; queued replication and summary", flush=True)
        return

    selection = json.loads((state / "normalization-selection.json").read_text())
    records = [json.loads((state / "replication" / (str(index) + ".json")).read_text()) for index in range(12)]
    records += [entry for entry in plan["references"] if entry["kind"] in ["single", "appnp"]]
    records += selection["selected_seed2020"] + selection["reused_replications"]
    records += [json.loads((state / "selected" / (str(index) + ".json")).read_text()) for index in range(len(selection["replication_tasks"]))]
    buckets = defaultdict(list)
    per_run = []
    for entry in records:
        verify_metrics(entry, plan)
        label = f"{entry['kind']}-k{entry['layers']}-a{entry['a']}-b{entry['b']}-beta{entry['beta']}"
        buckets[label].append(entry)
        per_run.append(dict(setting=label, run_id=entry["run_id"], seed=entry["seed"], **{key: entry["metrics"][key] for key in METRICS}))
    aggregate = []
    for label, rows in buckets.items():
        assert sorted(row["seed"] for row in rows) == [2020, 2021, 2022]
        for metric in METRICS:
            values = [row["metrics"][metric] for row in rows]
            aggregate.append(dict(setting=label, metric=metric, seed_count=3, mean=statistics.mean(values), sample_sd=statistics.stdev(values)))
    destination = source / "results/proposal-completion"
    destination.mkdir(exist_ok=False)
    for name, rows in [("validation-per-run.csv", per_run), ("validation-summary.csv", aggregate)]:
        with (destination / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    shutil.copyfile(state / "normalization-selection.json", destination / "normalization-selection.json")
    write_json(destination / "completion.json", dict(all_passed=True, plan_sha256=plan_hash, study_phase=plan["study_phase"], settings=len(buckets), runs=len(records), evaluation_split="validation", test_labels_opened=False))
    print("COMPLETE: three-seed validation summaries exported to " + str(destination), flush=True)


if __name__ == "__main__":
    main()
