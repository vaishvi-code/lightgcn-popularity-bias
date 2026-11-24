"""Freeze the missing test evaluations using existing validation-selected identities."""
import csv
import json
from pathlib import Path
from supplemental_test_gate import source_digest

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    original = read("provenance/final-test-plan.json")
    single = read("provenance/supplemental-single-test-plan.json")
    main_root = Path("results/gowalla/final-test-job337725")
    single_root = Path("results/gowalla/supplemental-single-test-job340785")
    run_root = Path("results/gowalla/proposal-completion-20261114/runs")
    single_roster = read("provenance/single-users-items-plan.json")
    lookup = {e["run_id"]: e for e in single_roster["evaluations"]}
    main_hashes = {e["run_id"]: e["checkpoint_sha256"] for e in read(main_root / "validation-verification.json")["evaluations"]}
    main_entries = {e["run_id"]: e for e in original["evaluations"] if e["kind"] == "checkpoint"}
    with (ROOT / "results/gowalla/proposal-completion-20261114/expanded-validation-per-run.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    plan = {k: original[k] for k in ["upstream_commit", "catalog_size", "original_train_sha256", "fitting_file_sha256", "validation_file_sha256", "popularity_file_sha256"]}
    plan.update(protocol_version=1, frozen_date="2026-11-23", study_phase="post_hoc_extended_test",
        original_test_already_viewed=True, new_test_evaluation_authorized=True,
        training_policy="Reuse existing best-validation checkpoints. No training, tuning, setting selection or epoch extension.",
        new_test_evaluations=18, repeated_test_evaluations_for_user_groups=6,
        top_k=20, graph_source="fitting only", test_exclusions="fitting plus validation",
        grouping="Existing fitting-history thirds, using fixed fitting-degree popular/tail labels.",
        metric_absolute_tolerance=1e-5, exposure_l1_tolerance=20,
        test_file_sha256=single["test_file_sha256"],
        primary_comparison="14 settings x 3 seeds; retain the 18 main frozen test rows and 6 earlier supplemental/exploratory rows, add the 18 missing rows.",
        user_groups="Reuse original 18-main-checkpoint group results; evaluate groups for all 24 extension checkpoints, including 6 earlier test checkpoints. No user IDs, histories or individual recommendation lists exported.",
        interpretation="Post-hoc descriptive extension, not a new blind test or significance analysis. Keep earlier results and disclose the validation-selection and test-access chronology.",
        evaluations=[], preserved_main=[])
    for row in rows:
        if row["model"] == "BPR-MF" or row["model"].startswith("LightGCN K=") or row["model"] == "APPNP beta=0.5":
            old = main_entries[row["run_id"]]
            plan["preserved_main"].append(dict(model=row["model"], run_id=row["run_id"], training_seed=int(row["training_seed"]),
                metrics=(main_root / "test" / row["run_id"] / "metrics.json").as_posix()))
            continue
        run, seed = row["run_id"], int(row["training_seed"])
        if run in lookup:
            old = lookup[run]
            entry = dict(model=row["model"], family="single", layers=old["layers"], training_seed=seed,
                run_id=run, config=old["config"], checkpoint_epoch=old["checkpoint_epoch"], checkpoint_sha256=old["checkpoint_sha256"],
                validation_metrics=old["reference_metrics"], validation_exposure=old["reference_exposure"], beta=None, a=.5, b=.5)
        elif run in main_entries:
            old = main_entries[run]
            entry = dict(model=row["model"], family="appnp", layers=3, training_seed=seed,
                run_id=run, config=old["config"], checkpoint_epoch=old["checkpoint_epoch"], checkpoint_sha256=main_hashes[run],
                validation_metrics=old["validation_metrics"], validation_exposure=(main_root / "verify" / run / "exposure.csv").as_posix(),
                beta=old["appnp_beta"], a=.5, b=.5)
        else:
            root = run_root / run
            metadata = read(root / "config.json")
            metrics = read(root / "best-validation-metrics.json")
            provenance = read(root / "followup-provenance.json")
            entry = dict(model=row["model"], family="appnp" if metadata["appnp_beta"] is not None else "normalization",
                layers=metrics["layers"], training_seed=seed, run_id=run, config=metadata["config"],
                checkpoint_epoch=metrics["checkpoint_epoch"], checkpoint_sha256=provenance["checkpoint_sha256"],
                validation_metrics=metrics, validation_exposure=(root / "best-validation-exposure.csv").as_posix(),
                beta=metadata["appnp_beta"], a=metadata["normalization"]["a"], b=metadata["normalization"]["b"])
        old_test = (single_root if entry["family"] == "single" else main_root) / "test" / run
        entry["previous_test_metrics"] = (old_test / "metrics.json").as_posix() if (ROOT / old_test / "metrics.json").exists() else None
        entry["previous_test_exposure"] = (old_test / "exposure.csv").as_posix() if entry["previous_test_metrics"] else None
        plan["evaluations"].append(entry)
    assert len(plan["evaluations"]) == 24 and len(plan["preserved_main"]) == 18
    assert sum(e["previous_test_metrics"] is None for e in plan["evaluations"]) == 18
    assert len({e["run_id"] for e in plan["evaluations"]}) == 24
    for e in plan["evaluations"]:
        assert (ROOT / e["validation_exposure"]).is_file()
    sources = ["evaluate_extended_test.py", "supplemental_test_gate.py", "final_metrics.py", "user_groups.py", "weighted_lightgcn.py", "single_lightgcn.py", "normalized_graph.py"]
    plan["source_sha256"] = {n: source_digest(ROOT / "scripts" / n) for n in sources}
    path = ROOT / "provenance/extended-test-plan.json"
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print("Frozen 18 new test evaluations and 6 repeated evaluations for matched user-group analysis.")


if __name__ == "__main__":
    main()
