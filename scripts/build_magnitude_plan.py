"""Freeze nine already-selected validation checkpoints for a scoring-only diagnostic."""
import json
from pathlib import Path
from supplemental_test_gate import source_digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    original = json.loads((ROOT / "provenance/final-test-plan.json").read_text())
    verification = json.loads((ROOT / "results/gowalla/final-test-job337725/validation-verification.json").read_text())
    hashes = {r["run_id"]: r["checkpoint_sha256"] for r in verification["evaluations"]}
    single_plan = json.loads((ROOT / "provenance/single-users-items-plan.json").read_text())
    entries = []
    for entry in original["evaluations"]:
        if entry.get("comparison_group") == "main_three_seed" and entry.get("appnp_beta") is None and entry["config"]["lightGCN_n_layers"] in [0, 3]:
            entries.append(dict(model="MF" if entry["config"]["lightGCN_n_layers"] == 0 else "LightGCN K=3", family="uniform",
                layers=entry["config"]["lightGCN_n_layers"], training_seed=entry["training_seed"], run_id=entry["run_id"],
                checkpoint_epoch=entry["checkpoint_epoch"], checkpoint_sha256=hashes[entry["run_id"]], config=entry["config"],
                validation_metrics=entry["validation_metrics"], validation_exposure=f"results/gowalla/final-test-job337725/verify/{entry['run_id']}/exposure.csv"))
    for entry in single_plan["evaluations"]:
        if entry["family"] == "single" and entry["layers"] == 3:
            entries.append(dict(model="LightGCN-single K=3", family="single", layers=3,
                **{k: entry[k] for k in ["training_seed", "run_id", "checkpoint_epoch", "checkpoint_sha256", "config"]},
                validation_metrics=entry["reference_metrics"], validation_exposure=entry["reference_exposure"]))
    assert len(entries) == 9
    assert {(e["model"], e["training_seed"]) for e in entries} == {(m, s) for m in ["MF", "LightGCN K=3", "LightGCN-single K=3"] for s in [2020, 2021, 2022]}
    plan = {k: original[k] for k in ["upstream_commit", "catalog_size", "fitting_file_sha256", "validation_file_sha256", "popularity_file_sha256"]}
    plan.update(frozen_date="2026-11-24", study_phase="post_hoc_validation_item_magnitude", evaluation_split="validation",
        original_test_already_viewed=True, test_evaluation_authorized=False, top_k=20, epsilon=1e-12,
        training_policy="Frozen existing best-validation checkpoints. No retraining, tuning, new selection or test evaluation.",
        verification="First reproduce all nine saved sigmoid-based validation evaluations before intervention access.",
        scoring_modes=["legacy_sigmoid", "raw_dot", "unit_item"],
        primary_comparison="unit_item versus raw_dot on the same frozen user and item vectors; both use raw scores, fitting-only exclusions, same full catalog and top20.",
        legacy_control="legacy_sigmoid reproduces old evaluation and measures numerical ranking sensitivity to the otherwise monotonic sigmoid. It is not the magnitude intervention.",
        intervention="For each final item vector e_i use e_i/max(L2norm(e_i),epsilon), keep final user vectors unchanged. Zero vectors remain zero. This changes scoring, not graph degree normalization.",
        ranking_diagnostics="Mean top20 set overlap, Jaccard similarity, fraction of changed top1/set/ordered lists and mean positional agreement. Export aggregates only, no per-user lists.",
        metric_absolute_tolerance=1e-5, exposure_l1_tolerance=20,
        grouping="Reuse fixed fitting-history thirds for descriptive grouped relevance and exposure; no groups are inferred from validation labels.",
        interpretation="Inference intervention demonstrates ranking sensitivity to magnitude removal, not a causal training explanation or a newly trained model. Three seeds share one split; descriptive SD only.",
        evaluations=entries)
    names=["analyze_magnitude.py","magnitude_scoring.py","final_metrics.py","single_lightgcn.py","user_groups.py","supplemental_test_gate.py"]
    plan["source_sha256"]={n: source_digest(ROOT / "scripts" / n) for n in names}
    (ROOT / "provenance/magnitude-plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=True)+"\n", encoding="utf-8")
    print("Frozen MF, standard K3 and single-output K3 across three seeds; two scoring arms plus a sigmoid numerical control, validation only.")


if __name__ == "__main__": main()
