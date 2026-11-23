"""Check that the aggregate audit rejects corrupted summaries and item exports."""
import csv
import gzip
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_single_users_items import audit


def write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    groups = ["head_oriented", "mixed", "tail_oriented"]
    metrics = ["recall_at_20", "ndcg_at_20", "tail_recall_at_20", "tail_share_at_20"]
    write(root / "group-profiles.csv", [dict(group=g, evaluation_users=2, tail_evaluation_users=1) for g in groups])
    evaluations, user_rows, item_rows = [], [], []
    for family in ["single", "uniform"]:
        for layers in range(1, 5):
            for seed in [2020, 2021, 2022]:
                identity = dict(family=family, layers=layers, training_seed=seed, run_id=f"{family}-{layers}-{seed}")
                grouped = {g: dict(evaluation_users=2, tail_evaluation_users=1, recommendation_slots=40, **{m: .5 for m in metrics}) for g in groups}
                overall = dict(evaluation_users=6, tail_evaluation_users=3, recommendation_slots=120, catalog_size=20, **{m: .5 for m in metrics})
                evaluations.append(dict(identity, grouped=grouped, measured=overall, metric_deltas={m: 0 for m in metrics}, exposure_l1_difference=0, group_exposures_reconcile=True))
                user_rows.extend(dict(identity, group=g, **values) for g, values in grouped.items())
                item_rows.append(dict(identity, rho_example=.5))
                with gzip.open(root / (identity["run_id"] + "-items.csv.gz"), "wt", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["item_id", "fitting_degree", "raw_norm", "final_norm", "last_norm", "validation_exposure"])
                    writer.writerows((i, 1, 1, 1, 1, 6) for i in range(20))
    (root / "audit.json").write_text(json.dumps(dict(all_passed=True, evaluation_count=24, evaluation_split="validation", test_labels_opened=False, evaluations=evaluations)))
    for prefix, rows, keys, names in [("user-groups", user_rows, ["family", "layers", "group"], metrics),
                                      ("item-correlations", item_rows, ["family", "layers"], ["rho_example"])]:
        write(root / (prefix + "-per-run.csv"), rows)
        summary = [dict({k: r[k] for k in keys}, metric=m, seed_count=3, mean=.5, sample_sd=0)
                   for r in rows if r["training_seed"] == 2020 for m in names]
        write(root / (prefix + "-summary.csv"), summary)
    audit(root)
    path = root / "user-groups-summary.csv"
    original = path.read_text()
    path.write_text(original.replace(",0.5,0", ",0.7,0", 1))
    try:
        audit(root)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupt summary was accepted")
    path.write_text(original)
    item = root / "single-1-2020-items.csv.gz"
    with gzip.open(item, "rt") as stream:
        original_item = stream.read()
    with gzip.open(item, "wt") as stream:
        stream.write(original_item.replace("0,1,1,1,1,6", "0,1,1,2,1,6", 1))
    try:
        audit(root)
    except AssertionError:
        pass
    else:
        raise AssertionError("A mismatched final-layer-only norm was accepted")
print("PASS: valid exports accepted; corrupt seed summary and single-output norms rejected")
