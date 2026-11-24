"""Hand-check pooled seed summaries and item exposure calculations."""
import csv
import math
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_extended_test_report import METRICS, aggregate, exposure_check

rows = [dict(model="example", training_seed=seed, **{k: value for k in METRICS})
        for seed, value in zip([2020, 2021, 2022], [.1, .2, .3])]
summary = aggregate(rows, ["model"])
assert len(summary) == 6
assert all(math.isclose(r["mean"], .2) and math.isclose(r["sample_sd"], .1) for r in summary)
for bad in [rows[:2], [rows[0], rows[0], rows[2]]]:
    try:
        aggregate(bad, ["model"])
    except AssertionError:
        pass
    else:
        raise AssertionError("Incomplete or duplicate seeds accepted")
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "exposure.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["item_id", "recommendation_count"])
        writer.writerows(enumerate([1] * 20))
    metrics = dict(evaluation_users=1, recommendation_slots=20, catalog_size=20,
                   distinct_recommended_items=20, tail_share_at_20=.5, coverage_at_20=1, exposure_gini=0)
    exposure_check(path, metrics, set(range(10)), 20)
    for key, value in [("tail_share_at_20", .6), ("coverage_at_20", .9), ("exposure_gini", .1), ("recommendation_slots", 40)]:
        try:
            exposure_check(path, dict(metrics, **{key: value}), set(range(10)), 20)
        except AssertionError:
            pass
        else:
            raise AssertionError("Corrupt exposure metric accepted: " + key)
print("PASS: hand-calculated three-seed sample SD, exact exposures and corrupt-artifact rejection")
