"""Check that changed or incomplete identity gates and group totals are rejected."""
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from supplemental_test_gate import assert_reconciles, plan_digest, require_gate

checkpoints = {"a": "hash-a", "b": "hash-b"}
sources = {"evaluator": "source"}
valid = dict(phase="verify", all_passed=True, plan_sha256="plan", source_sha256=sources,
             evaluations=[dict(run_id=k, checkpoint_sha256=v, status="passed") for k, v in checkpoints.items()])
require_gate(valid, "plan", sources, checkpoints)
for field, value in [("phase", "test"), ("all_passed", False), ("plan_sha256", "changed"),
                     ("source_sha256", {}), ("evaluations", valid["evaluations"][:1])]:
    changed = copy.deepcopy(valid)
    changed[field] = value
    try:
        require_gate(changed, "plan", sources, checkpoints)
    except ValueError:
        pass
    else:
        raise AssertionError(field)
for change in ["duplicate", "hash", "failed"]:
    changed = copy.deepcopy(valid)
    if change == "duplicate": changed["evaluations"][1]["run_id"] = "a"
    if change == "hash": changed["evaluations"][1]["checkpoint_sha256"] = "wrong"
    if change == "failed": changed["evaluations"][1]["status"] = "failed"
    try:
        require_gate(changed, "plan", sources, checkpoints)
    except ValueError:
        pass
    else:
        raise AssertionError(change)
assert plan_digest({"a": 1, "b": 2}) == plan_digest({"b": 2, "a": 1})
group = dict(evaluation_users=2, tail_evaluation_users=1, recommendation_slots=40,
             recall_at_20=.25, ndcg_at_20=.2, tail_recall_at_20=.1, tail_share_at_20=.3)
overall = dict(group, evaluation_users=4, tail_evaluation_users=2, recommendation_slots=80)
assert_reconciles(overall, {"a": group, "b": group})
try:
    assert_reconciles(dict(overall, tail_recall_at_20=.2), {"a": group, "b": group})
except AssertionError:
    pass
else:
    raise AssertionError("Corrupt tail denominator/pooled metric accepted")
print("PASS: frozen plan/source/checkpoint gate and exact grouped-metric reconciliation")
