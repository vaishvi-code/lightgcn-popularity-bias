"""Reject incomplete or changed supplemental verification gates."""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_supplemental_single import require_gate

checkpoints = {"one": "hash-one", "two": "hash-two"}
sources = {"evaluator": "source-hash"}
gate = {
    "all_passed": True, "plan_sha256": "plan-hash", "source_sha256": sources,
    "evaluations": [{"run_id": name, "status": "passed", "checkpoint_sha256": digest} for name, digest in checkpoints.items()],
}
require_gate(gate, "plan-hash", sources, checkpoints)
bad_gates = []
for key, value in [("all_passed", False), ("plan_sha256", "changed"), ("source_sha256", {}), ("evaluations", gate["evaluations"][:1])]:
    changed = copy.deepcopy(gate)
    changed[key] = value
    bad_gates.append(changed)
changed = copy.deepcopy(gate)
changed["evaluations"][1]["run_id"] = "one"
bad_gates.append(changed)
changed = copy.deepcopy(gate)
changed["evaluations"][0]["checkpoint_sha256"] = "changed"
bad_gates.append(changed)
for bad in bad_gates:
    try:
        require_gate(bad, "plan-hash", sources, checkpoints)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid verification gate accepted")
print("PASS: supplemental gate accepts only complete matching source, plan, roster, and checkpoint identities.")
