"""Dependency-free identity and verification gates for the post-hoc test extension."""
import hashlib
import json


def plan_digest(plan):
    data = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(data).hexdigest()


def source_digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def require_gate(gate, plan_hash, sources, checkpoints):
    if not gate.get("all_passed") or gate.get("phase") != "verify":
        raise ValueError("All frozen validation checks must pass before test access")
    if gate.get("plan_sha256") != plan_hash or gate.get("source_sha256") != sources:
        raise ValueError("Plan or evaluator sources changed after verification")
    rows = gate.get("evaluations", [])
    if len(rows) != len(checkpoints) or len({r["run_id"] for r in rows}) != len(rows):
        raise ValueError("Incomplete or duplicate checkpoint roster")
    observed = {r["run_id"]: r["checkpoint_sha256"] for r in rows if r.get("status") == "passed"}
    if observed != checkpoints:
        raise ValueError("Checkpoint identities differ from the verified roster")


def assert_reconciles(overall, grouped):
    for metric, denominator in [("recall_at_20", "evaluation_users"),
                                 ("ndcg_at_20", "evaluation_users"),
                                 ("tail_recall_at_20", "tail_evaluation_users"),
                                 ("tail_share_at_20", "recommendation_slots")]:
        assert sum(r[denominator] for r in grouped.values()) == overall[denominator]
        value = sum(r[metric] * r[denominator] for r in grouped.values()) / overall[denominator]
        assert abs(value - overall[metric]) <= 1e-12
