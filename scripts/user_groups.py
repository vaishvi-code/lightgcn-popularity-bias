"""Fixed observed-history groups, independent of held-out outcomes."""

GROUPS = ("head_oriented", "mixed", "tail_oriented")


def assign_group(tail_count, history_count):
    if history_count <= 0 or not 0 <= tail_count <= history_count:
        raise ValueError("A nonempty valid fitting history is required")
    if 3 * tail_count <= history_count:
        return "head_oriented"
    if 3 * tail_count >= 2 * history_count:
        return "tail_oriented"
    return "mixed"
