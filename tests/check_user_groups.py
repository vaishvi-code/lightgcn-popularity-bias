"""Check exact boundaries and invalid histories without ML dependencies."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from user_groups import assign_group

assert assign_group(0, 3) == "head_oriented"
assert assign_group(1, 3) == "head_oriented"
assert assign_group(1, 2) == "mixed"
assert assign_group(2, 3) == "tail_oriented"
assert assign_group(3, 3) == "tail_oriented"
for counts in [(0, 0), (-1, 2), (3, 2)]:
    try:
        assign_group(*counts)
    except ValueError:
        pass
    else:
        raise AssertionError(counts)
print("User group boundary checks passed")
